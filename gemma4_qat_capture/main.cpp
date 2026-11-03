// SPDX-License-Identifier: AGPL-3.0-or-later
// Gemma 4 QAT GGUF activation collector using llama.cpp's public cb_eval.
//
// This source belongs ONLY to ara-gemma4-26BA4B-qat. Shared GGUF editing
// and training remain on the generic ara branch.
//
// The graph node kqv_out-<layer> in llama.cpp src/models/gemma4.cpp is the
// input to the attention output projection (wo). We collect its LAST prompt
// token for each example, from the actual Q4_0 GGUF runtime.
//
// We intentionally DO NOT pretend that a post-norm or post-residual output is
// a linear projection's y. The matching y=x@W.T is reconstructed later from
// the SAME Q4_0 tensor bytes and checked in Python.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "chat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct capture_state {
    fs::path out;
    std::set<int> layers;
    std::map<int, int64_t> dims;
    std::map<int, std::vector<float>> current;
    std::string error;
    bool active = false;
};

static bool parse_kqv_layer(const std::string & name, int & layer) {
    const std::string prefix = "kqv_out-";
    if (name.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    const std::string suffix = name.substr(prefix.size());
    if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(),
                                      [](char c) { return c >= '0' && c <= '9'; })) {
        return false;
    }
    try {
        layer = std::stoi(suffix);
    } catch (...) {
        return false;
    }
    return true;
}

static bool capture_eval(ggml_tensor * tensor, bool ask, void * userdata) {
    capture_state & state = *static_cast<capture_state *>(userdata);
    int layer = -1;
    if (!state.active || !parse_kqv_layer(tensor->name, layer) ||
        state.layers.count(layer) == 0) {
        // For ask==false the scheduler interprets false as cancel, not skip.
        return true;
    }
    if (ask) {
        return true;
    }

    // The Gemma 4 graph normally gives [features, tokens] here. Refuse
    // other layouts instead of accidentally training on wrong token/features.
    if (!ggml_is_contiguous(tensor) || tensor->ne[0] < 1 || tensor->ne[1] < 1 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
        (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16)) {
        state.error = "Unexpected kqv_out layout/type for layer " + std::to_string(layer);
        return false;
    }
    if (state.current.count(layer) != 0) {
        state.error = "Duplicate kqv_out callback for layer " + std::to_string(layer);
        return false;
    }
    const int64_t width = tensor->ne[0];
    const int64_t token_pos = tensor->ne[1] - 1;
    if (width > 1'000'000 || token_pos > 65'536) {
        state.error = "Unreasonable feature/token dimension";
        return false;
    }
    auto existing = state.dims.find(layer);
    if (existing != state.dims.end() && existing->second != width) {
        state.error = "Feature width changed across prompts";
        return false;
    }
    state.dims[layer] = width;

    const size_t element_bytes = tensor->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
    const size_t offset = static_cast<size_t>(token_pos) * static_cast<size_t>(width) * element_bytes;
    std::vector<float> values(static_cast<size_t>(width));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, values.data(), offset, values.size() * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> half_values(static_cast<size_t>(width));
        ggml_backend_tensor_get(tensor, half_values.data(), offset,
                                half_values.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < half_values.size(); ++i) {
            values[i] = ggml_fp16_to_fp32(half_values[i]);
        }
    }
    if (!std::all_of(values.begin(), values.end(),
                     [](float v) { return std::isfinite(v); })) {
        state.error = "Non-finite activations at layer " + std::to_string(layer);
        return false;
    }
    state.current.emplace(layer, std::move(values));
    return true;
}

static std::vector<std::string> read_lines(const fs::path & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open prompt file: " + path.string());
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) lines.push_back(std::move(line));
    }
    if (lines.empty()) throw std::runtime_error("Empty prompt file: " + path.string());
    return lines;
}

static std::set<int> parse_layers(const std::string & csv, int n_layers) {
    std::set<int> out;
    size_t start = 0;
    while (start < csv.size()) {
        size_t end = csv.find(',', start);
        if (end == std::string::npos) end = csv.size();
        const std::string field = csv.substr(start, end - start);
        if (field.empty() ||
            !std::all_of(field.begin(), field.end(),
                         [](char c) { return c >= '0' && c <= '9'; })) {
            throw std::runtime_error("Invalid layer list: " + csv);
        }
        const long index = std::stol(field);
        if (index < 0 || index >= n_layers) {
            throw std::runtime_error("Layer out of range: " + field);
        }
        out.insert(static_cast<int>(index));
        start = end + 1;
    }
    if (out.empty()) throw std::runtime_error("Specify --layers, e.g. 0,1,2");
    return out;
}

static std::vector<llama_token> format_and_tokenize(
    const llama_model * model, const common_chat_templates * templates,
    const std::string & system, const std::string & user
) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    if (!system.empty()) inputs.messages.push_back({"system", system});
    inputs.messages.push_back({"user", user});
    const auto params = common_chat_templates_apply(templates, inputs);
    const std::string formatted = params.prompt.str();
    if (formatted.empty()) throw std::runtime_error("Empty rendered chat template");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int count = llama_tokenize(vocab, formatted.data(), formatted.size(), nullptr, 0, true, true);
    const int required = count < 0 ? -count : count;
    if (required <= 0) throw std::runtime_error("Cannot tokenize rendered chat");
    std::vector<llama_token> ids(static_cast<size_t>(required));
    const int actual = llama_tokenize(vocab, formatted.data(), formatted.size(),
                                      ids.data(), required, true, true);
    if (actual <= 0) throw std::runtime_error("Tokenization failed");
    ids.resize(static_cast<size_t>(actual));
    return ids;
}

static void run_group(
    const char * group, const std::vector<std::string> & prompts,
    llama_model * model, llama_context * ctx,
    const common_chat_templates * templates, const std::string & system,
    capture_state & state, int n_ctx, int n_batch
) {
    for (size_t prompt_index = 0; prompt_index < prompts.size(); ++prompt_index) {
        std::vector<llama_token> tokens = format_and_tokenize(
            model, templates, system, prompts[prompt_index]
        );
        if (tokens.size() > static_cast<size_t>(n_ctx)) {
            throw std::runtime_error(
                std::string(group) + " prompt " + std::to_string(prompt_index) +
                " exceeds context length"
            );
        }
        llama_memory_clear(llama_get_memory(ctx), false);
        state.current.clear();
        state.error.clear();
        int pos = 0;
        while (pos < static_cast<int>(tokens.size())) {
            const int count = std::min(n_batch, static_cast<int>(tokens.size()) - pos);
            state.active = pos + count == static_cast<int>(tokens.size());
            llama_batch batch = llama_batch_init(count, 0, 1);
            for (int i = 0; i < count; i++) {
                batch.token[i] = tokens[static_cast<size_t>(pos + i)];
                batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = pos + i == static_cast<int>(tokens.size()) - 1;
            }
            batch.n_tokens = count;
            const int rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (rc != 0 || !state.error.empty()) {
                throw std::runtime_error(
                    std::string(group) + " prompt " + std::to_string(prompt_index) +
                    " failed in llama_decode/cb_eval: " + state.error +
                    " code=" + std::to_string(rc)
                );
            }
            pos += count;
        }
        if (state.current.size() != state.layers.size()) {
            throw std::runtime_error(
                "Missing kqv_out tensor(s) for prompt " + std::to_string(prompt_index) +
                "; requested " + std::to_string(state.layers.size()) +
                " captured " + std::to_string(state.current.size())
            );
        }
        // No partial successful output: use a temporary directory in the
        // caller and publish it only after all groups succeed.
        for (int layer : state.layers) {
            const fs::path file = state.out / ("layer-" + std::to_string(layer) +
                                               "." + group + ".f32");
            std::ofstream out(file, std::ios::binary | std::ios::app);
            if (!out) throw std::runtime_error("Cannot write " + file.string());
            const auto & vector = state.current.at(layer);
            out.write(reinterpret_cast<const char *>(vector.data()),
                      static_cast<std::streamsize>(vector.size() * sizeof(float)));
            if (!out) throw std::runtime_error("Short write to " + file.string());
        }
        std::cerr << group << " " << (prompt_index + 1) << "/" << prompts.size() << "\r";
    }
    std::cerr << "\n";
}

int main(int argc, char ** argv) {
    // Usage: gemma4-qat-capture MODEL.gguf GOOD.txt BAD.txt OUTPUT_DIRECTORY
    //        LAYERS_CSV [n_gpu_layers=999] [n_batch=256] [n_ctx=8192] [system_text=""]
    if (argc < 6 || argc > 10) {
        std::cerr << "Usage: " << argv[0]
                  << " model.gguf good.txt bad.txt output_dir layers_csv"
                  << " [gpu_layers=999] [batch=256] [ctx=8192] [system_text]\n";
        return 2;
    }
    try {
        const fs::path model_path = argv[1];
        const auto good = read_lines(argv[2]);
        const auto bad = read_lines(argv[3]);
        const fs::path output_dir = argv[4];
        if (fs::exists(output_dir)) {
            throw std::runtime_error("Output directory must not exist (no overwrite)");
        }
        const int gpu_layers = argc > 6 ? std::stoi(argv[6]) : 999;
        const int n_batch = argc > 7 ? std::stoi(argv[7]) : 256;
        const int n_ctx = argc > 8 ? std::stoi(argv[8]) : 8192;
        const std::string system = argc > 9 ? argv[9] : "";
        if (n_batch < 1 || n_ctx < n_batch) throw std::runtime_error("Invalid batch/context sizes");

        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = gpu_layers;
        llama_model * model = llama_model_load_from_file(model_path.string().c_str(), model_params);
        if (!model) throw std::runtime_error("Cannot load Gemma 4 GGUF model");
        capture_state state;
        state.layers = parse_layers(argv[5], llama_model_n_layer(model));
        state.out = output_dir;
        // The GGUF carries Google's canonical template. Do not hardcode
        // <start_of_turn> or use llama_chat_apply_template's limited parser.
        auto templates = common_chat_templates_init(model, "");
        if (!templates) throw std::runtime_error("Embedded GGUF chat template unavailable");

        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
        ctx_params.n_batch = static_cast<uint32_t>(n_batch);
        ctx_params.n_ubatch = static_cast<uint32_t>(n_batch);
        ctx_params.cb_eval = capture_eval;
        ctx_params.cb_eval_user_data = &state;
        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) throw std::runtime_error("Cannot initialize llama context with cb_eval");
        fs::create_directories(output_dir);
        try {
            run_group("good", good, model, ctx, templates.get(), system, state, n_ctx, n_batch);
            run_group("bad", bad, model, ctx, templates.get(), system, state, n_ctx, n_batch);
            std::ofstream manifest(output_dir / "capture.txt");
            manifest << "format=gemma4.kqv_out.last_prompt_token.f32.v1\n";
            manifest << "good_count=" << good.size() << "\n";
            manifest << "bad_count=" << bad.size() << "\n";
            for (auto [layer, width] : state.dims) {
                manifest << "layer=" << layer << ",features=" << width << "\n";
            }
            if (!manifest) throw std::runtime_error("Cannot finalize capture manifest");
        } catch (...) {
            llama_free(ctx);
            throw;
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        std::cout << "Gemma 4 QAT GGUF capture complete: " << output_dir << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
