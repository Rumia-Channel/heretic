# SPDX-License-Identifier: AGPL-3.0-or-later
"""Boundary cases for sparse MoE expert input/output samples."""

import pytest
import torch

from heretic.utils import mean_distances_to_knn


def test_knn_caps_k_to_available_expert_samples():
    query = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    reference = torch.tensor([[1.0, 0.0]])
    expected = torch.tensor([1.0, 3.0])
    torch.testing.assert_close(mean_distances_to_knn(query, reference, 15), expected)


@pytest.mark.parametrize(
    "query,reference,k",
    [
        (torch.empty(0, 3), torch.zeros(1, 3), 1),
        (torch.zeros(1, 3), torch.empty(0, 3), 1),
        (torch.zeros(1, 3), torch.zeros(1, 3), 0),
        (torch.zeros(1, 3), torch.zeros(1, 2), 1),
    ],
)
def test_knn_rejects_invalid_samples(query, reference, k):
    with pytest.raises(ValueError):
        mean_distances_to_knn(query, reference, k)
