import math
from types import SimpleNamespace

import pytest

from exllamav3.modules.block_sparse_mlp_cpu import BlockSparseMLP_CPU, _profile_permutation


def test_profile_permutation_is_stable_hot_to_cold():
    assert _profile_permutation([2, 9, 9, 1], 4) == [1, 2, 0, 3]


@pytest.mark.parametrize("counts", (
    None,
    [1, 2, 3],
    [1, -1, 2, 3],
    [1, math.inf, 2, 3],
    [1, math.nan, 2, 3],
))
def test_profile_permutation_rejects_incomplete_or_invalid_counts(counts):
    with pytest.raises((TypeError, ValueError)):
        _profile_permutation(counts, 4)


def test_profiled_split_disables_deferred_router_load(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_CPU_SPLIT_STATS", "/tmp/profile.json")
    module = BlockSparseMLP_CPU.__new__(BlockSparseMLP_CPU)
    module.config = SimpleNamespace(infer_params = SimpleNamespace(moe_cpu_split = 2))
    assert module.can_defer_load() is False
