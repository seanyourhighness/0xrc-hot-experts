import pytest

from exllamav3.champion.profile_build import build_expert_stats


def test_profile_uses_disjoint_train_and_heldout_windows():
    rows = []
    for index in range(8):
        # Training windows (even) prefer experts 0/1; held-out windows preserve that ranking.
        counts = [10.0, 8.0, 1.0, 1.0] if index % 2 == 0 else [7.0, 6.0, 2.0, 1.0]
        rows.append({"counts": {"layer.0": counts, "layer.1": counts}})
    stats, evidence = build_expert_stats(rows, moe_cpu_split = 2)
    assert stats["layer.0"] == [40.0, 32.0, 4.0, 4.0]
    assert evidence["heldout_capture"] > evidence["uniform_capture"]
    assert evidence["train_windows"] == evidence["heldout_windows"] == 4


def test_profile_warns_on_non_generalizing_ranking():
    rows = []
    for index in range(8):
        counts = [10.0, 9.0, 0.0, 0.0] if index % 2 == 0 else [0.0, 0.0, 10.0, 9.0]
        rows.append({"counts": {"layer.0": counts}})
    _stats, evidence = build_expert_stats(rows, moe_cpu_split = 2)
    assert evidence["warnings"]


def test_profile_refuses_too_few_windows():
    with pytest.raises(ValueError, match = "at least eight"):
        build_expert_stats([{"counts": {"layer.0": [1.0, 2.0]}}] * 7, moe_cpu_split = 1)
