from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

from .schema import ChampionConfig, TrialResult


def candidate_cpu_threads(physical_cores: int) -> list[int]:
    cores = max(1, int(physical_cores))
    values = {
        max(1, cores // 4),
        max(1, cores // 2),
        max(1, (3 * cores) // 4),
        min(cores, 16),
    }
    return sorted(values)


def candidate_cpu_splits(num_experts: int, vram_gib: float) -> list[int]:
    """Return safe-first CPU expert counts for the broad discovery sweep.

    The initial resident estimate is deliberately conservative. Actual selection is based on
    measured VRAM/RAM headroom, and later refinement brackets the fastest passing candidate.
    """
    experts = int(num_experts)
    if experts < 2:
        raise ValueError("num_experts must be at least 2")
    resident = max(16, min(experts - 1, int(max(vram_gib - 2.0, 2.0) * 8)))
    center = experts - resident
    step = max(8, int(math.ceil(experts / 32 / 8) * 8))
    values = {
        min(experts - 1, max(1, center + delta * step))
        for delta in (2, 1, 0, -1, -2)
    }
    return sorted(values, reverse = True)


def trial_is_admissible(
    trial: TrialResult,
    *,
    total_vram_gb: float,
    min_vram_headroom_gb: float = 1.0,
    min_host_available_gb: float = 8.0,
) -> bool:
    return (
        trial.correctness_passed
        and trial.vram_peak_gb <= total_vram_gb - min_vram_headroom_gb
        and trial.host_available_gb_min >= min_host_available_gb
    )


def select_fastest(
    trials: Iterable[TrialResult],
    *,
    total_vram_gb: float,
    min_vram_headroom_gb: float = 1.0,
    min_host_available_gb: float = 8.0,
) -> TrialResult:
    admitted = [
        trial for trial in trials
        if trial_is_admissible(
            trial,
            total_vram_gb = total_vram_gb,
            min_vram_headroom_gb = min_vram_headroom_gb,
            min_host_available_gb = min_host_available_gb,
        )
    ]
    if not admitted:
        raise ValueError("no candidate passed correctness and memory headroom gates")
    return max(admitted, key = lambda trial: trial.aggregate_decode_tok_per_s)


def adaptive_gain_percent(static: TrialResult, adaptive: TrialResult) -> float:
    if static.aggregate_decode_tok_per_s <= 0:
        raise ValueError("static throughput must be positive")
    return 100.0 * (
        adaptive.aggregate_decode_tok_per_s / static.aggregate_decode_tok_per_s - 1.0
    )


def promote_adaptive(
    static_trials: Iterable[TrialResult],
    adaptive_trials: Iterable[TrialResult],
    *,
    minimum_gain_percent: float = 5.0,
) -> tuple[bool, dict[str, Any]]:
    static_rows = list(static_trials)
    adaptive_rows = list(adaptive_trials)
    if len(static_rows) < 3 or len(adaptive_rows) < 3:
        return False, {"reason": "three fresh-process repeats per arm are required"}
    if not all(row.correctness_passed for row in static_rows + adaptive_rows):
        return False, {"reason": "one or more correctness gates failed"}
    static_rates = sorted(row.aggregate_decode_tok_per_s for row in static_rows)
    adaptive_rates = sorted(row.aggregate_decode_tok_per_s for row in adaptive_rows)
    static_median = static_rates[len(static_rates) // 2]
    adaptive_median = adaptive_rates[len(adaptive_rates) // 2]
    gain = 100.0 * (adaptive_median / static_median - 1.0)
    return gain >= minimum_gain_percent, {
        "static_median_tps": static_median,
        "adaptive_median_tps": adaptive_median,
        "gain_percent": gain,
        "minimum_gain_percent": minimum_gain_percent,
    }
