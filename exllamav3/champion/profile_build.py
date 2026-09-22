from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any


def _sum_vectors(vectors: Iterable[list[float]]) -> list[float]:
    rows = list(vectors)
    if not rows:
        return []
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("expert count vectors have inconsistent widths")
    return [sum(row[index] for row in rows) for index in range(width)]


def build_expert_stats(
    per_prompt_counts: list[dict[str, Any]],
    *,
    moe_cpu_split: int,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    if len(per_prompt_counts) < 8:
        raise ValueError("expert profiling requires at least eight prompt windows")
    train_rows = per_prompt_counts[::2]
    heldout_rows = per_prompt_counts[1::2]
    layer_names = sorted(set.intersection(*[set(row["counts"]) for row in per_prompt_counts]))
    if not layer_names:
        raise ValueError("profile capture contains no common MoE layers")
    stats = {}
    train_hot = 0.0
    train_total = 0.0
    heldout_hot = 0.0
    heldout_total = 0.0
    oracle_hot = 0.0
    for layer in layer_names:
        train = _sum_vectors(row["counts"][layer] for row in train_rows)
        heldout = _sum_vectors(row["counts"][layer] for row in heldout_rows)
        if not train or len(train) != len(heldout):
            raise ValueError(f"invalid expert vectors for layer {layer}")
        gpu_slots = len(train) - int(moe_cpu_split)
        if not 0 < gpu_slots < len(train):
            raise ValueError("moe_cpu_split leaves no valid GPU/CPU partition")
        hot = sorted(range(len(train)), key = lambda expert: (-train[expert], expert))[:gpu_slots]
        oracle = sorted(range(len(heldout)), key = lambda expert: (-heldout[expert], expert))[:gpu_slots]
        train_hot += sum(train[expert] for expert in hot)
        train_total += sum(train)
        heldout_hot += sum(heldout[expert] for expert in hot)
        heldout_total += sum(heldout)
        oracle_hot += sum(heldout[expert] for expert in oracle)
        stats[layer] = train
    experts = len(next(iter(stats.values())))
    uniform = (experts - moe_cpu_split) / experts
    train_capture = train_hot / train_total if train_total else 0.0
    heldout_capture = heldout_hot / heldout_total if heldout_total else 0.0
    oracle_capture = oracle_hot / heldout_total if heldout_total else 0.0
    warnings = []
    if heldout_capture <= uniform + 0.03:
        warnings.append("held-out capture is within three percentage points of uniform placement")
    if train_capture - heldout_capture > 0.25:
        warnings.append("in-sample capture exceeds held-out capture by more than 25 points")
    evidence = {
        "windows": len(per_prompt_counts),
        "train_windows": len(train_rows),
        "heldout_windows": len(heldout_rows),
        "layers": len(layer_names),
        "experts_per_layer": experts,
        "gpu_slots_per_layer": experts - moe_cpu_split,
        "uniform_capture": round(uniform, 6),
        "train_capture": round(train_capture, 6),
        "heldout_capture": round(heldout_capture, 6),
        "oracle_capture": round(oracle_capture, 6),
        "warnings": warnings,
    }
    return stats, evidence


def write_expert_stats(path: str | Path, stats: dict[str, list[float]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents = True, exist_ok = True)
    target.write_text(json.dumps(stats, indent = 2) + "\n", encoding = "utf-8")
    return str(target.resolve())
