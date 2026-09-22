from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Any


DEFAULT_PROMPTS = [
    {"lane": "reasoning", "prompt": "What is 17*23? Show the multiplication steps.", "max_tokens": 128},
    {"lane": "prose", "prompt": "Explain in detail how a refrigerator works.", "max_tokens": 128},
    {"lane": "prose", "prompt": "Write a short story about a lighthouse keeper.", "max_tokens": 128},
    {"lane": "prose", "prompt": "Summarize the main causes of the French Revolution.", "max_tokens": 128},
    {"lane": "code", "prompt": "Write a Python function that reverses a singly linked list.", "max_tokens": 128},
    {"lane": "reasoning", "prompt": "A bat and ball cost $1.10 total. The bat costs $1 more. Explain the ball price.", "max_tokens": 128},
    {"lane": "code", "prompt": "Implement binary search in Python and explain its complexity.", "max_tokens": 128},
    {"lane": "prose", "prompt": "Explain why seasons occur on Earth without using equations.", "max_tokens": 128},
    {"lane": "reasoning", "prompt": "If five machines make five parts in five minutes, how long for 100 machines to make 100 parts?", "max_tokens": 128},
    {"lane": "code", "prompt": "Write a Python context manager that measures elapsed wall time.", "max_tokens": 128},
    {"lane": "prose", "prompt": "Describe how a bill becomes law in the United States.", "max_tokens": 128},
    {"lane": "reasoning", "prompt": "A train travels 120 km in 90 minutes. Explain its average speed in km/h.", "max_tokens": 128},
]


def _rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _host_memory(worker_pids: list[int]) -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            fields = raw.split()
            if fields and fields[0].isdigit():
                values[key] = int(fields[0]) * 1024
    available = values.get("MemAvailable", 0)
    return {
        "available_gb": round(available / 1e9, 3),
        "parent_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
        "worker_rss_gb": round(sum(_rss_bytes(pid) for pid in worker_pids) / 1e9, 3),
    }


def _gpu_memory(torch) -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "used_gb": round((total - free) / 1e9, 3),
        "total_gb": round(total / 1e9, 3),
        "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
    }


def _worker_pids(*configs) -> list[int]:
    pids = []
    for config in configs:
        for host in (getattr(config, "moe_cpu_hosts", None) or {}).values():
            process = getattr(host, "proc", None)
            pid = getattr(process, "pid", None)
            if pid and int(pid) not in pids:
                pids.append(int(pid))
    return pids


def _load_prompts(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    source = Path(path)
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding = "utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(source.read_text(encoding = "utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("prompt file must contain a non-empty JSON array or JSONL rows")
    normalized = []
    for row in rows:
        prompt = str(row["prompt"])
        tokens = int(row.get("max_tokens", 128))
        if tokens < 2:
            raise ValueError("every prompt must request at least two output tokens")
        normalized.append({"lane": str(row.get("lane", "custom")), "prompt": prompt, "max_tokens": tokens})
    return normalized


def _render_chat_ids(tokenizer, prompt: str, thinking: bool):
    ids = tokenizer.hf_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt = True,
        enable_thinking = bool(thinking),
    )
    return ids.unsqueeze(0) if ids.dim() == 1 else ids


def _run_generation(generator, tokenizer, prompt, max_tokens, sampler, stop_ids, thinking, seed):
    from exllamav3 import Job

    input_ids = _render_chat_ids(tokenizer, prompt, thinking)
    job = Job(
        input_ids = input_ids,
        max_new_tokens = max_tokens,
        sampler = sampler,
        stop_conditions = stop_ids,
        seed = seed,
    )
    serial = generator.enqueue(job)
    token_ids = []
    final = None
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if result.get("serial") != serial:
                continue
            if result.get("stage") == "error":
                raise RuntimeError(result.get("error"))
            if result.get("stage") == "streaming" and result.get("token_ids") is not None:
                token_ids.extend(int(value) for value in result["token_ids"].reshape(-1).tolist())
            if result.get("eos"):
                final = result
    final = final or {}
    held = final.get("held") or {}
    if held.get("token_ids") is not None:
        token_ids.extend(int(value) for value in held["token_ids"].reshape(-1).tolist())
    new_tokens = int(final.get("new_tokens") or len(token_ids))
    decode_s = float(final.get("time_generate") or 0.0)
    decode_tokens = max(new_tokens - 1, 0)
    return {
        "prompt_tokens": int(input_ids.shape[-1]),
        "new_tokens": new_tokens,
        "max_tokens": max_tokens,
        "decode_seconds": decode_s,
        "decode_tok_per_s": round(decode_tokens / decode_s, 3) if decode_s > 0 else 0.0,
        "prefill_seconds": round(float(final.get("time_prefill") or 0.0), 4),
        "eos_reason": final.get("eos_reason"),
        "output_sha256": hashlib.sha256(
            ",".join(str(value) for value in token_ids).encode("utf-8")
        ).hexdigest(),
        "token_ids": token_ids,
    }


def _split_modules(model):
    return [
        module for module in model
        if getattr(module, "cpu_split_first", None) is not None
        and getattr(module, "_split_hist", None) is not None
    ]


def _capture_counts(modules) -> dict[str, list[float]]:
    return {
        module.key: [float(value) for value in module._split_hist.detach().cpu().tolist()]
        for module in modules
    }


def _zero_counts(modules):
    import torch
    with torch.inference_mode():
        for module in modules:
            module._split_hist.zero_()


def run(args) -> dict[str, Any]:
    import torch
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator.sampler.presets import ArgmaxSampler

    prompts = _load_prompts(args.prompt_file)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    config = Config.from_directory(args.model)
    config.infer_params.moe_cpu_split = args.mcs
    config.infer_params.moe_cpu_threads = args.mct
    config.infer_params.ngram_stream_from_disk = True

    draft_model = Model.from_config(config, component = "mtp") if args.mtp else None
    max_history = max(
        draft_model.caps.get("default_draft_size", 4) if draft_model else 0,
        args.ndt or 0,
    )
    model = Model.from_config(config)
    cache = Cache(
        model,
        max_num_tokens = args.ctx,
        layer_type = CacheLayer_quant,
        k_bits = args.cq,
        v_bits = args.cq,
        max_history = max_history,
        max_batch_size = 1,
    )
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    # Autosplit probes one loader chunk through the model. Its default is 4096 tokens, which
    # is invalid when a qualification smoke intentionally uses a smaller cache.
    load_chunk = min(args.ctx, 4096)
    model.load(progressbar = False, max_chunk_size = load_chunk)
    draft_cache = None
    if draft_model is not None:
        draft_cache = Cache(
            draft_model,
            max_num_tokens = args.ctx,
            layer_type = CacheLayer_quant,
            k_bits = args.cq,
            v_bits = args.cq,
            max_batch_size = 1,
        )
        draft_model.load(progressbar = False, max_chunk_size = load_chunk)
    load_seconds = time.perf_counter() - load_start
    tokenizer = Tokenizer.from_config(config)
    generator_kwargs = {}
    if draft_model is not None:
        generator_kwargs.update({"draft_model": draft_model, "draft_cache": draft_cache})
        if args.ndt:
            generator_kwargs["num_draft_tokens"] = args.ndt
    generator = Generator(model, cache, tokenizer, **generator_kwargs)
    sampler = ArgmaxSampler()
    stop_ids = list(getattr(model.config, "eos_token_id_list", None) or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_ids:
        stop_ids.append(tokenizer.eos_token_id)
    pids = _worker_pids(config, getattr(model, "config", None))
    host_samples = [_host_memory(pids)]
    gpu_samples = [_gpu_memory(torch)]

    warmup_remaining = args.warmup_tokens
    warmup_results = []
    warmup_index = 0
    while warmup_remaining > 0:
        row = prompts[warmup_index % len(prompts)]
        limit = min(int(row["max_tokens"]), warmup_remaining)
        result = _run_generation(
            generator, tokenizer, row["prompt"], limit, sampler, stop_ids, args.thinking, args.seed
        )
        warmup_results.append(result)
        warmup_remaining -= limit
        warmup_index += 1
        host_samples.append(_host_memory(pids))
        gpu_samples.append(_gpu_memory(torch))

    modules = _split_modules(model)
    if args.mode == "capture" and not modules:
        raise RuntimeError("capture mode requires dynamic CPU-split modules with routing histograms")
    per_prompt_counts = []
    results = []
    repeats = []
    for prompt_index, row in enumerate(prompts):
        if args.mode == "capture":
            _zero_counts(modules)
        first = _run_generation(
            generator, tokenizer, row["prompt"], int(row["max_tokens"]), sampler, stop_ids,
            args.thinking, args.seed,
        )
        first.update({"lane": row["lane"], "prompt_index": prompt_index, "prompt": row["prompt"]})
        results.append(first)
        if args.mode == "capture":
            per_prompt_counts.append({
                "prompt_index": prompt_index,
                "lane": row["lane"],
                "counts": _capture_counts(modules),
            })
        if args.same_prompt_repeat:
            second = _run_generation(
                generator, tokenizer, row["prompt"], int(row["max_tokens"]), sampler, stop_ids,
                args.thinking, args.seed,
            )
            second.update({"lane": row["lane"], "prompt_index": prompt_index, "prompt": row["prompt"]})
            repeats.append(second)
        host_samples.append(_host_memory(pids))
        gpu_samples.append(_gpu_memory(torch))

    usable = [row for row in results if row["new_tokens"] > 1 and row["decode_seconds"] > 0]
    decode_tokens = sum(row["new_tokens"] - 1 for row in usable)
    decode_seconds = sum(row["decode_seconds"] for row in usable)
    same_hashes = None
    if args.same_prompt_repeat:
        same_hashes = len(results) == len(repeats) and all(
            first["output_sha256"] == second["output_sha256"]
            for first, second in zip(results, repeats)
        )
    all_rows = warmup_results + results
    return {
        "mode": args.mode,
        "config": {
            "mcs": args.mcs,
            "mct": args.mct,
            "ctx": args.ctx,
            "cq": args.cq,
            "mtp": args.mtp,
            "ndt": args.ndt,
            "warmup_tokens": args.warmup_tokens,
        },
        "load_seconds": round(load_seconds, 3),
        "aggregate_decode_tok_per_s": round(decode_tokens / decode_seconds, 3) if decode_seconds else 0.0,
        "correctness": {
            "no_short_stops": all(row["new_tokens"] == row["max_tokens"] for row in all_rows),
            "all_measured_lanes_full_length": all(
                row["new_tokens"] == row["max_tokens"] for row in results
            ),
            "same_process_hash_match": same_hashes,
        },
        "resources": {
            "vram_peak_gb": max(row["used_gb"] for row in gpu_samples),
            "vram_total_gb": max(row["total_gb"] for row in gpu_samples),
            "torch_peak_reserved_gb": max(row["torch_peak_reserved_gb"] for row in gpu_samples),
            "host_available_gb_min": min(row["available_gb"] for row in host_samples),
            "worker_rss_gb_max": max(row["worker_rss_gb"] for row in host_samples),
            "parent_peak_rss_gb": max(row["parent_peak_rss_gb"] for row in host_samples),
        },
        "output_hashes": [row["output_sha256"] for row in results],
        "output_lengths": [row["new_tokens"] for row in results],
        "warmup": warmup_results,
        "results": results,
        "same_prompt_repeats": repeats,
        "per_prompt_counts": per_prompt_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Internal fresh-process Champion benchmark worker")
    parser.add_argument("--mode", choices = ("benchmark", "capture"), default = "benchmark")
    parser.add_argument("--model", required = True)
    parser.add_argument("--mcs", type = int, required = True)
    parser.add_argument("--mct", type = int, required = True)
    parser.add_argument("--ctx", type = int, default = 8192)
    parser.add_argument("--cq", type = int, default = 3)
    parser.add_argument("--mtp", action = "store_true")
    parser.add_argument("--ndt", type = int)
    parser.add_argument("--warmup-tokens", type = int, default = 512)
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-prompts", type = int)
    parser.add_argument("--same-prompt-repeat", action = "store_true")
    parser.add_argument("--thinking", action = "store_true")
    parser.add_argument("--seed", type = int, default = 123456)
    parser.add_argument("--json-out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    rendered = json.dumps(result, separators = (",", ":"))
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_text(json.dumps(result, indent = 2) + "\n", encoding = "utf-8")
        rendered = json.dumps({
            "mode": result["mode"],
            "config": result["config"],
            "load_seconds": result["load_seconds"],
            "aggregate_decode_tok_per_s": result["aggregate_decode_tok_per_s"],
            "correctness": result["correctness"],
            "resources": result["resources"],
            "output_hashes": result["output_hashes"],
            "output_lengths": result["output_lengths"],
            "json_out": str(target.resolve()),
        }, separators = (",", ":"))
    print("CHAMPION_RESULT_JSON:" + rendered, flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
