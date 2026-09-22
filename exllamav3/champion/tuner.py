from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from .hardware import doctor_report
from .policy import (
    candidate_cpu_splits,
    candidate_cpu_threads,
    promote_adaptive,
    select_fastest,
)
from .profile_build import build_expert_stats, write_expert_stats
from .schema import ChampionConfig, ChampionProfile, TrialResult, stable_sha256
from .worker import _load_prompts


RUNTIME_SOURCE_PATHS = (
    "exllamav3/exllamav3_ext/cpu/moe_mul1.cpp",
    "exllamav3/model/moe_cpu_host.py",
    "exllamav3/modules/block_sparse_mlp.py",
    "exllamav3/modules/block_sparse_mlp_cpu.py",
    "exllamav3/version.py",
    "exllamav3/champion/schema.py",
    "exllamav3/champion/cli.py",
    "exllamav3/champion/hardware.py",
    "exllamav3/champion/policy.py",
    "exllamav3/champion/profile_build.py",
    "exllamav3/champion/tuner.py",
    "exllamav3/champion/worker.py",
)


def runtime_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    source_sha256 = {}
    for relative in RUNTIME_SOURCE_PATHS:
        path = root / relative
        source_sha256[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output = True, text = True, timeout = 10, check = True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary"],
            capture_output = True, timeout = 30, check = False,
        ).stdout
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output = True, text = True, timeout = 10, check = False,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        commit, diff, status = None, b"", []
    identity = {
        "commit": commit,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dirty_tracked_paths": status,
        "python": sys.version.split()[0],
        "source_sha256": source_sha256,
    }
    identity["identity_sha256"] = stable_sha256({
        "commit": identity["commit"],
        "dirty_diff_sha256": identity["dirty_diff_sha256"],
        "python": identity["python"],
        "source_sha256": identity["source_sha256"],
    })
    return identity


def _worker_command(
    *,
    model: str,
    config: ChampionConfig,
    mode: str,
    prompt_file: str | None,
    warmup_tokens: int,
    same_prompt_repeat: bool,
    max_prompts: int | None,
    json_out: str,
) -> list[str]:
    command = [
        sys.executable, "-m", "exllamav3.champion.worker",
        "--mode", mode,
        "--model", model,
        "--mcs", str(config.moe_cpu_split),
        "--mct", str(config.moe_cpu_threads),
        "--ctx", str(config.cache_tokens),
        "--cq", str(config.cache_quant),
        "--warmup-tokens", str(warmup_tokens),
        "--json-out", json_out,
    ]
    if config.mtp:
        command.append("--mtp")
    if config.num_draft_tokens:
        command += ["--ndt", str(config.num_draft_tokens)]
    if prompt_file:
        command += ["--prompt-file", prompt_file]
    if max_prompts:
        command += ["--max-prompts", str(max_prompts)]
    if same_prompt_repeat:
        command.append("--same-prompt-repeat")
    return command


def run_worker(
    *,
    model: str,
    config: ChampionConfig,
    trial_id: str,
    output_dir: Path,
    mode: str = "benchmark",
    prompt_file: str | None = None,
    warmup_tokens: int = 512,
    same_prompt_repeat: bool = True,
    max_prompts: int | None = None,
    timeout_seconds: int = 1800,
    capture: bool = False,
    resume_token: str | None = None,
    reuse_existing: bool = False,
) -> tuple[TrialResult, dict[str, Any] | None]:
    output_dir.mkdir(parents = True, exist_ok = True)
    result_path = output_dir / f"{trial_id}.json"
    log_path = output_dir / f"{trial_id}.log"
    manifest_path = output_dir / f"{trial_id}.manifest.json"
    prompt_sha256 = None
    if prompt_file:
        prompt_sha256 = hashlib.sha256(Path(prompt_file).read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "resume_token": resume_token,
        "trial_id": trial_id,
        "config": asdict(config),
        "mode": mode,
        "prompt_sha256": prompt_sha256,
        "warmup_tokens": warmup_tokens,
        "same_prompt_repeat": same_prompt_repeat,
        "max_prompts": max_prompts,
        "capture": capture,
    }
    payload = None
    error = None
    if reuse_existing and resume_token and result_path.is_file() and manifest_path.is_file():
        try:
            saved_manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
            if saved_manifest == manifest:
                payload = json.loads(result_path.read_text(encoding = "utf-8"))
                log_path.write_text("RESUMED: validated existing trial evidence\n", encoding = "utf-8")
        except (OSError, ValueError, json.JSONDecodeError):
            payload = None
    environment = dict(os.environ)
    environment.update(config.runtime_env(output_dir))
    if capture:
        environment.update({
            "EXL3_MOE_CPU_SWAP": "1",
            "EXL3_MOE_CPU_SWAP_SEED": "0",
            "EXL3_MOE_CPU_SWAP_INTERVAL": "1000000000",
        })
        environment.pop("EXL3_MOE_CPU_SPLIT_STATS", None)
    command = _worker_command(
        model = model,
        config = config,
        mode = mode,
        prompt_file = prompt_file,
        warmup_tokens = warmup_tokens,
        same_prompt_repeat = same_prompt_repeat,
        max_prompts = max_prompts,
        json_out = str(result_path),
    )
    started = time.time()
    if payload is None:
        try:
            process = subprocess.run(
                command,
                env = environment,
                capture_output = True,
                text = True,
                timeout = timeout_seconds,
                check = False,
            )
            log_path.write_text(
                "$ " + " ".join(command) + "\n\n[stdout]\n" + (process.stdout or "")
                + "\n[stderr]\n" + (process.stderr or ""),
                encoding = "utf-8",
            )
            payload = json.loads(result_path.read_text(encoding = "utf-8")) if result_path.exists() else None
            error = None if process.returncode == 0 and payload is not None else (
                f"worker exit {process.returncode}; see {log_path.name}"
            )
            if error is None and resume_token:
                manifest_path.write_text(json.dumps(manifest, indent = 2) + "\n", encoding = "utf-8")
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors = "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors = "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            log_path.write_text(
                "$ " + " ".join(command) + f"\n\nTIMEOUT after {timeout_seconds}s\n"
                + stdout + "\n" + stderr,
                encoding = "utf-8",
            )
            payload = None
            error = f"worker timeout after {timeout_seconds}s"
    if payload is None:
        trial = TrialResult(
            trial_id = trial_id,
            config = config,
            aggregate_decode_tok_per_s = 0.0,
            output_hashes = [],
            output_lengths = [],
            no_short_stops = False,
            all_measured_lanes_full_length = False,
            same_process_hash_match = False,
            vram_peak_gb = 1e9,
            host_available_gb_min = 0.0,
            load_seconds = round(time.time() - started, 3),
            error = error,
        )
        return trial, None
    correctness = payload["correctness"]
    resources = payload["resources"]
    trial = TrialResult(
        trial_id = trial_id,
        config = config,
        aggregate_decode_tok_per_s = float(payload["aggregate_decode_tok_per_s"]),
        output_hashes = list(payload["output_hashes"]),
        output_lengths = [int(value) for value in payload["output_lengths"]],
        no_short_stops = bool(correctness["no_short_stops"]),
        all_measured_lanes_full_length = bool(correctness["all_measured_lanes_full_length"]),
        same_process_hash_match = correctness["same_process_hash_match"],
        vram_peak_gb = float(resources["vram_peak_gb"]),
        host_available_gb_min = float(resources["host_available_gb_min"]),
        load_seconds = float(payload["load_seconds"]),
        lane_results = list(payload["results"]),
        error = error,
    )
    return trial, payload


def _fresh_process_hashes_match(trials: list[TrialResult]) -> bool:
    passing = [trial for trial in trials if trial.correctness_passed]
    return len(passing) == len(trials) and all(
        trial.output_hashes == passing[0].output_hashes
        and trial.output_lengths == passing[0].output_lengths
        for trial in passing[1:]
    )


def _trial_outputs_match_reference(reference: TrialResult, trials: list[TrialResult]) -> bool:
    return reference.correctness_passed and all(
        trial.correctness_passed
        and trial.output_hashes == reference.output_hashes
        and trial.output_lengths == reference.output_lengths
        for trial in trials
    )


def _median_trial(trials: list[TrialResult]) -> TrialResult:
    ordered = sorted(trials, key = lambda row: row.aggregate_decode_tok_per_s)
    return ordered[len(ordered) // 2]


def _repeatability(trials: list[TrialResult]) -> dict[str, Any]:
    rates = [trial.aggregate_decode_tok_per_s for trial in trials]
    median = statistics.median(rates)
    deviations = [abs(value - median) for value in rates]
    return {
        "repeats": len(rates),
        "median_tps": median,
        "mad_tps": statistics.median(deviations),
        "fresh_process_hash_match": _fresh_process_hashes_match(trials),
    }


def tune(
    *,
    model: str,
    output_dir: str | Path,
    prompt_file: str | None = None,
    context_tokens: int = 8192,
    cache_quant: int = 3,
    min_vram_headroom_gb: float = 1.5,
    min_host_available_gb: float = 8.0,
    minimum_adaptive_gain_percent: float = 5.0,
    test_mtp: bool = True,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents = True, exist_ok = True)
    doctor = doctor_report(model)
    (target / "doctor.json").write_text(json.dumps(doctor, indent = 2) + "\n", encoding = "utf-8")
    if not doctor["ok"]:
        raise RuntimeError("doctor failed: " + "; ".join(doctor["fatal"]))
    experts = int(doctor["model"].get("num_experts") or 0)
    if experts < 2:
        raise RuntimeError("model config does not identify a routed expert count")
    torch_devices = doctor["hardware"]["torch"].get("devices") or []
    if not torch_devices:
        raise RuntimeError("doctor found no CUDA devices")
    total_vram_gb = float(torch_devices[0]["total_memory"]) / 1e9
    physical_cores = int(doctor["hardware"]["cpu"]["physical_cores"])
    splits = candidate_cpu_splits(experts, total_vram_gb)
    threads = candidate_cpu_threads(physical_cores)
    discovery_threads = min(threads, key = lambda value: abs(value - max(1, round(physical_cores * 0.75))))
    plan = {
        "model": str(Path(model).expanduser().resolve()),
        "output_dir": str(target),
        "moe_cpu_split_candidates": splits,
        "moe_cpu_thread_candidates": threads,
        "discovery_threads": discovery_threads,
        "context_tokens": context_tokens,
        "cache_quant": cache_quant,
        "test_mtp": test_mtp,
    }
    (target / "plan.json").write_text(json.dumps(plan, indent = 2) + "\n", encoding = "utf-8")
    if dry_run:
        return plan

    workload_path = target / "workload-prompts.json"
    workload_rows = _load_prompts(prompt_file)
    workload_path.write_text(json.dumps(workload_rows, indent = 2) + "\n", encoding = "utf-8")
    qualified_prompt_file = str(workload_path)
    runtime_at_start = runtime_identity()
    tune_token = stable_sha256({
        "hardware": doctor["hardware"]["fingerprint"],
        "model": doctor["model"]["fingerprint"],
        "runtime": runtime_at_start["identity_sha256"],
        "workload": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
        "context_tokens": context_tokens,
        "cache_quant": cache_quant,
        "min_vram_headroom_gb": min_vram_headroom_gb,
        "min_host_available_gb": min_host_available_gb,
        "minimum_adaptive_gain_percent": minimum_adaptive_gain_percent,
        "test_mtp": test_mtp,
    })
    worker_resume = {"resume_token": tune_token, "reuse_existing": resume}

    trials = []
    for split in splits:
        config = ChampionConfig(
            moe_cpu_split = split,
            moe_cpu_threads = discovery_threads,
            cache_tokens = context_tokens,
            cache_quant = cache_quant,
        )
        trial, _payload = run_worker(
            model = model, config = config, trial_id = f"split-{split}", output_dir = target / "trials",
            prompt_file = qualified_prompt_file, warmup_tokens = 512, same_prompt_repeat = True, max_prompts = 6,
            **worker_resume,
        )
        trials.append(trial)
    split_winner = select_fastest(
        trials,
        total_vram_gb = total_vram_gb,
        min_vram_headroom_gb = min_vram_headroom_gb,
        min_host_available_gb = min_host_available_gb,
    )

    thread_trials = []
    for thread_count in threads:
        for pin in (0, 1):
            config = replace(
                split_winner.config,
                moe_cpu_threads = thread_count,
                env = {**split_winner.config.env, "EXL3_MOE_CPU_PIN": str(pin)},
            )
            trial, _payload = run_worker(
                model = model,
                config = config,
                trial_id = f"threads-{thread_count}-pin-{pin}",
                output_dir = target / "trials",
                prompt_file = qualified_prompt_file,
                warmup_tokens = 512,
                same_prompt_repeat = True,
                max_prompts = 6,
                **worker_resume,
            )
            thread_trials.append(trial)
            trials.append(trial)
    hardware_winner = select_fastest(
        thread_trials,
        total_vram_gb = total_vram_gb,
        min_vram_headroom_gb = min_vram_headroom_gb,
        min_host_available_gb = min_host_available_gb,
    )

    if test_mtp:
        mtp_trials = []
        for ndt in (2, 3, 4):
            config = replace(hardware_winner.config, mtp = True, num_draft_tokens = ndt)
            trial, _payload = run_worker(
                model = model, config = config, trial_id = f"mtp-{ndt}", output_dir = target / "trials",
                prompt_file = qualified_prompt_file, warmup_tokens = 512, same_prompt_repeat = True, max_prompts = 6,
                **worker_resume,
            )
            mtp_trials.append(trial)
            trials.append(trial)
        try:
            mtp_winner = select_fastest(
                mtp_trials,
                total_vram_gb = total_vram_gb,
                min_vram_headroom_gb = min_vram_headroom_gb,
                min_host_available_gb = min_host_available_gb,
            )
            if mtp_winner.aggregate_decode_tok_per_s > hardware_winner.aggregate_decode_tok_per_s:
                hardware_winner = mtp_winner
        except ValueError:
            pass

    capture_config = replace(hardware_winner.config, placement_mode = "static", expert_stats = None)
    capture_trial, capture_payload = run_worker(
        model = model,
        config = capture_config,
        trial_id = "expert-profile-capture",
        output_dir = target / "trials",
        mode = "capture",
        prompt_file = qualified_prompt_file,
        warmup_tokens = 512,
        same_prompt_repeat = False,
        max_prompts = None,
        capture = True,
        **worker_resume,
    )
    trials.append(capture_trial)
    if not capture_trial.correctness_passed or capture_payload is None:
        raise RuntimeError("expert profile capture failed correctness")
    stats, capture_evidence = build_expert_stats(
        capture_payload["per_prompt_counts"],
        moe_cpu_split = capture_config.moe_cpu_split,
    )
    stats_path = Path(write_expert_stats(target / "expert-stats.json", stats))
    (target / "expert-profile-evidence.json").write_text(
        json.dumps(capture_evidence, indent = 2) + "\n", encoding = "utf-8"
    )

    static_config = replace(capture_config, expert_stats = str(stats_path), placement_mode = "static")
    adaptive_config = replace(static_config, placement_mode = "seeded-adaptive")
    static_trials = []
    adaptive_trials = []
    # Alternate arms to prevent run-order and temperature effects from masquerading as gain.
    for repeat_index in range(3):
        for label, config, bucket in (
            ("static", static_config, static_trials),
            ("adaptive", adaptive_config, adaptive_trials),
        ):
            trial, _payload = run_worker(
                model = model,
                config = config,
                trial_id = f"qualification-{repeat_index + 1}-{label}",
                output_dir = target / "trials",
                prompt_file = qualified_prompt_file,
                warmup_tokens = 512,
                same_prompt_repeat = True,
                max_prompts = None,
                **worker_resume,
            )
            bucket.append(trial)
            trials.append(trial)
    if not _fresh_process_hashes_match(static_trials):
        raise RuntimeError("static profiled placement failed fresh-process deterministic hash gate")
    adaptive_promoted, adaptive_evidence = promote_adaptive(
        static_trials,
        adaptive_trials,
        minimum_gain_percent = minimum_adaptive_gain_percent,
    )
    if adaptive_promoted and not _fresh_process_hashes_match(adaptive_trials):
        adaptive_promoted = False
        adaptive_evidence["rejected_reason"] = "adaptive arm failed fresh-process deterministic hash gate"
    if adaptive_promoted and not _trial_outputs_match_reference(static_trials[0], adaptive_trials):
        adaptive_promoted = False
        adaptive_evidence["rejected_reason"] = "adaptive outputs did not match the static correctness baseline"
    selected_trials = adaptive_trials if adaptive_promoted else static_trials
    selected_mode = "seeded-adaptive" if adaptive_promoted else "static"
    selected = replace(
        adaptive_config if adaptive_promoted else static_config,
        placement_mode = selected_mode,
        expert_stats = "expert-stats.json",
    )
    representative = _median_trial(selected_trials)
    verification = {
        "selected_trial_id": representative.trial_id,
        "selected_mode": selected_mode,
        "static": _repeatability(static_trials),
        "adaptive": _repeatability(adaptive_trials),
        "adaptive_decision": adaptive_evidence,
        "adaptive_promoted": adaptive_promoted,
        "correctness_policy": {
            "warmup_tokens": 512,
            "fresh_process_repeats": 3,
            "same_prompt_repeat": True,
            "full_length_required": True,
        },
    }
    workload = {
        "prompt_file": workload_path.name,
        "prompt_sha256": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
        "source_prompt_file": str(Path(prompt_file).resolve()) if prompt_file else None,
        "capture": capture_evidence,
        "capture_payload_sha256": stable_sha256(capture_payload["per_prompt_counts"]),
    }
    if runtime_identity()["identity_sha256"] != runtime_at_start["identity_sha256"]:
        raise RuntimeError("runtime changed during tuning; results were not sealed")
    profile = ChampionProfile(
        hardware = doctor["hardware"],
        model = doctor["model"],
        runtime = runtime_at_start,
        selected = selected,
        trials = trials,
        verification = verification,
        workload = workload,
    )
    profile_path = target / "champion-profile.json"
    profile.write(profile_path)
    return {
        "profile": str(profile_path),
        "profile_id": profile.profile_id,
        "selected": asdict(selected),
        "verification": verification,
    }


def verify_profile(
    profile_path: str | Path,
    *,
    model: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(profile_path).expanduser().resolve()
    profile = ChampionProfile.read(source)
    model_path = model or profile.model.get("path")
    doctor = doctor_report(model_path)
    if not doctor["ok"]:
        raise RuntimeError("doctor failed: " + "; ".join(doctor["fatal"]))
    if doctor["hardware"]["fingerprint"] != profile.hardware["fingerprint"]:
        raise RuntimeError("hardware fingerprint does not match the profile")
    if doctor["model"]["fingerprint"] != profile.model["fingerprint"]:
        raise RuntimeError("model fingerprint does not match the profile")
    current_runtime = runtime_identity()
    if current_runtime["identity_sha256"] != profile.runtime.get("identity_sha256"):
        raise RuntimeError("runtime identity does not match the profile; re-run tune")
    config = replace(profile.selected, expert_stats = str(source.parent / profile.selected.expert_stats))
    prompt_file = profile.workload.get("prompt_file")
    qualified_prompt_file = str(source.parent / prompt_file) if prompt_file else None
    if qualified_prompt_file:
        actual_prompt_sha = hashlib.sha256(Path(qualified_prompt_file).read_bytes()).hexdigest()
        if actual_prompt_sha != profile.workload.get("prompt_sha256"):
            raise RuntimeError("qualified workload prompt snapshot does not match the profile")
    trials = []
    receipt_dir = source.parent / "verification"
    for index in range(3):
        trial, _payload = run_worker(
            model = model_path,
            config = config,
            trial_id = f"verify-{index + 1}",
            output_dir = receipt_dir,
            prompt_file = qualified_prompt_file,
            warmup_tokens = 512,
            same_prompt_repeat = True,
        )
        trials.append(trial)
    receipt = {
        "profile_id": profile.profile_id,
        "runtime_identity": current_runtime,
        "ok": _fresh_process_hashes_match(trials),
        "repeatability": _repeatability(trials),
        "trials": [asdict(trial) for trial in trials],
    }
    target = Path(output_path) if output_path else source.with_suffix(".verification.json")
    target.write_text(json.dumps(receipt, indent = 2) + "\n", encoding = "utf-8")
    if not receipt["ok"]:
        raise RuntimeError(f"verification failed; receipt written to {target}")
    return receipt
