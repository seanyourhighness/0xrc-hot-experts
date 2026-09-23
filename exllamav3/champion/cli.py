from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from .hardware import doctor_report
from .schema import ChampionProfile
from .tuner import runtime_identity, tune, verify_profile


def _write_json(value, path: str | None):
    rendered = json.dumps(value, indent = 2) + "\n"
    if path:
        target = Path(path)
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_text(rendered, encoding = "utf-8")
    print(rendered, end = "")


def command_doctor(args) -> int:
    report = doctor_report(args.model)
    _write_json(report, args.json_out)
    return 0 if report["ok"] else 2


def command_profile_check(args) -> int:
    profile = ChampionProfile.read(args.profile)
    print(json.dumps({
        "ok": True,
        "profile_id": profile.profile_id,
        "hardware_fingerprint": profile.hardware["fingerprint"],
        "model_fingerprint": profile.model["fingerprint"],
        "selected": profile.payload()["selected"],
    }, indent = 2))
    return 0


def command_tune(args) -> int:
    result = tune(
        model = args.model,
        output_dir = args.output,
        prompt_file = args.prompts,
        context_tokens = args.context,
        cache_quant = args.cache_quant,
        min_vram_headroom_gb = args.min_vram_headroom,
        min_host_available_gb = args.min_host_available,
        minimum_adaptive_gain_percent = args.min_adaptive_gain,
        test_mtp = not args.no_mtp,
        dry_run = args.dry_run,
        resume = args.resume,
    )
    print(json.dumps(result, indent = 2))
    return 0


def command_verify(args) -> int:
    result = verify_profile(args.profile, model = args.model, output_path = args.json_out)
    print(json.dumps(result, indent = 2))
    return 0


def command_serve(args) -> int:
    profile_path = Path(args.profile).expanduser().resolve()
    profile = ChampionProfile.read(profile_path)
    report = doctor_report(args.model or profile.model.get("path"))
    if not report["ok"]:
        raise SystemExit("doctor failed: " + "; ".join(report["fatal"]))
    if not args.allow_hardware_mismatch and report["hardware"]["fingerprint"] != profile.hardware["fingerprint"]:
        raise SystemExit("hardware fingerprint mismatch; re-run tune for this machine")
    if report["model"]["fingerprint"] != profile.model["fingerprint"]:
        raise SystemExit("model fingerprint mismatch; re-run tune for this checkpoint")
    current_runtime = runtime_identity()
    if current_runtime["identity_sha256"] != profile.runtime.get("identity_sha256"):
        raise SystemExit("runtime identity mismatch; re-run tune and verify with this installation")
    receipt_path = Path(args.verification).expanduser().resolve() if args.verification else \
        profile_path.with_suffix(".verification.json")
    if not args.allow_unverified:
        try:
            receipt = json.loads(receipt_path.read_text(encoding = "utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"verified receipt required at {receipt_path}; run champion verify first"
            ) from exc
        if not receipt.get("ok") or receipt.get("profile_id") != profile.profile_id:
            raise SystemExit("verification receipt is not valid for this profile")
        receipt_runtime = receipt.get("runtime_identity", {}).get("identity_sha256")
        if receipt_runtime != current_runtime["identity_sha256"]:
            raise SystemExit("verification receipt was produced by a different runtime")
    config = profile.selected
    environment = dict(os.environ)
    environment.update(config.runtime_env(profile_path.parent))
    environment.update({
        "EXL3_MOE_CPU_SPLIT": str(config.moe_cpu_split),
        "EXL3_MOE_CPU_THREADS": str(config.moe_cpu_threads),
        "EXL3_CHAMPION_CACHE_TOKENS": str(config.cache_tokens),
        "EXL3_CHAMPION_CACHE_QUANT": str(config.cache_quant),
        "EXL3_CHAMPION_MTP": "1" if config.mtp else "0",
        "EXL3_CHAMPION_NDT": str(config.num_draft_tokens or 0),
        "EXL3_CHAMPION_PROFILE_ID": str(profile.profile_id),
    })
    if args.print_env:
        print(json.dumps({key: value for key, value in environment.items() if key.startswith("EXL3_")}, indent = 2))
        return 0
    command = list(args.backend_command or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("serve requires a backend command after -- (or use --print-env)")
    model_path = str(Path(args.model or profile.model["path"]).expanduser().resolve())
    substitutions = {
        "{model}": model_path,
        "{mcs}": str(config.moe_cpu_split),
        "{mct}": str(config.moe_cpu_threads),
        "{ctx}": str(config.cache_tokens),
        "{cq}": str(config.cache_quant),
        "{ndt}": str(config.num_draft_tokens or 0),
    }
    expanded = []
    for value in command:
        if value == "{champion_args}":
            expanded.extend([
                "-m", model_path,
                "-mcs", str(config.moe_cpu_split),
                "-mct", str(config.moe_cpu_threads),
                "-cs", str(config.cache_tokens),
                "-cq", str(config.cache_quant),
            ])
            if config.mtp:
                expanded.extend(["-mtp", "-ndt", str(config.num_draft_tokens or 4)])
        else:
            expanded.append(substitutions.get(value, value))
    command = expanded
    return subprocess.run(command, env = environment, check = False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog = "0xrc-exllamav3",
        description = "Qualify and launch the 0xrc ExLlamaV3 hot-expert runtime",
    )
    sub = parser.add_subparsers(dest = "command", required = True)

    doctor = sub.add_parser("doctor", help = "inspect hardware, software and model identity")
    doctor.add_argument("--model", help = "optional EXL3 model directory to fingerprint")
    doctor.add_argument("--json-out", help = "also write the report to this path")
    doctor.set_defaults(func = command_doctor)

    check = sub.add_parser("profile-check", help = "validate a sealed 0xrc profile")
    check.add_argument("profile")
    check.set_defaults(func = command_profile_check)

    tune_parser = sub.add_parser("tune", help = "discover, profile and qualify the fastest safe configuration")
    tune_parser.add_argument("--model", required = True)
    tune_parser.add_argument("--output", required = True)
    tune_parser.add_argument("--prompts", help = "representative JSON or JSONL workload")
    tune_parser.add_argument("--context", type = int, default = 8192)
    tune_parser.add_argument("--cache-quant", type = int, default = 3)
    tune_parser.add_argument("--min-vram-headroom", type = float, default = 1.5)
    tune_parser.add_argument("--min-host-available", type = float, default = 8.0)
    tune_parser.add_argument("--min-adaptive-gain", type = float, default = 5.0)
    tune_parser.add_argument("--no-mtp", action = "store_true")
    tune_parser.add_argument("--dry-run", action = "store_true")
    tune_parser.add_argument("--resume", action = "store_true", help = "reuse matching completed trials")
    tune_parser.set_defaults(func = command_tune)

    verify = sub.add_parser("verify", help = "re-run three fresh-process correctness/performance gates")
    verify.add_argument("profile")
    verify.add_argument("--model")
    verify.add_argument("--json-out")
    verify.set_defaults(func = command_verify)

    serve = sub.add_parser("serve", help = "launch a backend with a verified profile")
    serve.add_argument("profile")
    serve.add_argument("--model")
    serve.add_argument("--print-env", action = "store_true")
    serve.add_argument("--allow-hardware-mismatch", action = "store_true")
    serve.add_argument("--verification", help = "verification receipt (defaults beside profile)")
    serve.add_argument("--allow-unverified", action = "store_true")
    serve.add_argument("backend_command", nargs = argparse.REMAINDER)
    serve.set_defaults(func = command_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
