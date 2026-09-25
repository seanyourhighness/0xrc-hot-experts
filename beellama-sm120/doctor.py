#!/usr/bin/env python3
"""Verify a release payload and optionally a model; emit an opt-in local report."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# WSL exposes the driver utilities outside the default PATH.
NVIDIA_SMI = next(
    (candidate for candidate in
     (shutil.which("nvidia-smi"), "/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi")
     if candidate and Path(candidate).exists()),
    "nvidia-smi")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_memory_mib() -> int | None:
    probe = subprocess.run(
        [NVIDIA_SMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False)
    try:
        return int(probe.stdout.splitlines()[0].strip()) if probe.returncode == 0 else None
    except (IndexError, ValueError):
        return None


def request_json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def smoke(model_kind: str, model_path: Path) -> dict:
    """Exercise the installed launcher; retain only bounded, content-free metrics."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    baseline = gpu_memory_mib()
    result = {"status": "fail", "profile": "sm120-kvarn3-2-mtp-n2-32k",
              "health": False, "control_pass": False, "gpu_baseline_mib": baseline,
              "gpu_peak_mib": baseline, "host_peak_rss_kib": None}
    stop_sample = threading.Event()
    process = None
    with tempfile.TemporaryFile(mode="w+t") as log:
        try:
            process = subprocess.Popen(
                [str(ROOT / "escha"), "serve", "--model", model_kind,
                 "--model-path", str(model_path), "--port", str(port)],
                stdout=log, stderr=subprocess.STDOUT)

            def sample() -> None:
                while not stop_sample.wait(0.5):
                    used = gpu_memory_mib()
                    if used is not None:
                        result["gpu_peak_mib"] = max(result["gpu_peak_mib"] or 0, used)
                    try:
                        lines = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                        rss = next(int(line.split()[1]) for line in lines
                                   if line.startswith("VmRSS:"))
                        result["host_peak_rss_kib"] = max(result["host_peak_rss_kib"] or 0, rss)
                    except (OSError, StopIteration, ValueError):
                        pass

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            base = f"http://127.0.0.1:{port}"
            for _ in range(150):
                if process.poll() is not None:
                    result["error_category"] = "server-exited-before-health"
                    return result
                try:
                    if request_json(base + "/health").get("status") == "ok":
                        result["health"] = True
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                result["error_category"] = "health-timeout"
                return result
            started = time.perf_counter()
            response = request_json(base + "/v1/chat/completions", {
                "model": model_path.name,
                "messages": [{"role": "user", "content":
                              "Reply with exactly PARITY_OK and nothing else."}],
                "temperature": 0, "max_tokens": 32,
            })
            result["request_latency_s"] = round(time.perf_counter() - started, 3)
            result["prompt_tokens"] = response.get("usage", {}).get("prompt_tokens")
            result["completion_tokens"] = response.get("usage", {}).get("completion_tokens")
            result["control_pass"] = (
                response["choices"][0]["message"]["content"].strip() == "PARITY_OK")
            result["status"] = "pass" if result["control_pass"] else "fail"
            if not result["control_pass"]:
                result["error_category"] = "control-output-mismatch"
        except (OSError, urllib.error.URLError, TimeoutError, KeyError,
                IndexError, ValueError) as error:
            result["error_category"] = type(error).__name__
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            stop_sample.set()
            if process is not None:
                sampler.join(timeout=2)
                result["server_exit_code"] = process.returncode
            result["gpu_after_mib"] = gpu_memory_mib()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("native", "e3", "w2"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if bool(args.model) != bool(args.model_path):
        parser.error("--model and --model-path must be supplied together")
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    failures = []
    payload = {}
    for name, expected in manifest["files"].items():
        path = ROOT / name
        actual = sha256(path) if path.is_file() else None
        good = actual == expected
        payload[name] = {"ok": good, "sha256": actual}
        if not good:
            failures.append(f"payload hash mismatch: {name}")
    gpu = None
    if Path(NVIDIA_SMI).exists():
        result = subprocess.run(
            [NVIDIA_SMI, "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, check=False)
        gpu = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    if not gpu or "12.0" not in gpu:
        failures.append("no accessible SM120 GPU reported by nvidia-smi")
    loader = subprocess.run(["ldd", str(ROOT / "bin/llama-server")],
                            capture_output=True, text=True, check=False)
    loader_lines = (loader.stdout + loader.stderr).splitlines()
    loader_ok = loader.returncode == 0 and not any(
        "not found" in line or ("version" in line and
                                ("GLIBC_" in line or "GLIBCXX_" in line))
        for line in loader_lines)
    if not loader_ok:
        abi = manifest["host_abi"]
        failures.append("host runtime ABI incompatible (requires glibc " +
                        abi["minimum_glibc"] + " and GLIBCXX_" + abi["minimum_glibcxx"] + ")")
    model = None
    if args.model_path:
        actual = sha256(args.model_path) if args.model_path.is_file() else None
        expected = manifest["models"][args.model]["sha256"]
        model = {"kind": args.model, "filename": args.model_path.name, "sha256": actual,
                 "expected_sha256": expected, "ok": actual == expected}
        if not model["ok"]:
            failures.append("model missing or SHA-256 differs from qualified model")
    result = {"release": manifest["release"], "source_commit": manifest["beellama_preview_commit"],
              "architecture": manifest["architecture"], "cuda": manifest["build"]["cuda"],
              "host_abi_required": manifest["host_abi"],
              "profile": "sm120-kvarn3-2-ordinary-32k",
              "runtime_sha256": payload.get("bin/libllama.so.0.4.7", {}).get("sha256"),
              "bridge_sha256": payload.get(
                  "bridge/libescha_official_bridge_cuda_sm120.so", {}).get("sha256"),
              "status": "pass" if not failures else "fail",
              "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                       "gpu": gpu, "glibc": platform.libc_ver()[1],
                       "loader_ok": loader_ok},
              "payload": payload, "model": model, "failures": failures}
    if args.report and model and not failures:
        result["smoke"] = smoke(args.model, args.model_path)
        if result["smoke"]["status"] != "pass":
            failures.append("local launcher smoke failed")
            result["status"] = "fail"
    elif args.report:
        result["smoke"] = {"status": "skipped", "reason": "no verified model"}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['status'].upper()}: {manifest['release']} source={manifest['beellama_preview_commit'][:12]} "
          f"CUDA={manifest['build']['cuda']} payload={len(payload)} files; GPU: {gpu or 'unavailable'}")
    if model:
        print(f"Model {model['kind']}: {'hash matched' if model['ok'] else 'hash mismatch'}")
    if "smoke" in result:
        print(f"Launcher smoke: {result['smoke']['status']}")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
