from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import time
from typing import Any

from .schema import stable_sha256


def _read_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding = "utf-8", errors = "replace")
    except (OSError, PermissionError):
        return ""


def _meminfo() -> dict[str, int]:
    values = {}
    for line in _read_text("/proc/meminfo").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0]) * 1024
    return values


def _cpu_info() -> dict[str, Any]:
    blocks = _read_text("/proc/cpuinfo").split("\n\n")
    rows = []
    flags = set()
    model = platform.processor() or "unknown"
    for block in blocks:
        data = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
        if not data:
            continue
        rows.append(data)
        model = data.get("model name", model)
        flags.update(data.get("flags", "").split())
    physical = set()
    for row in rows:
        physical.add((row.get("physical id", "0"), row.get("core id", row.get("processor", "0"))))
    return {
        "model": model,
        "logical_cores": os.cpu_count() or len(rows) or 1,
        "physical_cores": len(physical) or os.cpu_count() or 1,
        "isa": [name for name in ("avx2", "avx512f", "avx512bw", "avx512vnni", "avx512vbmi")
                if name in flags],
    }


def _numa_nodes() -> int:
    root = Path("/sys/devices/system/node")
    try:
        return len([p for p in root.glob("node[0-9]*") if p.is_dir()]) or 1
    except OSError:
        return 1


def _nvidia_smi() -> list[dict[str, Any]]:
    command = shutil.which("nvidia-smi")
    if not command:
        return []
    query = "index,name,uuid,memory.total,driver_version,pci.bus_id"
    try:
        result = subprocess.run(
            [command, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output = True,
            text = True,
            timeout = 10,
            check = True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            continue
        rows.append({
            "index": int(fields[0]),
            "name": fields[1],
            "uuid": fields[2],
            "vram_mib": int(fields[3]),
            "driver": fields[4],
            "pci_bus_id": fields[5],
        })
    return rows


def _torch_info() -> dict[str, Any]:
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        devices = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory": int(props.total_memory),
                })
        return {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_available,
            "devices": devices,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "cuda_available": False, "devices": []}


def _model_hash_cache_path() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return base / "exllamav3" / "champion-model-hashes-v1.json"


def _read_model_hash_cache() -> dict[str, str]:
    try:
        value = json.loads(_model_hash_cache_path().read_text(encoding = "utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _write_model_hash_cache(value: dict[str, str]):
    try:
        target = _model_hash_cache_path()
        target.parent.mkdir(parents = True, exist_ok = True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(dict(list(value.items())[-1024:])), encoding = "utf-8")
        temporary.replace(target)
    except OSError:
        pass


def model_identity(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    tensors = []
    hash_cache = _read_model_hash_cache()
    cache_changed = False
    for path in sorted(root.glob("*.safetensors")):
        stat = path.stat()
        cache_key = stable_sha256({
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        })
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"invalid safetensors header: {path}")
            header_length = struct.unpack("<Q", length_bytes)[0]
            if header_length > 64 * 1024 * 1024:
                raise ValueError(f"unreasonably large safetensors header: {path}")
            header = handle.read(header_length)
            if len(header) != header_length:
                raise ValueError(f"truncated safetensors header: {path}")
            # Some filesystems expose timestamp granularity coarse enough for two same-size
            # writes to share all stat fields. Never trust a cache entry while a shard is still
            # inside that ambiguity window; stable model files regain the fast path after 2 s.
            metadata_settled = max(stat.st_mtime_ns, stat.st_ctime_ns) < time.time_ns() - 2_000_000_000
            content_sha256 = hash_cache.get(cache_key) if metadata_settled else None
            if content_sha256 is None:
                content_hash = hashlib.sha256()
                content_hash.update(length_bytes)
                content_hash.update(header)
                while chunk := handle.read(8 * 1024 * 1024):
                    content_hash.update(chunk)
                content_sha256 = content_hash.hexdigest()
                hash_cache[cache_key] = content_sha256
                cache_changed = True
        tensors.append({
            "name": path.name,
            "size": stat.st_size,
            "header_sha256": hashlib.sha256(header).hexdigest(),
            "sha256": content_sha256,
        })
    if cache_changed:
        _write_model_hash_cache(hash_cache)
    if not tensors:
        raise FileNotFoundError(f"no safetensors files found in {root}")
    text_config = config.get("text_config") or config
    identity = {
        "path": str(root),
        "architecture": (config.get("architectures") or [None])[0],
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_experts": text_config.get("num_experts") or text_config.get("n_routed_experts"),
        "num_experts_per_tok": text_config.get("num_experts_per_tok") or text_config.get("num_experts_per_token"),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "safetensors": tensors,
    }
    identity["fingerprint"] = stable_sha256({k: v for k, v in identity.items() if k != "path"})
    return identity


def collect_hardware(model_dir: str | Path | None = None) -> dict[str, Any]:
    mem = _meminfo()
    release = platform.release()
    hardware = {
        "platform": platform.system(),
        "release": release,
        "machine": platform.machine(),
        "wsl": "microsoft" in release.lower() or "microsoft" in _read_text("/proc/version").lower(),
        "cpu": _cpu_info(),
        "numa_nodes": _numa_nodes(),
        "memory": {
            "total_bytes": mem.get("MemTotal", 0),
            "available_bytes": mem.get("MemAvailable", 0),
        },
        "nvidia": _nvidia_smi(),
        "torch": _torch_info(),
    }
    if model_dir is not None:
        try:
            usage = shutil.disk_usage(Path(model_dir).expanduser().resolve())
            hardware["model_filesystem"] = {"free_bytes": usage.free, "total_bytes": usage.total}
        except OSError:
            # doctor_report will turn the missing/unreadable model into a structured fatal
            # result. Hardware discovery must still complete so the report remains useful.
            hardware["model_filesystem"] = None
    stable = {
        "platform": hardware["platform"],
        "machine": hardware["machine"],
        "wsl": hardware["wsl"],
        "cpu": hardware["cpu"],
        "numa_nodes": hardware["numa_nodes"],
        "memory_total_bytes": hardware["memory"]["total_bytes"],
        "nvidia": hardware["nvidia"],
        "torch_version": hardware["torch"].get("version"),
        "torch_cuda_build": hardware["torch"].get("cuda_build"),
    }
    hardware["fingerprint"] = stable_sha256(stable)
    return hardware


def doctor_report(model_dir: str | Path | None = None) -> dict[str, Any]:
    hardware = collect_hardware(model_dir)
    fatal = []
    warnings = []
    if hardware["platform"] != "Linux":
        warnings.append("Champion qualification is currently validated on Linux and WSL2")
    if not hardware["nvidia"]:
        fatal.append("nvidia-smi did not report an NVIDIA GPU")
    if not hardware["torch"].get("cuda_available"):
        fatal.append("the installed PyTorch build cannot access CUDA")
    if "avx2" not in hardware["cpu"]["isa"]:
        fatal.append("the CPU does not advertise AVX2")
    if hardware["memory"]["total_bytes"] < 32 * 1024 ** 3:
        warnings.append("less than 32 GiB of system RAM is available")
    model = None
    if model_dir is not None:
        try:
            model = model_identity(model_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            fatal.append(str(exc))
    return {
        "ok": not fatal,
        "fatal": fatal,
        "warnings": warnings,
        "hardware": hardware,
        "model": model,
    }
