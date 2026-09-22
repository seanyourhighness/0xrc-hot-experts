from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_NAME = "exllamav3-champion-profile"
SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys = True, separators = (",", ":"), ensure_ascii = True)


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen = True)
class ChampionConfig:
    moe_cpu_split: int
    moe_cpu_threads: int
    cache_tokens: int = 8192
    cache_quant: int = 3
    mtp: bool = False
    num_draft_tokens: int | None = None
    mtp_cpu_split: int = 0
    placement_mode: str = "static"
    expert_stats: str | None = None
    env: dict[str, str] = field(default_factory = dict)

    def __post_init__(self):
        if self.moe_cpu_split < 0:
            raise ValueError("moe_cpu_split must be nonnegative")
        if self.moe_cpu_threads < 1:
            raise ValueError("moe_cpu_threads must be positive")
        if self.cache_tokens < 256:
            raise ValueError("cache_tokens must be at least 256")
        if self.cache_quant not in (2, 3, 4, 6, 8, 16):
            raise ValueError("cache_quant must be one of 2, 3, 4, 6, 8 or 16")
        if self.placement_mode not in ("static", "seeded-adaptive"):
            raise ValueError("placement_mode must be static or seeded-adaptive")
        if self.placement_mode == "seeded-adaptive" and not self.expert_stats:
            raise ValueError("seeded-adaptive placement requires expert_stats")
        if self.num_draft_tokens is not None and self.num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be positive")

    def runtime_env(self, profile_dir: Path | None = None) -> dict[str, str]:
        env = {
            "EXL3_MOE_CPU_SWAP": "1" if self.placement_mode == "seeded-adaptive" else "0",
            "EXL3_MOE_CPU_SWAP_SEED": "1" if self.placement_mode == "seeded-adaptive" else "0",
            "EXL3_MOE_CPU_PIN": "1",
            "EXL3_MOE_CPU_SWIZZLE": "1",
            "EXL3_MOE_FUSED_DET": "1",
            "EXL3_MOE_STREAM_DET": "1",
            "EXL3_MOE_K3_PAIR": "1",
            "EXL3_MOE_SPLIT_FUSED": "1",
            "EXL3_MOE_BATCH_RECON": "1",
            "EXL3_MOE_MTILE": "1",
            "EXL3_NGRAM_PREFETCH": "1",
            "EXL3_NGRAM_STREAM": "1",
            "EXL3_MOE_PINNED_ARENA": "0",
        }
        if self.placement_mode == "seeded-adaptive":
            env.update({
                "EXL3_MOE_CPU_SWAP_INTERVAL": "32",
                "EXL3_MOE_CPU_SWAP_MAX": "64",
                "EXL3_MOE_CPU_SWAP_HYST": "2.0",
                "EXL3_MOE_CPU_SWAP_FLOOR": "8.0",
            })
        if self.expert_stats:
            stats = Path(self.expert_stats)
            if not stats.is_absolute() and profile_dir is not None:
                stats = profile_dir / stats
            env["EXL3_MOE_CPU_SPLIT_STATS"] = str(stats.resolve())
        env.update({str(k): str(v) for k, v in self.env.items()})
        return env


@dataclass(frozen = True)
class TrialResult:
    trial_id: str
    config: ChampionConfig
    aggregate_decode_tok_per_s: float
    output_hashes: list[str]
    output_lengths: list[int]
    no_short_stops: bool
    all_measured_lanes_full_length: bool
    same_process_hash_match: bool | None
    vram_peak_gb: float
    host_available_gb_min: float
    load_seconds: float
    lane_results: list[dict[str, Any]] = field(default_factory = list)
    error: str | None = None

    @property
    def correctness_passed(self) -> bool:
        return (
            self.error is None
            and self.no_short_stops
            and self.all_measured_lanes_full_length
            and self.same_process_hash_match is not False
            and len(self.output_hashes) > 0
            and len(self.output_hashes) == len(self.output_lengths)
        )


@dataclass
class ChampionProfile:
    hardware: dict[str, Any]
    model: dict[str, Any]
    runtime: dict[str, Any]
    selected: ChampionConfig
    trials: list[TrialResult]
    verification: dict[str, Any] = field(default_factory = dict)
    workload: dict[str, Any] = field(default_factory = dict)
    schema: str = SCHEMA_NAME
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory = lambda: datetime.now(timezone.utc).isoformat())
    profile_id: str | None = None

    def payload(self, include_profile_id: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if not include_profile_id:
            value.pop("profile_id", None)
        return value

    def seal(self) -> str:
        self.profile_id = stable_sha256(self.payload(include_profile_id = False))
        return self.profile_id

    def validate(self):
        if self.schema != SCHEMA_NAME or self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported profile schema {self.schema!r} version {self.schema_version!r}"
            )
        expected = stable_sha256(self.payload(include_profile_id = False))
        if not self.profile_id:
            raise ValueError("profile is not sealed")
        if self.profile_id != expected:
            raise ValueError("profile checksum does not match its contents")
        if not self.hardware.get("fingerprint"):
            raise ValueError("profile has no hardware fingerprint")
        if not self.model.get("fingerprint"):
            raise ValueError("profile has no model fingerprint")
        if not any(t.trial_id == self.verification.get("selected_trial_id") for t in self.trials):
            raise ValueError("verification does not identify one of the recorded trials")

    def write(self, path: str | Path):
        self.seal()
        target = Path(path)
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_text(json.dumps(self.payload(), indent = 2) + "\n", encoding = "utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "ChampionProfile":
        value = json.loads(Path(path).read_text(encoding = "utf-8"))
        value["selected"] = ChampionConfig(**value["selected"])
        value["trials"] = [
            TrialResult(config = ChampionConfig(**row.pop("config")), **row)
            for row in value["trials"]
        ]
        profile = cls(**value)
        profile.validate()
        return profile
