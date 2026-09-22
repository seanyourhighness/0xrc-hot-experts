import json
from types import SimpleNamespace

import pytest

from exllamav3.champion.policy import (
    candidate_cpu_splits,
    candidate_cpu_threads,
    promote_adaptive,
    select_fastest,
)
from exllamav3.champion.schema import ChampionConfig, ChampionProfile, TrialResult
from exllamav3.champion.tuner import (
    _fresh_process_hashes_match,
    _trial_outputs_match_reference,
    run_worker,
)


def make_trial(name, tps, *, correct = True, vram = 10.0, ram = 12.0, adaptive = False):
    return TrialResult(
        trial_id = name,
        config = ChampionConfig(
            moe_cpu_split = 400,
            moe_cpu_threads = 8,
            placement_mode = "seeded-adaptive" if adaptive else "static",
            expert_stats = "experts.json" if adaptive else None,
        ),
        aggregate_decode_tok_per_s = tps,
        output_hashes = ["a", "b"] if correct else [],
        output_lengths = [128, 128] if correct else [],
        no_short_stops = correct,
        all_measured_lanes_full_length = correct,
        same_process_hash_match = correct,
        vram_peak_gb = vram,
        host_available_gb_min = ram,
        load_seconds = 1.0,
    )


def test_schema_round_trip_and_tamper_detection(tmp_path):
    trial = make_trial("static-1", 20.0)
    profile = ChampionProfile(
        hardware = {"fingerprint": "hardware"},
        model = {"fingerprint": "model"},
        runtime = {"commit": "abc"},
        selected = trial.config,
        trials = [trial],
        verification = {"selected_trial_id": trial.trial_id},
    )
    path = tmp_path / "profile.json"
    profile.write(path)
    loaded = ChampionProfile.read(path)
    assert loaded.profile_id == profile.profile_id
    assert loaded.selected == profile.selected

    value = json.loads(path.read_text())
    value["selected"]["moe_cpu_threads"] = 7
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match = "checksum"):
        ChampionProfile.read(path)


def test_runtime_env_is_fail_closed_and_resolves_stats(tmp_path):
    static = ChampionConfig(moe_cpu_split = 400, moe_cpu_threads = 8)
    assert static.runtime_env()["EXL3_MOE_CPU_SWAP"] == "0"
    adaptive = ChampionConfig(
        moe_cpu_split = 400,
        moe_cpu_threads = 8,
        placement_mode = "seeded-adaptive",
        expert_stats = "experts.json",
    )
    env = adaptive.runtime_env(tmp_path)
    assert env["EXL3_MOE_CPU_SWAP"] == "1"
    assert env["EXL3_MOE_CPU_SWAP_SEED"] == "1"
    assert env["EXL3_MOE_CPU_SWAP_INTERVAL"] == "32"
    assert env["EXL3_MOE_CPU_SWAP_MAX"] == "64"
    assert env["EXL3_MOE_CPU_SWAP_HYST"] == "2.0"
    assert env["EXL3_MOE_CPU_SWAP_FLOOR"] == "8.0"
    assert env["EXL3_MOE_CPU_SPLIT_STATS"] == str((tmp_path / "experts.json").resolve())


def test_candidate_generation_is_safe_first_for_4070_ti():
    splits = candidate_cpu_splits(512, 12.0)
    assert splits == sorted(splits, reverse = True)
    assert all(0 < value < 512 for value in splits)
    assert splits[0] >= 416
    assert candidate_cpu_threads(16) == [4, 8, 12, 16]


def test_selection_rejects_correctness_and_headroom_failures():
    good = make_trial("good", 20.0)
    faster_bad = make_trial("bad", 30.0, correct = False)
    too_tight = make_trial("tight", 40.0, vram = 11.5)
    assert select_fastest([good, faster_bad, too_tight], total_vram_gb = 12.0) is good


def test_adaptive_requires_three_repeats_and_minimum_gain():
    static = [make_trial(f"s{i}", value) for i, value in enumerate((20.0, 20.2, 19.8))]
    adaptive = [make_trial(f"a{i}", value, adaptive = True)
                for i, value in enumerate((21.2, 21.4, 21.0))]
    promoted, evidence = promote_adaptive(static, adaptive, minimum_gain_percent = 5.0)
    assert promoted
    assert evidence["gain_percent"] >= 5.0
    assert promote_adaptive(static[:2], adaptive)[0] is False


def test_fresh_process_gate_compares_hashes_and_lengths():
    repeats = [make_trial(f"s{i}", 20.0 + i / 10) for i in range(3)]
    assert _fresh_process_hashes_match(repeats)
    repeats[2].output_hashes[1] = "changed"
    assert not _fresh_process_hashes_match(repeats)


def test_runtime_env_allows_tuned_cpu_pinning_override():
    config = ChampionConfig(
        moe_cpu_split = 400,
        moe_cpu_threads = 8,
        env = {"EXL3_MOE_CPU_PIN": "0"},
    )
    assert config.runtime_env()["EXL3_MOE_CPU_PIN"] == "0"


def test_adaptive_outputs_must_match_static_baseline():
    static = make_trial("static", 20.0)
    adaptive = [make_trial(f"a{i}", 21.0, adaptive = True) for i in range(3)]
    assert _trial_outputs_match_reference(static, adaptive)
    adaptive[1].output_hashes[0] = "different"
    assert not _trial_outputs_match_reference(static, adaptive)


def test_worker_resume_reuses_only_matching_manifest(tmp_path, monkeypatch):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps([
        {"lane": "test", "prompt": "hello", "max_tokens": 8},
    ]))
    payload = {
        "aggregate_decode_tok_per_s": 20.0,
        "output_hashes": ["abc"],
        "output_lengths": [8],
        "correctness": {
            "no_short_stops": True,
            "all_measured_lanes_full_length": True,
            "same_process_hash_match": True,
        },
        "resources": {"vram_peak_gb": 8.0, "host_available_gb_min": 16.0},
        "load_seconds": 1.0,
        "results": [],
    }

    def fake_run(command, **_kwargs):
        result_path = command[command.index("--json-out") + 1]
        with open(result_path, "w", encoding = "utf-8") as handle:
            json.dump(payload, handle)
        return SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr("exllamav3.champion.tuner.subprocess.run", fake_run)
    config = ChampionConfig(moe_cpu_split = 400, moe_cpu_threads = 8)
    first, _ = run_worker(
        model = "/model",
        config = config,
        trial_id = "resume",
        output_dir = tmp_path,
        prompt_file = str(prompt_path),
        resume_token = "sealed-inputs",
    )
    assert first.correctness_passed

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("matching trial should have been resumed")

    monkeypatch.setattr("exllamav3.champion.tuner.subprocess.run", must_not_run)
    resumed, _ = run_worker(
        model = "/model",
        config = config,
        trial_id = "resume",
        output_dir = tmp_path,
        prompt_file = str(prompt_path),
        resume_token = "sealed-inputs",
        reuse_existing = True,
    )
    assert resumed.correctness_passed
    assert (tmp_path / "resume.log").read_text().startswith("RESUMED")
