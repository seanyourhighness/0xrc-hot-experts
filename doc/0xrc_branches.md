# 0xrc branch topology

The immutable release snapshot and the upstream-oriented feature branches serve different
purposes. Do not rewrite the archive refs to make the history look cleaner.

```text
30846b8  pre-0xrc base
  |
  +-- feature/runtime-correctness       1aafb3b
        |
        +-- feature/profile-placement   3f1e7fc
              |
              +-- feature/adaptive-hot-experts  e8e8d28

30846b8
  |
  +-- feature/mtp-hot-head              48309ac  (independent)
```

The stacked order is intentional: profile placement depends on the deterministic runtime,
and adaptive placement depends on a validated static profile. MTP remains independent so it can
be benchmarked and reviewed without changing target-model expert placement.

## Branch responsibilities

### `feature/runtime-correctness`

- Deterministic streamed CPU-MoE reduction support.
- CPU worker/K3 runtime fixes and bounded diagnostics used to qualify correctness and throughput.
- No profile-seeded adaptive placement changes.

Primary gate:

```bash
python -m pytest -q tests/test_moe_cpu_tiers_.py
```

### `feature/profile-placement`

- Builds on `feature/runtime-correctness`.
- Makes static placement the safe default.
- Validates each complete per-layer count vector before applying a stable hot-to-cold permutation.
- Uses configured inference parameters—not a process-wide split environment variable—to disable
  deferred router loading only for an actual profiled split.
- Fails closed on missing, non-finite, negative, or incorrectly sized profile data.

Primary gate:

```bash
python -m pytest -q tests/test_moe_profile_placement_.py tests/test_moe_cpu_tiers_.py
```

### `feature/adaptive-hot-experts`

- Builds on `feature/profile-placement`.
- Requires explicit `EXL3_MOE_CPU_SWAP=1` and `EXL3_MOE_CPU_SWAP_SEED=1` opt-in.
- Refuses seeded mode without a complete static profile.
- Translates permuted router IDs back to original checkpoint expert IDs for promotion and demotion.
- Leaves static profile placement as the default and control arm.

Primary gate:

```bash
python -m pytest -q \
  tests/test_moe_profile_placement_.py \
  tests/test_moe_seed_swap_.py \
  tests/test_moe_cpu_tiers_.py
```

### `feature/mtp-hot-head`

- Starts independently from `30846b8`.
- Contains only the three Qwen4-Exp MTP hot-head files.
- Must be tested after the non-MTP runtime qualifies on the target machine.

## Immutable and release refs

- `archive/champion-runtime-2026-09-22` / `champion-runtime-baseline-2026-09-22` preserve the exact
  validated four-file 0xrc runtime at `983c4b0`.
- `archive/champion-full-2026-09-22` preserves the runtime plus the original MTP snapshot.
- `release/champion-runtime-v0.1` contains the installable qualification workflow and is the branch
  users should run.

The focused branches are intentionally not force-pushed over the immutable baseline. They are
clean review surfaces for independent tests and eventual upstream pull requests.
