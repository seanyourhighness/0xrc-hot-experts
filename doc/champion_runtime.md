# Champion Runtime release candidate

Champion Runtime turns the experimental EXL3 MoE CPU path into a fail-closed workflow:

```text
doctor -> tune static hardware -> capture expert routing -> qualify static vs adaptive -> verify -> serve
```

The profile is specific to one hardware fingerprint, model fingerprint, runtime identity, and prompt workload. Do not copy a profile from a different GPU or checkpoint. The runtime starts with static placement; profile-seeded adaptive swapping is promoted only after three fresh-process runs per arm, exact token-hash checks, full-length generation checks, memory-headroom gates, and a median throughput gain of at least 5%.

This is a release candidate until the clean RTX 4070 Ti / 48 GB qualification receipt is published.

The public branch topology and upstream-validation order are documented in
[champion_branches.md](champion_branches.md).

## Clean install

Linux or WSL2, a working NVIDIA driver, AVX2, Git, and `uv` are required. The model must already be an EXL3 checkpoint on a fast local filesystem.

```bash
git clone https://github.com/seanyourhighness/exllamav3-champion-runtime.git
cd exllamav3-champion-runtime
git checkout champion-v0.1.0-rc4

uv venv --python 3.11
source .venv/bin/activate
uv pip install torch setuptools wheel ninja --torch-backend=auto
MAX_JOBS=4 uv pip install --no-build-isolation -e .
```

The release-candidate tag is immutable. Use `release/champion-runtime-v0.1` only when intentionally
testing work newer than the published candidate.

## RTX 4070 Ti / 48 GB qualification

Set paths for the local checkpoint and a durable output directory:

```bash
export CHAMPION_MODEL=/absolute/path/to/exl3-model
export CHAMPION_RUN="$PWD/champion-runs/4070ti"
```

Inspect the machine and model before allocating either GPU or system memory:

```bash
exllamav3-champion doctor \
  --model "$CHAMPION_MODEL" \
  --json-out "$CHAMPION_RUN/doctor.json"
```

The first doctor pass streams every safetensors shard to compute a cryptographic checkpoint
fingerprint. Later commands reuse a cache keyed by path, size, inode, and nanosecond change times;
any replaced or modified shard is rehashed.

Preview the candidate matrix:

```bash
exllamav3-champion tune \
  --model "$CHAMPION_MODEL" \
  --output "$CHAMPION_RUN" \
  --context 8192 \
  --cache-quant 3 \
  --no-mtp \
  --dry-run
```

Run qualification. `--resume` reuses only trials whose sealed manifest matches this hardware, model, runtime, workload, and tuning policy, so the same command is safe after an interruption:

```bash
exllamav3-champion tune \
  --model "$CHAMPION_MODEL" \
  --output "$CHAMPION_RUN" \
  --context 8192 \
  --cache-quant 3 \
  --min-vram-headroom 1.5 \
  --min-host-available 8 \
  --min-adaptive-gain 5 \
  --no-mtp \
  --resume
```

The tuner discovers safe-first CPU expert counts, physical-core thread counts, and CPU pinning on/off before expert profiling. It then captures routing over twelve prose/code/reasoning windows and evaluates the learned placement on disjoint held-out windows. The full run starts many fresh model processes and can take a while; logs and JSON evidence are retained under `champion-runs/4070ti/trials/`.

For a production workload, replace the built-in prompts with `--prompts prompts.json`. The file can be a JSON array or JSONL; each row uses this shape:

```json
{"lane":"code","prompt":"Write a bounded queue in Python.","max_tokens":128}
```

Use at least eight representative windows. The tuner snapshots the normalized workload into the run directory, so later verification does not depend on the original file.

## Verify and launch

Validate the profile seal, then run the independent three-process verification gate:

```bash
export CHAMPION_PROFILE="$CHAMPION_RUN/champion-profile.json"

exllamav3-champion profile-check "$CHAMPION_PROFILE"
exllamav3-champion verify "$CHAMPION_PROFILE"
```

`verify` writes `champion-profile.verification.json`. `serve` refuses to launch if that receipt, the hardware fingerprint, the model fingerprint, or the runtime source identity does not match.

To inspect the exact runtime environment without launching a backend:

```bash
exllamav3-champion serve "$CHAMPION_PROFILE" --print-env
```

To launch an ExLlamaV3 command, use `{champion_args}` where the model, CPU split, worker threads, cache length/quantization, and optional qualified MTP settings should be inserted. Quote the placeholder so the shell does not expand it:

```bash
exllamav3-champion serve "$CHAMPION_PROFILE" -- \
  python examples/chat.py '{champion_args}' -mode qwen35
```

Other backends can read the exported `EXL3_*` variables or accept the individual `{model}`, `{mcs}`, `{mct}`, `{ctx}`, `{cq}`, and `{ndt}` placeholders.

## Evidence and safety behavior

The run directory contains:

- `doctor.json`: hardware, CUDA/PyTorch, RAM, GPU, and model identity.
- `plan.json`: generated tuning candidates.
- `workload-prompts.json`: immutable normalized qualification workload.
- `expert-stats.json`: per-layer routing counts used for hot/cold placement.
- `expert-profile-evidence.json`: train/held-out capture quality and warnings.
- `trials/*.json`, `*.log`, and `*.manifest.json`: complete per-process evidence and resumability manifests.
- `champion-profile.json`: sealed selected configuration and all trial summaries.
- `champion-profile.verification.json`: final independent verification receipt.

Adaptive swapping remains off when it fails any correctness gate, does not match static output hashes, has fewer than three passing fresh-process repeats, or improves median throughput by less than the configured threshold. A static profiled placement is the normal safe result, not a failure.

Do not use `--allow-hardware-mismatch` or `--allow-unverified` for a published benchmark or production launch. Those flags exist only for explicit diagnosis.

## Optional MTP qualification

Keep the first 4070 Ti qualification focused with `--no-mtp`. MTP changes live on the separate `feature/mtp-hot-head` branch and should be merged or installed only after the base runtime receipt passes. Then rerun the complete tuning workflow without `--no-mtp`; the tuner compares draft lengths 2, 3, and 4 and keeps MTP only when it is faster and admissible.
