# BeeLLaMA SM120 runtime — Escha E3/W2 + stock GGUF (r9)

This directory is a self-contained release component of this repository. It is
not part of the ExLlamaV3 Hot Experts project and does not modify it.

The installable runtime is attached to the matching GitHub release as
`escha-beellama-v047-sm120-r9.tar.zst`. This directory carries the launcher,
the qualification tooling, the exact source patches, the kernel payload and
the measured results, so the release can be reviewed and rebuilt without
downloading the binary bundle.

## What it is

An RTX 5090 (SM120) BeeLLaMA v0.4.7 runtime that serves three targets through
one launcher: the stock GGUF path and the Escha E3 and W2 targets, with the
qualified SM120 kernel routes selected automatically. DFlash2 depth four
speculative decoding is supported for all three.

## Install and run

```bash
tar --zstd -xf escha-beellama-v047-sm120-r9.tar.zst
cd escha-beellama-v047-sm120-r9
./escha doctor --model e3 --model-path /path/to/escha-e3-with-mtp.gguf --report doctor-e3.json
./escha serve  --model e3 --model-path /path/to/escha-e3-with-mtp.gguf --profile ordinary-8k
./escha serve  --model e3 --model-path /path/to/escha-e3-with-mtp.gguf --profile dflash-8k \
  --draft-path /path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

Changing target means changing `--model` and `--model-path` only. See
`QUICKSTART.md` for profiles, `MODELS.md` for the exact external model files
and hashes, and `PARITY.md` for the measured results and limitations.

## Performance requirement

Candidate throughput divided by a matched native GGUF control on the same card
in the same operating mode must be at least 0.95 for both E3 and W2, for prose
and code separately. Measured through the packaged build:

| Mode | E3 | W2 |
| --- | ---: | ---: |
| Ordinary prefill | 0.984-0.990 | 0.985-1.000 |
| Ordinary decode | 0.963-0.981 | 0.960-0.978 |
| DFlash2 prose | 0.994 | 1.006 |
| DFlash2 code | 1.004 | 1.026 |

Ranges span repeat runs; `PARITY.md` gives the exact per-cell numbers, sample
counts and spread.

## Layout

| Path | Contents |
| --- | --- |
| `escha`, `doctor.py` | launcher and payload/model/GPU verifier |
| `tools/release-correctness.py` | correctness and integration battery |
| `profiles/escha-sm120-bridge.env` | qualified SM120 route defaults |
| `bridge/`, `bridge-e3-down/` | selected CUDA bridge payload and cubins |
| `source/` | exact source patch against the pinned Preview commit, bridge sources |
| `docs/escha-v047-sm120-parity-ledger.md` | full experiment history, including rejections |
| `receipts/` | machine-readable parity and correctness receipts |
| `MANIFEST.json`, `SHA256SUMS` | payload hashes for the release asset |

`MANIFEST.json` describes the release asset, including the ELF payload that is
not stored in git. Verify a downloaded asset against `SHA256SUMS` inside it.

## Limits

Support is claimed only for the tested environment: RTX 5090, Ubuntu 24.04
under WSL, driver 616.56, glibc 2.38 and `GLIBCXX_3.4.32`. The SM89 path is
untouched by this release. See `PARITY.md` for the performance caveats.

