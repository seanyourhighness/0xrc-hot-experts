# Escha BeeLLaMA 0.4.7 SM120 release (r9)

One installable RTX 5090 (SM120) runtime for BeeLLaMA v0.4.7 that serves the
stock GGUF path and the Escha E3/W2 targets through the same launcher, with
the qualified SM120 routes selected automatically. Validated on Ubuntu 24.04
under WSL on an RTX 5090 with driver 616.56.

## Install

```bash
tar --zstd -xf escha-beellama-v047-sm120-r9.tar.zst
cd escha-beellama-v047-sm120-r9
./escha doctor --model e3 --model-path /absolute/path/to/escha-e3-with-mtp.gguf --report doctor-e3.json
```

`doctor` verifies every payload file against `MANIFEST.json`, checks that an
SM120 GPU is reachable, checks the model hash when one is supplied, and with
`--report` runs a bounded local launcher, health and control-output smoke. It
sends no data.

Models are not in this archive. See `MODELS.md` for the exact files and their
SHA-256 values.

## Run

```bash
# Escha E3, ordinary decode, 32K context
./escha serve --model e3 --model-path /path/to/escha-e3-with-mtp.gguf

# Escha W2, ordinary decode, 8K context
./escha serve --model w2 --model-path /path/to/escha-w2-with-mtp.gguf --profile ordinary-8k

# Stock GGUF target on the same runtime
./escha serve --model native --model-path /path/to/stock-model.gguf --profile ordinary-8k

# DFlash2 depth four (E3, W2 or the stock GGUF target)
./escha serve --model e3 --model-path /path/to/escha-e3-with-mtp.gguf --profile dflash-8k \
  --draft-path /path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

Changing model means changing `--model` and `--model-path`. Everything else is
automatic: the launcher selects the E3 multi-row verification head, the W2
original-INT8 head for ordinary decode, the multi-row head for five-row DFlash
verification, and the SM120 coded-projection, attention and GDN routes. Do not
set kernel-route environment variables by hand.

Profiles:

| Profile | Context | Microbatch | Speculation |
| --- | ---: | ---: | --- |
| `ordinary-32k` (default) | 32768 | 1024 | none |
| `ordinary-8k` | 8192 | 2048 | none |
| `dflash-8k` | 8192 | 512 | DFlash2 depth four, Q4 draft KV |

Extra `llama-server` flags may be appended; they can move the server outside
the qualified profile. `./escha env --model e3 --profile ordinary-8k` prints
the exact route environment the launcher resolves. Stop the server with Ctrl-C.

## Qualification

Every performance claim in `PARITY.md` was measured on the packaged build
through this launcher against a matched native GGUF control on the same card.
The gate is candidate throughput divided by matched native throughput in the
same mode, and the requirement is at least 0.95 for both E3 and W2, for prose
and code separately. docs/escha-v047-sm120-parity-ledger.md carries the full
experiment history, including rejected work.

## Host requirements

The ELF payload needs glibc 2.38 and a libstdc++ exporting GLIBCXX_3.4.32.
The archive bundles CUDA 13.0 libcudart, libcublas and libcublasLt;
libcuda.so.1 comes from the host driver. Support is claimed only for the
tested RTX 5090 Linux/WSL environment. The SM89 path is unchanged by this
release.

