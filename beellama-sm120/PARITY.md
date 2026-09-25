# Matched native-GGUF parity results

All numbers below were measured on the packaged r9 build, launched through the
shipped `./escha` launcher against a matched native GGUF control on the same
RTX 5090 in the same session. The gate is candidate divided by matched native
in the same operating mode, and the requirement is at least 0.95 for both E3
and W2. Prose and code are separate cells.

Fixed inputs for every arm: KVarN3/2 target cache, one slot, FlashAttention on,
batch 2048, 8 threads, full GPU offload. DFlash2 arms use the identical
Qwen3.8-27B-DFlash2-Q4_K_M drafter at depth four with Q4 draft KV, and every
request was forced to exactly 500 output tokens so no arm could win by
stopping early. Five measured requests per arm after one warmup.

## Ordinary decode and prefill (no speculation)

| Cell | Native | E3 | E3 ratio | W2 | W2 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill p2048 UB2048 | 3207.35 | 3176.22 | **0.990** | 3207.50 | **1.000** |
| Decode p0 n256 UB512 | 91.25 | 87.91 | **0.963** | 87.62 | **0.960** |

## DFlash2 depth four (identical drafter and configuration)

Four interleaved fresh-start rounds of five measured requests each, pooled:

| Cell | Native | E3 | E3 ratio | W2 | W2 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 125.38 | 124.68 | **0.994** | 126.07 | **1.006** |
| Code | 172.64 | 173.29 | **1.004** | 177.15 | **1.026** |

## Spread and how to read this

The Escha measurements are the stable side. The native control drifts more
between fresh server starts on this host, so single-round ratios are noisy and
must not be quoted alone: for DFlash2 prose the per-round ratios were 0.912,
1.002, 1.054 and 1.017 for E3 and 0.953, 1.024, 1.018 and 1.031 for W2, and
for DFlash2 code 0.998, 1.018, 1.015 and 0.984 for E3. An earlier
three-round bracket of the ordinary-decode cell measured 0.9485 to 0.9694 per
round around a pooled 0.959. Treat the pooled interleaved means as the result
and the per-round range as the uncertainty.

## Known limitations

- Scope is the tested RTX 5090 on Ubuntu 24.04 under WSL with driver 616.56.
  Other hardware and hosts are untested.
- The stock GGUF control runs with the Escha K=1 GDN chunk bridge disabled:
  that bridge is an Escha-specific route and the stock IQ3 model fails its
  numerical guard. Its other SM120 routes are enabled, and one CUDA library
  serves every arm.
- DFlash2 prose on the stock GGUF target and on W2 remains the noisiest cell;
  it passes on pooled interleaved means, not on every single round.
- The ordinary-decode margin is real but narrow. The remaining penalty is
  attributed to the per-projection split-K finalization launches that the
  native fused matvec does not need; see the parity ledger.
- Qualification covers one-slot 8K and 32K profiles. Longer contexts, multiple
  slots and MTP are not part of this release claim.

