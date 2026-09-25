# Model assets (not included in this archive)

Download or build these separately, then pass them to `./escha` by path.
`doctor` verifies the SHA-256 of the target model you pass to it.

| Role | File | SHA-256 |
| --- | --- | --- |
| Escha E3 target | escha-e3-with-mtp.gguf | 746bd40841fb18b9c1918923e89c007df70b5e4f3de64290d1399593e70c96b0 |
| Escha W2 target | escha-w2-with-mtp.gguf | 3f93cbe77a20f1fa7272741757596cac66a66457d7ecaed1e5a6e4baa409535e |
| Matched native control | IQ3 GGUF, Qwen3.8-27B family | ad85e40a28aa (see MANIFEST.json for the full value) |
| DFlash2 drafter | Qwen3.8-27B-DFlash2-Q4_K_M.gguf | 1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd |

The E3 and W2 targets are the merged Escha GGUFs used for every measurement in
`PARITY.md`. They are intentionally a different quantization from the native
control; that difference is inherent to the comparison and is not corrected
for. No model weights are modified by this runtime, and no weight
requantization happens at load time.

The DFlash2 drafter is a stock upstream artifact and must be the exact file
above for the speculative results in `PARITY.md` to apply.

