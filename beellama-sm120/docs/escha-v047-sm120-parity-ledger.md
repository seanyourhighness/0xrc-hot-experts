# Escha BeeLLaMA v0.4.7 SM120 parity and release ledger

Updated: 2026-09-24 PDT. **Current artifact: r8, bounded experimental MTP
stage; no release gate below has passed.** r4 remains withdrawn for non-finite
logits from its old mismatched GDN pipeline. Source-matched GDN-12 through
GDN-17 are separate, later evidence; do not invalidate or qualify them by
association with that retired pipeline. r7/r8 archives remain frozen.

Current authoritative gates, superseding historical objectives below:

1. **Gate A:** E3 and W2 each reach >=95% of matched native GGUF BeeLLaMA
   prefill AND decode, with all speculative decoding disabled.
2. **Gate B:** DFlash2 depth four reaches >=2.0x matched **served**,
   non-speculative native GGUF decode on each declared release workload.
   The next-sprint proposal makes E3/W2 x narrative/code explicit. Historical
   150 prose / 200 code targets are directional, not the authoritative gate.
3. Qualify the exact winning artifact for its declared release scope after
   speed passes; keep cheap benchmark-validity checks throughout development.

Historical corrected raw KVarN3/2 medians (prefill/decode tok/s): native IQ3
3104.70/90.50; E3 2869.16/79.67; W2 2790.75/75.62. All four Escha cells
were below Gate A then; GDN-20/RES-01 supersede this discovery screen.
GDN-17's E3 3148.47/83.76 and W2 2983.57/86.47 are MTP served results
and cannot satisfy Gate A. GDN-18 failed its initial numerical screen
(GDN-19). Its K=1 refers
to recurrent state slots during 2048-token prefill, not single-token decode.
GDN-20's corrected full-batch K=1 route is linked into a clean SM120 CUDA
library and passes the p2048 raw prefill screen for E3 and W2. The remaining
raw Gate A deficit is decode; the served 8K release workload remains unqualified.

Next-sprint review and ranked plan: [roadmap](escha-v047-sm120-next-sprint.md).
Reusable execution goal: [goal prompt](escha-v047-sm120-next-sprint-goal.md).
Preserve and promote validated cumulative gains even below the final release
threshold. Historical experiment decisions retain their original context.

## Source and build

- BeeLLaMA Preview source pin: `1156b183630eb8bd76ff931348d0b1a258ae171b`.
  Local Escha candidate: `/home/sean/work/escha-beellama-v047-candidate`, an
  uncommitted merge based on `2d1411c2c9eaca7d92bdd6a84279e34e580c4cf7`.
  The same source files were synced to z840. The current tracked
  diff against the Escha base, including the pinned Preview merge, is saved as
  `/home/sean/work/escha-v047-current-source.patch` (SHA-256
  `26b03479206139eec888fcae565fe4a627b62751f0ae16b364cc2933d5f2d85a`).
- SM89 CUDA 12.8 Release build:
  `z840:/home/sean/work/escha-beellama-v047-candidate/build-sm89/bin`.
  `libllama.so` SHA-256 `34b79eff3ded96aafc80467ff01e039102a6e0464f6c95f7e58bd4a845a1f7f8`;
  `llama-bench` SHA-256 `12954fb794aa6c920202691ba8dc929f65596f9d9a4e931539bb4f4261465f91`.
- SM120 cross-build: `z840:/mnt/storage/ai-builds/escha-v047-sm120a` from the
  same source, CUDA 13.0.88 copied from local `/usr/local/cuda-13.0` to
  `z840:/mnt/storage/ai-builds/cuda-13.0`. z840 has 78 GiB RAM; this avoids
  compiling large CUDA templates on the memory-constrained 5090 host. Config:
  Release, `CMAKE_CUDA_ARCHITECTURES=120a`, `GGML_CUDA=ON`, FlashAttention and
  KVarN on, `GGML_NATIVE=OFF`, UI/CURL off. Targets: `llama-server`, `llama-cli`,
  `llama-bench`. Build log: `/mnt/storage/ai-builds/escha-v047-sm120a-build.log`.
  Build completed successfully. The `llama-server`, `llama-cli`, and
  `llama-bench` targets linked; `libggml-cuda.so` contains `sm_120a` cubins.
  Runtime `libllama.so` SHA-256
  `865e4690d82946eaadf1c54d7b56c234471441b586e0f49da33a49082056bc67`;
  `llama-bench` SHA-256
  `99f8b839d9dac1070837e7d4bcdff627b3373d9683f53f046e937847f3a45dfb`.
- Source-critical file hashes match local and remote: `escha-moe.cu`
  `7043844d73216d4d721945948eb8612332470605691245d260998e99a7c46ea3`,
  `src/models/dflash.cpp`
  `3465774d5bcf57d492942031bf6743707eb9f235a469eb8c09221fc52316e5c5`.
- Rebase source patch against the pinned Preview commit:
  `/home/sean/work/escha-v047-sm120-rebase.patch` SHA-256
  `886c10a96d027eab0126c43cfd0e5befcf4b8fb14a1950af389b78de0ee5cfa9`.
  It applied cleanly in detached `/home/sean/work/escha-v047-rebase-check`;
  all 56 changed source/text files match the working candidate byte-for-byte
  (`tuning/v047-rebase-source-manifest.json`). The large converter
  `metadata-template.json` and two `dep_k*.npy` tables are separately hashed
  converter inputs, not runtime build dependencies. The full binary patch is
  `/home/sean/work/escha-v047-sm120-from-preview.patch` SHA-256
  `b9eed23ae577fd9a2ccd2157c46c91d14fe5522fb2cfd2ab672d3eaf4247fdac`.
- Portable relink: CMake install runpath `$ORIGIN`; copied to
  `build-sm120a-portable/bin` and bundled with CUDA 13.0 `libcudart`, `libcublas`,
  and `libcublasLt`. The driver library remains host-provided. Portable
  `libllama.so` SHA-256 `5e5f4cd8db75c32533e5b5b2bab69449d524dde403f6880c7686c6809d8a66ab`;
  portable `llama-bench` SHA-256
  `b9b79bc77c5c986e454a9f020567469d158315d27e72c053183c0ea6297294a7`.
  Relink changed ELF hashes, not source or CUDA kernels.

## Comparison inputs

Use the same merged release-model GGUFs on both hosts, with no MTP enabled for
the first bridge/target speed screen. The z840 hashes match the historical
5090 merged-artifact hashes:

| Model | SHA-256 | z840 path | 5090 path |
|---|---|---|---|
| E3 | `746bd40841fb18b9c1918923e89c007df70b5e4f3de64290d1399593e70c96b0` | `/home/sean/escha-assets/escha-e3-with-mtp.gguf` | `/home/sean/kernel-lab5090/escha-mtp/escha-e3-with-mtp.gguf` |
| W2 | `3f93cbe77a20f1fa7272741757596cac66a66457d7ecaed1e5a6e4baa409535e` | `/home/sean/escha-assets/escha-w2-with-mtp.gguf` | `/home/sean/kernel-lab5090/escha-mtp/escha-w2-with-mtp.gguf` |

SM89 selected bridge: `z840:/home/sean/kernel-lab/escha-sprint-rc-sm89-fast/bridge`,
code-GEMM decode cubin and the named Sprint route flags. SM120 selected bridge:
`/home/sean/kernel-lab5090/preview-bridge-sm120`, using the qualified legacy
wrapper `libescha_official_bridge_cuda_sm120.so` SHA-256
`459e04d492a32a1d46df078341276642962dcac50f59846410df8d555551a0f9`,
code-GEMM cubin `6cdbf1ab440958d570ef0b3425df5fcea14c40f48c02d5774da983514d676d8b`,
direct-F32 decode cubin `04692f328b386967c02b68a2061e6ec846675f82e516978a7f1ab9555c99e39b`
with `ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32`, and GDN wrapper
`32ad67be3344759298940db0e1b70315ea1d9c43b5ea3491b939f57ec90b0544`.
The existing SM120 `official_bridge_manifest.json` names a different `_mtp.so`
wrapper, so a release manifest must name the **actual** qualified wrapper.

Fast screen harness: `scripts/escha-v047-parity-bench.sh`; same route selectors
except architecture-specific bridge payload/F32 ABI. First measure F16-KV
bridge isolation: p2048/n0, B2048/UB2048 and p0/n256, B2048/UB512,
threads8, FA on, full GPU offload. Two runs screen; at least five paired runs
confirm. The release-shape KVarN3/2 and speculative/MTP server workload is a
separate required speed and qualification gate; a F16 bridge screen alone does
not prove it. Compare E3 prefill, E3 decode, W2 prefill, W2 decode separately.
The first screen includes SM89's R248 down-add-RMS, R244 split-12, R234
QKV/Z beta-alpha overlap, and E3 GPU embedding setting. The GBrain page
“Escha SM120 stretch plan — 90 decode / 3500 prefill” identifies those as
concrete profile gaps in the older SM120 preview; this is a higher-value first
move than a broad new-kernel sweep.

## Runtime dependencies and state

- The local RTX 5090's `qwen-flash-next-350-static.service` on port 8895 was
  stopped after its generation requests and connection ended. The user said
  the 5090 is free and explicitly does not need the Qwen service restored.
- z840 port 8080 model server was restored and health-checked after tuning.
  The temporary command and environment capture was removed.

## Experiment log

| ID | Change | Predicted benefit | Measured four-cell result | Decision |
|---|---|---|---|---|
| BLD-01 | Same v0.4.7 candidate, CUDA 13.0 SM120a cross-build on z840 | Enable 5090 baseline without local RAM pressure | Build and link succeeded; `sm_120a` cubins present | Keep |
| SCR-01 | Same v0.4.7 source and E3/W2 GGUF hashes; SM89 Sprint route versus SM120 port, F16 KV bridge screen, 2 repeats | Reach all four SM89 speed cells before long qualification | SM120/SM89 median ratios: E3 prefill 1.047, E3 decode 1.308, W2 prefill 1.065, W2 decode 1.360. Receipts `tuning/v047-{sm89,sm120}-parity-screen-20260923/` | Pass screen |
| CNF-01 | Repeat exact four workloads five times per card/model | Confirm all four medians and quantify cold first sample | E3 prefill 2771.73 → 2964.63 (1.070); E3 decode 70.79 → 90.34 (1.276); W2 prefill 2751.99 → 2945.98 (1.070); W2 decode 62.70 → 84.03 (1.340) tok/s. Receipts `tuning/v047-{sm89,sm120}-parity-confirm-20260923/` | Pass F16 bridge gate; release-shape KVarN/MTP gate and functional qualification remain |
| FUN-01 | v0.4.7 SM120 `llama-server`, E3 and W2, KVarN3/2, embedded MTP N2, 8K context, two deterministic controls | Confirm model load, bridge ABI, draft acceptance, output/logit stability and cleanup | Both models returned `PARITY_OK` twice; selected token IDs and log probabilities matched exactly across repeats; first-token top-20 overlap 20/20. Bridge wrapper was mapped; KVarN `decode-split` and `prompt-generic-mma` observed; MTP accepted 2/2 on controls. GPU returned to 16 MiB. Receipts `tuning/v047-sm120-{e3-mtp-abi,w2-mtp}-smoke-20260923/` | Pass bounded control |
| FUN-02 | Same server route with DFlash2 Q4 draft on GPU | Confirm LowGPU DFlash placement and parity with MTP | E3 and W2 returned `PARITY_OK` twice; selected token IDs and chosen log probabilities equal MTP for each model; draft accepted 2/2 on short controls; bridge mapped; GPU returned to 16 MiB. Receipts `tuning/v047-sm120-{e3,w2}-dflash-smoke-20260923/` | Pass bounded control |
| KVR-01 | Matched KVarN3/2 llama-bench four-cell screen, two repeats | Confirm release-cache route still exceeds SM89 | SM120/SM89 median ratios: E3 prefill 1.083, E3 decode 1.226, W2 prefill 1.081, W2 decode 1.306. Receipts `tuning/v047-{sm89,sm120}-kvarn-screen-20260923/` | Pass screen |
| KVR-02 | Same KVarN3/2 cells, five repeats each card/model | Confirm release-cache speed gate | E3 prefill 2616.37 → 2824.18 (1.079); E3 decode 65.77 → 80.84 (1.229); W2 prefill 2602.01 → 2834.28 (1.089); W2 decode 58.47 → 74.13 (1.268) tok/s. Receipts `tuning/v047-{sm89,sm120}-kvarn-confirm-20260923/` | Pass KVarN bridge speed gate; speculative server and long-context gates remain |
| LNG-01 | E3/W2 KVarN3/2 + MTP N2, 32K context, fresh 28,841-token needle at 10/50/90% depth | Long prompt retrieval, cache reuse, fit and cleanup | All six returned `BLUE-FLAMINGO-42`; mid-depth repeat cached 28,800 tokens for both models. Post-long GPU allocation E3 ~11 GiB, W2 ~15 GiB; GPU returned to 16 MiB after each process. Receipts `tuning/v047-sm120-{e3,w2}-long-{early,mid,late}-20260923/` | Pass bounded 32K-context ladder; larger contexts unqualified |
| NAT-01 | Native LowGPU IQ3XXXS GGUF, same v0.4.7 SM120 binary, p2048/d256, F16 and KVarN3/2, five repeats | Same-card standard-GGUF speed anchor | Native F16 prefill 3419.63/decode 103.75; native KVarN3/2 prefill 3267.14/decode 88.45 tok/s. F16 Escha E3 is 86.7%/87.1%, W2 86.1%/81.0% of native prefill/decode; KVarN E3 is 86.4%/91.4%, W2 86.8%/83.8%. Receipts `tuning/v047-sm120-native-iq3-{f16,kvarn}-20260923/` | Gap recorded; native SGLang and effective speculative server comparison open |
| NAT-02 | Same native IQ3 GGUF and v0.4.7 binary on SM89/SM120, five repeats each, F16 and KVarN3/2 | Measure actual hardware scaling for this model family | SM120/SM89 native IQ3 ratio: F16 prefill 1.286/decode 1.388; KVarN prefill 1.301/decode 1.281. Receipts `tuning/v047-{sm89,sm120}-native-iq3-{f16,kvarn}-20260923/` | Escha KVarN decode scaling 1.229 E3/1.268 W2 is near native 1.281; prefill scaling 1.079/1.089 falls short of native 1.301 |
| API-01 | Native W2 safetensors on Escha SGLang, RTX 5090, single request 2K prompt/forced 256 decode, 5 measured + 1 warmup | Intended-runtime same-card reference | Client medians 3076 prefill/83.30 decode tok/s. Runtime config in `tuning/v047-sglang-w2-native-20260923/server.log`; per-request `tuning/v047-sglang-w2-native-api-speed-20260924/` | Reference; original safetensors head differs from merged GGUF head, so interpret as runtime comparison |
| API-02 | Bee v0.4.7 SM120 W2 server, same API workload, F16 no draft; KVarN3/2 MTP N2; plus native IQ3 F16 server | Measure served prefill/decode and draft acceptance | Bee W2 F16 no-draft 2204/78.68; KVarN MTP 1905/82.18; Bee native IQ3 F16 2882/94.63 tok/s. MTP synthetic output acceptance 60.4%. Receipts `tuning/v047-bee-{w2-f16,w2-kvarn-mtp,native-iq3-f16}-api-speed-20260924/` | Packed Escha prefill remains the release gap; MTP raises effective decode only modestly |
| API-03 | Bee E3 KVarN3/2 MTP N2; W2 DFlash2 Q4 N5 short speed arm | Test effective E3 and one plausible draft depth improvement | E3 MTP 1884/82.70 tok/s, 53.5% acceptance. W2 DFlash2 N5 1916/81.16, 37.6% acceptance versus W2 MTP 1905/82.18. Receipts `tuning/v047-bee-{e3-kvarn-mtp,w2-kvarn-dflash5}-api-speed-20260924/` | Keep MTP N2; reject DFlash2 N5 for speed preset |
| API-04 | W2 KVarN MTP N2 server, two-run UB2048 repeat | Check served batch sensitivity | 1859 prefill/80.54 decode tok/s versus prior 1905/82.18; `tuning/v047-sm120-w2-ub2048-mtp-screen-20260924/` | Prior server log also records target UB2048, so this was a repeat, not a UB512→2048 comparison. Discard the original batch-size conclusion. |
| PKG-01 | Five-repeat four-cell recheck on exact portable package binary and bundled bridge | Verify `$ORIGIN` relink preserved absolute SM89 parity | F16 E3 2939.35/89.17, W2 2933.33/83.57; KVarN3/2 E3 2803.63/80.36, W2 2813.22/74.35 tok/s. All eight cells beat the SM89 reference. Receipts `tuning/v047-sm120-package-{f16,kvarn}-confirm-20260924/` | Pass exact packaged binary speed gate |
| COR-01 | Exact SM89 and packaged SM120 E3/W2 GGUF hashes; six fixed prompts, two repeats per prompt, KVarN3/2 MTP N2, top-20 first-token logprobs | Cross-architecture deterministic output and selected logit check | All 12 E3 and 12 W2 paired responses matched text and selected token sequence. Top-20 overlap ≥19/20; largest chosen-token absolute logprob difference E3 0.020, W2 0.032. Receipts `tuning/v047-crossarch-{e3,w2}-corpus-compare-20260924.json` plus raw `tuning/v047-{sm89,sm120}-*-crossarch-corpus*` | Pass bounded deterministic corpus; alternative ranks vary slightly |
| PKG-02 | r2 opt-in `doctor --report` on exact E3/W2 models | User install diagnostics, local launcher health/control, memory and report privacy | Both passed. E3 observed GPU peak 10,854 MiB, host RSS peak 9,636,756 KiB; W2 14,690 MiB / 11,202,536 KiB; both returned to 16 MiB; reports contained no full home path or prompt/output. Receipts `tuning/v047-sm120-r2-doctor-{e3,w2}-20260924.json` | Keep report feature |
| BRG-01 | Select existing vendor `(BM64, BN64, BK3)` cubin for only K3 `IC17408→OC5120` FP32-accum prefill down projection; same-source control uses `(BM128, BN64, BK2)` | Improve dominant packed FFN without changing model or runtime binary | Five alternating process pairs, exact E3/W2 model and portable binary hashes: F16 E3 2946.65→3438.41 prefill, 89.22→89.19 decode; W2 2934.99→3437.50, 84.93→84.56. KVarN3/2 E3 2811.63→3162.28, 79.52→80.19; W2 2825.81→3243.64, 75.27→75.12. Median paired prefill gain 13–17%; decode within 1% in medians. `tuning/v047-sm120-bm64-five-paired-20260924/` | Keep candidate for further qualification; source/build hashes in bridge probe manifest. |
| COR-02 | Candidate versus same-source control, E3/W2 KVarN3/2 MTP N2, six prompts twice | Check output preservation after speed pass | All 12 E3 and 12 W2 responses, selected token sequences, first-token top-20 sets and recorded chosen logprobs matched exactly. `tuning/v047-sm120-bm64-{e3,w2}-corpus-compare-20260924.json` | Pass bounded candidate corpus. |
| API-05 | Same-source control versus BM64 candidate, KVarN3/2 MTP N2 server, five measured requests each, target UB2048 and UB512 | Measure actual served path at matched batch settings | UB2048: W2 prefill 1860→2079/decode 83.00→83.16; E3 prefill 1870→2030/decode 81.88→80.39 tok/s. UB512: W2 1532→1672; E3 1544→1740 prefill tok/s. The older packaged bridge UB2048 medians were W2 1905, E3 1884. Receipts `tuning/v047-sm120-bm64-{control,candidate}-{e3,w2}-api-ub{512,2048}-20260924/` | Keep BM64 candidate; use UB2048 only in the short-context fast profile, pending its separate check. No shape gate. Native W2 SGLang served prefill remains faster at 3076 tok/s. |
| API-06 | Original E3 safetensors on native Escha SGLang, RTX 5090, same client 2K prefill/256 decode workload, five measured requests | Add missing native E3 runtime reference | 3278.74 prefill/82.53 decode tok/s, 20,976 MiB loaded, 16 MiB after stop. Receipt `tuning/v047-sglang-e3-native-20260924/`. Model files and runtime configuration are recorded there. | Candidate Bee E3 MTP decode is close (80.39), but served prefill is 61.9% of this native runtime reference. Different merged GGUF versus original safetensors head makes this a runtime comparison. |
| LNG-02 | Candidate and same-source control at target UB2048; candidate at UB512/1024/1536, 32K E3 mid-depth 28,841-token needle repeated twice | Check whether faster served batch setting preserves long-context retrieval | Both candidate and control at UB2048 answered `00000` twice (fail); candidate E3/W2 UB1024 and UB1536, and E3 UB512, answered `BLUE-FLAMINGO-42` twice (pass), with cache reuse and GPU cleanup. Receipts `tuning/v047-sm120-bm64-*-long-mid*-20260924/`. | UB2048 cannot be the 32K default. Keep UB1024 for 32K, since UB1536 measured slower in served speed; isolate UB2048 to a separately qualified 8K fast profile. |
| API-07 | BM64 candidate MTP server target UB1024 and UB1536, E3/W2, five measured requests each | Find safe 32K default with good served speed | UB1024: W2 1939.72/81.89, E3 1927.24/80.48 prefill/decode tok/s. UB1536: W2 1907.24/82.36, E3 1890.99/81.70. Receipts `tuning/v047-sm120-bm64-candidate-{e3,w2}-api-ub{1024,1536}-20260924/`. | Use UB1024 for `long-32k` default; reject UB1536 for no measured prefill gain. |
| LNG-03 | BM64 candidate E3/W2, 8K context, UB2048, 4,841-token mid-depth needle twice with cache repeat | Bound the short-context fast profile | Both E3 and W2 returned `BLUE-FLAMINGO-42` twice, mapped the bridge, reused prompt cache, and returned GPU memory to 16 MiB. Receipts `tuning/v047-sm120-bm64-candidate-{e3,w2}-fast8k-needle-20260924/`. | Keep `fast-8k` as opt-in profile; longer prompts at UB2048 remain unqualified and the 28,841-token check failed. |
| PKG-03 | Fresh r4 extraction, complete SHA256SUMS, E3/W2 `doctor --report`, selected bridge corpus versus SM89, and direct `./escha serve` on both named profiles | Qualify install and profile selection on exact packaged bytes | All manifest-listed payload hashes passed; both doctor reports passed model hash, GPU, launcher and control; both models passed all six cross-architecture prompts twice with identical selected token sequences; all four launcher/model cases mapped the selected bridge, returned `PARITY_OK`, logged the expected context/UB, and released GPU allocation. Receipts `tuning/v047-sm120-r4-{archive-doctor-*,*crossarch*,profile-launcher}-20260924*` and final archive hash check. | r4 is a reviewable experimental prerelease for the named RTX 5090/Ubuntu 24.04 target and two profiles. |
| API-08 | Exact final r4 binary/bridge on `fast-8k`, KVarN3/2 MTP N2, five measured requests on each E3/W2 GGUF | Measure the actual opt-in profile against native runtimes with 8K context | W2 prefill/decode 2157.39/80.88; E3 2061.49/82.23 tok/s. Native W2 SGLang 3076/83.30 and E3 3278.74/82.53 on same RTX 5090. Receipts `tuning/v047-sm120-r4-{e3,w2}-api-fast8k-mtp-20260924/`. | Decode near native SGLang; served prefill still 70% W2 and 63% E3 of native reference, so full native-runtime speed parity remains open. This measurement was made after r4 archive finalization and is in the source ledger, not the archived ledger snapshot. |
| API-09 | Exact final r4 W2 KVarN3/2 server at 32K/UB2048 with MTP disabled | Isolate speculative bookkeeping in served prefill | No-draft prefill/decode 2257.70/71.88 tok/s versus MTP 2078.63/83.16. Receipt `tuning/v047-sm120-r4-w2-api-kvarn-no-draft-ub2048-20260924/`. | MTP costs about 8% prefill but materially improves decode; it does not explain the entire served prefill gap. This is diagnostic only; 32K/UB2048 fails long retrieval and is not a release profile. |
| SHP-01 | W2 exact r4 bridge, fixed B/UB2048, warm bench p2048/p2047/p2041/p1024 and paired served p2048/p2047 | Test DeepSeek's exact-2048 route-cliff hypothesis | Bench p2048 3312.78 tok/s versus p2047 2843.10 and p2041 2845.08; this raw p2048 result used the later-invalid all-NaN K1 GDN bridge, so the 14% cliff is **not valid speed evidence**. Served adjacent prompts were flat: 2140.90 versus 2123.84 tok/s over five alternating pairs. Temporary bridge trace showed code-GEMM `M=2048` in bench and `M=1920` in server. Receipts `tuning/v047-sm120-r4-{shape-cliff,w2-server-shape-cliff}-20260924/`. | Discard the GDN-enabled bench inference. Served prompt length did not recover speed; corrected GDN-off shape screen is SHP-04. |
| SHP-02 | Isolated SM120 source build admitting 1920 rows into the raw SwiGLU fused gate/up route; r4 binary/bridge/model otherwise matched | Recover the server's skipped exact-2048 SwiGLU path without new kernel | z840 candidate `libggml-cuda.so` SHA-256 `d7300f588449431ae5426d81881f18f20ac1ebcd2a92292329c50e7d8745ef1e`; original z840 source/lib restored to hashes `7043844d...`/`4c84d8cf...`. Same-session W2 fast-8k two-run candidate prefill 2103.53 versus r4 control 2103.66 tok/s; decode 82.95 versus 83.87. Receipts `tuning/v047-sm120-shape1920-w2-fast8k-{api,control}-screen-20260924/`. | Reject: no prefill gain. Do not spend a correctness suite on this regression/neutral route. r4 remains unchanged. |
| SHP-03 | W2 r4 server B/UB2176, same 2048-token prompt, temporary bridge row trace | See whether a larger physical batch restores `M=2048` | Server still passed `M=1920` to the bridge; one-run prefill 2086 tok/s. Receipt `tuning/v047-sm120-r4-w2-b2176-ub2176-trace-20260924/`. | Reject simple batch-size workaround; shape comes from the server request path, not the physical batch cap. |
| GDN-01 | Isolated W2 KVarN p2048 bench with the retained GDN bridge flag toggled | Test whether GDN chunk routing explains the served gap | The on/off warm numbers were 3324/2890 tok/s (`tuning/v047-sm120-r4-gdn-bridge-ab-20260924/`). A later r4 `LD_DEBUG=libs` run proved the packaged GDN wrapper loaded at p2048; disassembly of that exact binary shows it accepts 128-element head strides. The adjacent wrapper `.cu` source was stale with a guard of 1 and has been aligned to 128 without changing r4 bytes. Receipt `tuning/v047-sm120-gdn1920-k1-bench-screen-20260924/p2048-r4-lddebug.log`. | Invalid timing comparison: the enabled path produces all-NaN logits (GDN-05). |
| GDN-02 | Trace the served W2 fast-8k MTP GDN operator and build an isolated T1920 wrapper | Find the actual route shape before broadening dispatch | The real prefill is `n_tokens=1920`, `K=3`, `H=48`, `S_v=128`, one sequence, ordinary non-KDA. The T1920 K1 wrapper was not mapped in the MTP server. Receipts `tuning/v047-sm120-gdn1920-w2-{shape-trace,shape-large}-20260924/`. | K3 rollback snapshots require a separate implementation; the retained K1 bridge cannot serve MTP as-is. |
| GDN-03 | Fix the head-stride guard in an isolated T1920/K1 wrapper and explicitly trace calls | Establish whether the cached chunk cubins can run this shape | The bridge was called on W2 p1920 KVarN bench; two-run warm prefill was 3193.88 versus 2861.20 tok/s for the baseline. Receipt `tuning/v047-sm120-gdn1920-k1-bench-screen-20260924/`. | Invalid speed lead: later K1 logits audit found the bridge state/output wrong (GDN-05/07). |
| GDN-04 | Isolated K3 split: T1856 chunk prefix plus native CUDA 64-token tail to write all three snapshot slots | Preserve MTP rollback while accelerating most of prefill | Bridge mapped and logged 144 calls. W2 fast-8k two-run prefill/decode was 2174.98/38.65 tok/s versus r4 2157.39/80.88. The server logged zero accepted draft tokens out of 507 generated on each decode request, so this is not a valid speed improvement. Receipt `tuning/v047-sm120-gdn1856-k3-w2-fast8k-screen-20260924/`. | Reject: prefill is within noise and decode regresses sharply; snapshot/output compatibility is unproven. Do not run a correctness suite or package this route. |
| GDN-05 | Exact r4 packaged binary and W2 GGUF, focused K=1 p2048 one-batch logits with GDN bridge enabled versus disabled | Qualify the fast bench route before relying on its throughput | Enabled bridge loaded and all 248,320 output logits were NaN; disabled bridge yielded 248,320 finite logits. The same native IQ3 GGUF also gave all-NaN enabled and all-finite disabled logits. Exact harness and raw outputs: `tuning/v047-sm120-gdn1920-k1-bench-screen-20260924/gdn-k1-logits.cpp`, `logits-{on,off,iq3-on,iq3-off}.f32`. | Withdraw all GDN-enabled K=1 p2048 speed and native IQ3 comparison claims. Keep GDN disabled in a repaired bundle; do not ship its broken wrapper/cubins. MTP K3 served route did not load the wrapper. |
| GDN-06 | Exact r4 binary and selected BM64 code-GEMM bridge with GDN explicitly disabled; E3/W2 F16 and KVarN3/2 bench, five samples per cell | Re-establish valid absolute SM89 speed gate | F16 E3 2980.36/90.77, W2 2952.68/85.09; KVarN E3 2869.16/79.67, W2 2790.75/75.62 prefill/decode tok/s. W2/E3 K=1 GDN-disabled logits were finite. Receipts `tuning/v047-sm120-r4-gdn-disabled-{f16,kvarn}-five-20260924/` and the focused logits harness. | All eight corrected SM120 cells remain above the recorded SM89 rates. The earlier GDN-enabled native IQ3 baseline and hardware-scaling ratios are invalid until replaced. |
| GDN-07 | DeepSeek 4.1 read-only review of cached Triton IR and wrapper launch arrays after the NaN finding | Identify a specific root cause before further kernel work | Review flagged a state-ABI mismatch and disconnected solve/merge buffers. Subsequent `cuobjdump --dump-elf` showed the **original state cubin has 14 parameters**, with `T` in slot 11, matching the original wrapper's 14 arguments; its adjacent TTIR has only 13. Thus the review's specific `T=0` diagnosis is unproven. The original KKT cubin and TTIR both have nine parameters; buffer typing and nominal argument order match. | Keep GDN quarantined; cached TTIR cannot be assumed to describe its neighboring cubin. Require finite full-batch logits and numerical parity before any speed screen or K3 work. |
| GDN-08 | Isolated K1 bridge repair: source head-stride guard, state/sequence pointers, per-stage device scans, and one KKT rebuild from cached TTIR | Resolve all-NaN logits using one targeted high-value kernel check | W2 p2048 input keys, values, beta, and gates were finite (key max abs 0.926, beta 0.065–0.999). Original KKT cubin wrote lower-triangular `A` up to 1.62e38 from those inputs; inverse/recompute/state then produced NaNs. Recompiled KKT TTIR cubin (57,120 bytes, shared 16,384) caused an illegal device memory access at the KKT stage, before logits, despite the same nine-parameter ELF layout. The isolated repair still fails the full-logits gate. Logs `/home/sean/work/escha-v047-gdn-k1-repair/w2-logits-{inputs,recompiled-kkt}.log`; r5 package was not changed. | Stop this GDN branch without speed timing or broad correctness testing. Further work needs a matched source/cubin provenance and an independently checked KKT contract. |
| NAT-03 | Same-card native IQ3 GGUF and exact r4 binary, GDN explicitly disabled, five F16 and KVarN3/2 bench samples per cell | Replace the invalid native IQ3 anchor | F16 prefill/decode 3342.83/103.86; KVarN3/2 3104.70/90.50 tok/s. The native IQ3 K1 logits were all NaN with the old bridge and all finite with it off; model SHA-256 `ad85e40a28aafd907eebb6ff6b21786b897dd750b0918427f1243d6d84ebcc72`. Receipt `tuning/v047-sm120-native-iq3-gdn-disabled-five-20260924/`. | Use this same-card SM120 native benchmark for raw speed comparison; see NAT-04 for the matched SM89 reference. |
| NAT-04 | Exact same native IQ3 GGUF on z840 SM89 with GDN=0, five F16 and KVarN3/2 samples, matched in configuration to NAT-03 | Establish corrected 4090→5090 hardware scaling | SM89 F16 prefill/decode 2654.31/74.79, KVarN3/2 2511.71/69.01 tok/s. Native SM120/SM89 ratios F16 1.259 prefill/1.389 decode; KVarN 1.236 prefill/1.311 decode. Source model hash matches NAT-03. Receipts `tuning/v047-sm89-native-iq3-gdn-disabled-{f16,kvarn}-five-retry-20260924/`; corrected scale summary `tuning/v047-sm120-corrected-hardware-scaling-20260924.json`. | This replaces NAT-02's GDN-enabled scaling anchor; z840 port-8080 server was restored healthy. |
| GDN-09 | z840 SM89 E3/W2 four-cell five-repeat rerun with GDN explicitly disabled, using the corrected configurable harness | Match SM89's bridge mode to r5 before claiming absolute/hardware parity | The first rerun inherited z840's old harness line `ESCHA_GDN_CHUNK_BRIDGE=1`; discard its mislabeled `...gdn-disabled-...-five-20260924/` receipts. The validated copied harness retains `ESCHA_GDN_CHUNK_BRIDGE=0` and excludes GDN payload from its hash list. SM89 F16 E3 2421.59/70.48, W2 2408.52/62.70; KVarN3/2 E3 2300.93/65.79, W2 2286.29/58.43 prefill/decode tok/s. Matched SM120/SM89 ratios F16 E3 1.231/1.288, W2 1.226/1.357; KVarN E3 1.247/1.211, W2 1.221/1.294. Receipts `tuning/v047-sm89-escha-gdn-disabled-{f16,kvarn}-five-validated-20260924/` and `tuning/v047-sm120-corrected-hardware-scaling-20260924.json`. | All eight absolute cells pass. KVarN prefill hardware scaling reaches 100.9% native IQ3 for E3 and 98.8% for W2; E3 decode scaling reaches only 92.3% of native IQ3 scaling, and served prefill still trails SGLang. z840 service restored healthy. |
| PKG-05 | r6 rebuilt from frozen r5 runtime bits plus corrected docs, exact r5 qualification and validated SM89/native receipts | Give reviewers one install asset with trustworthy speed provenance | Archive SHA-256 `13903183e73a611f77824d80eb2eb15e39ce75b6e4693dae27d7d3f308ceb3e6`; fresh extraction passed all 290/290 payload hashes. E3/W2 `doctor --report` passed model hashes, launcher control and GPU cleanup. r6 launcher, profile, binary, libllama, CUDA library and code-GEMM bridge match r5 byte-for-byte; no GPU kernel behavior changed. `tuning/v047-sm120-r6-hashcheck-20260924.txt`, `tuning/v047-sm120-r6-doctor-{e3,w2}-20260924.json`. | r6 supersedes r5 for local review; still bounded RTX 5090/Ubuntu 24.04 WSL experimental, not a native-speed-parity or public source release. |
| SHP-04 | Exact r6 W2 binary/bridge, GDN off, KVarN3/2, B/UB2048, p1920 versus p2048, three repeats per shape | Screen whether the actual server `M=1920` code-GEMM shape alone explains the native SGLang served prefill gap | Median raw p1920 2778.79 tok/s (690.95 ms), p2048 2890.96 (708.41 ms): 1920-token rate is 96.1% of 2048-token rate. The prior exact r5 fast8k MTP API median prompt time is 969.36 ms for 2048 tokens; a different long32k no-draft profile measured 903.19 ms. `tuning/v047-sm120-r6-w2-p1920-shape-screen-20260924/summary.json`. | Reject another shape-only code-GEMM edit. Shape explains a small raw-rate difference; direct server phase timing is needed before attributing the remaining gap. Do not treat bench/API latency difference as a causal decomposition because workload/context differ. |
| API-10 | Exact r6 W2 fast8k KVarN3/2 B/UB2048, no-draft three-request screen versus qualified r5 byte-identical MTP N2 five-request reference | Bound speculative overhead and assess a no-draft speed preset | No-draft median prefill/decode 2295.87/69.27 tok/s, prompt 887.92 ms; MTP N2 reference 2104.24/81.89, prompt 969.36 ms. No-draft gains 9.1% prefill but loses 15.4% effective decode on this synthetic workload. Same model/runtime bits; `tuning/v047-sm120-r6-w2-api-fast8k-no-draft-screen-20260924/` and r5 MTP receipt. | Keep MTP in the recommended speed profile; no-draft misses the native SGLang decode reference and does not close its prefill gap. Do not spend long-context/correctness qualification on this rejected primary profile. |
| API-11 | Exact frozen r6 server/bridge W2, KVarN3/2, 8K B/UB2048; temporary C API wall-time probe then `-ctxcp 0` three/five-request screens | Explain served prompt overhead without changing kernels | The normal 2048-token request runs as 1920+128 target batches because the near-end checkpoint is aligned down to 1920 for KVarN group 128; measured extra 128-token `llama_decode`+sync is about 125 ms. With checkpoints off, W2 no-draft prefill/decode is 2886.59/72.21 versus 2295.87/69.27; MTP N2 is 2635.28/83.88 versus exact-byte r5 2104.24/81.89 tok/s. Probe and responses in `tuning/v047-sm120-r6-server-phase-probe-20260924/`. | Keep the checkpointed cache profile. `-ctxcp 0` is useful only as a separate stateless 8K speed profile because KVarN repeat requests lose prompt reuse. |
| API-12 | Same r6 bits, `-ctxcp 0`, MTP draft microbatch 256 versus 64 | Screen one bounded speculative prefill overhead knob | W2 three-request median prefill/decode 2734.41/83.36 versus five-request UB64 2635.28/83.88 tok/s; E3 UB256 three-request 2751.08/77.49 versus five-request UB64 2631.18/81.58. MTP acceptance unchanged per model. Receipts `tuning/v047-sm120-r6-server-phase-probe-20260924/{w2,e3}-mtp-draftub256-no-checkpoints/`. | Use draft UB256 for W2 only in the stateless profile; reject it for E3 because the observed decode regression exceeds 5%. This is a small-screen result until exact launcher confirmation. |
| LNG-04 | r6 E3/W2 MTP, 8K UB2048, `-ctxcp 0`, draft UB256, 4,841-token mid-depth needle twice after two short controls | Bound the stateless setting after its speed pass | Both models returned `PARITY_OK` twice and `BLUE-FLAMINGO-42` twice, mapped the bridge, and returned GPU use to 16 MiB. Both repeated long requests reported `cache_n=0`, `cache_reason=no_restorable_kvarn_boundary`, and reprocessed all 4,841 tokens. Receipts `tuning/v047-sm120-r6-server-phase-probe-20260924/{e3,w2}-fast8k-needle-no-checkpoints/`. | Correct output is bounded at 8K; prompt reuse is absent. Keep the existing checkpointed `fast-8k` and `long-32k` profiles for cache-dependent use. |
| PKG-04 | Fresh r5 extraction; exact GDN-disabled archive, pinned E3/W2 models, both launcher profiles, long retrieval, corpus, K1 logits and five-request fast API | Recover a safe, reviewable install asset after GDN invalidation | Archive SHA-256 `0d0f66b1376de9feaea7f1abfbb43c8a361ece9a1840278fedbbc943d7c720f8`; all 276 payload hashes pass; GDN payload absent and flag 0. E3/W2 doctors and four launcher cases pass. Both 28,841-token mid-depth 32K requests pass twice with cache reuse, GPU back to 16 MiB. Each model matches all 12 SM89 corpus responses/selected token sequences; max chosen-logprob delta E3 0.020, W2 0.032, min top20 overlap 19. Exact r5 full p2048 K1 logits all finite and byte-identical to GDN-disabled control. Fast8k MTP five-request medians E3 2124.67/82.15, W2 2104.24/81.89 prefill/decode. Receipts `tuning/v047-sm120-r5-*20260924*` and `tuning/v047-sm120-r5-hashcheck-20260924.txt`. | r5 is a bounded RTX 5090/Ubuntu 24.04 WSL experimental review asset. It is not a native-speed-parity or public source release. |

| API-13 | Fresh r7 extraction, exact packaged binary/bridge, 8K stateless MTP profile settings, five measured requests per E3/W2 | Confirm the short screens on installable bytes | E3 draft UB64 prefill/decode 2522.66/78.87; W2 draft UB256 2647.42/82.78 tok/s. The earlier short screens were 2631.18/81.58 and 2734.41/83.36, respectively; report the exact archive five-run values as the conservative comparison. Native SGLang same-card references are E3 3278.74/82.53 and W2 3076/83.30; exact r7 prefill is 77%/86%, decode 95.6%/99.4%. Receipts `tuning/v047-sm120-r7-{e3,w2}-api-stateless-20260924/`. | The opt-in stateless profile improves served prefill versus checkpointed r5, but neither E3 nor W2 meets native SGLang prefill within 5%. Do not claim full parity. |


| GDN-10 | DeepSeek 4.1 read-only review proposed supplying non-null varlen descriptors to the retired K1 GDN bridge; local audit of the manifest-hashed original cubins | Falsify the ABI hypothesis before another GPU run | The neighboring TTIR lists descriptor arguments, but its cached cubin SHA differs from the manifest-hashed original. `cuobjdump --dump-elf` on the exact KKT cubin gives nine parameters and cbank base `0x380`; `--dump-sass` never reads KKT descriptor ordinals 4/5 at `0x3a0/0x3a8`. The exact merge, recompute and output cubins likewise never load their descriptor parameter addresses. All four exact cubin hashes match the manifest. | Reject the proposed NULL-descriptor repair without GPU timing or correctness work. The original KKT huge-value/NaN cause remains unresolved; GDN remains disabled and excluded. |
| PKG-06 | Final r7 local experimental review archive after exact-speed receipts and revised install/rebase notes | Expose independent-request speed without changing cached defaults or published assets | Archive SHA-256 `ee8957763ca183bd5dcd55d5cf42a8774786406b5a1ae6f45d0a7f4ef3233d6d`; fresh extraction passed all 299/299 payload hashes and E3/W2 release `doctor --report` passed model hashes and launcher smokes. The six-case profile smoke used the preliminary r7 package; the final launcher, profile env, server, libllama, CUDA library and selected bridge are byte-identical to that smoke-tested package. Final archive includes API-13 exact-byte speed summaries. Receipts `tuning/v047-sm120-r7-release-hashcheck-20260924.txt`, `tuning/v047-sm120-r7-release-doctor-{e3,w2}-20260924.json`, and `tuning/v047-sm120-r7-profile-launcher-20260924/`. | Stage for local review on the named host/model hashes. No native SGLang prefill parity, no public release/commit/tag. The checkpointed profiles and source rebase patch remain available. |
| DQG-01 | Isolated RTX 5090 transient dequantize plus cuBLAS screen on packed FFN projections, exact r7 bridge/cubin control, random F16 rotated inputs; then same-session W2 served prefill replay with only the K3 down dispatch changed | Test a high-reuse GEMM route before any full correctness qualification | K3 `17408→5120` down at M1920: 2.1706→1.7516 ms, finite output and rel RMS `9.46e-5` versus FP32 vendor cubin. Five measured W2 stateless served prefill requests: candidate 2816.15 versus same-session r7 control 2746.57 tok/s, a 2.53% gain; candidate remains 91.6% of native W2 SGLang 3076 tok/s and uses ~220 MiB transient GPU scratch per stream. K2 gate/K3 up `5120→17408` appeared favorable against an FP32 vendor control (~3.6→1.9 ms), but the actual `mixed` runtime policy uses accumulation mode 1: exact vendor control 1.26/1.25 ms versus candidate 1.93/1.94 ms, with rel RMS ~0.00264. At M1920 the runtime's special raw SwiGLU bridge route is not selected; prior SHP-02 widening with the vendor cubin showed no served gain. Receipts `tuning/v047-sm120-dequant-gemm-screen-20260924/`. | Reject gate/up without a full-model or correctness run. Keep the down-only route as an isolated probe; its 2.53% served gain does not close the ≤5% native-speed gap and does not justify a new package or its scratch cost yet. r7 remains the qualified staged asset. |
| GDN-11 | Isolated W2 p2048 K1 full-logit probe with two independent CUDA contract kernels replacing the cached KKT and 64×64 inverse stages one at a time; exact retained cubins for every downstream stage | Determine whether the original KKT failure is the only GDN blocker | Original KKT produced `A` up to `1.62e38`. Direct BF16-key/float-beta/gate KKT made all 6,291,456 `A` values finite, max abs `0.969`. The cached inverse then produced 2,145,666 NaNs. A direct `(I+A)^-1` lower-triangular solve made all inverse values finite, max abs `1`; the retained recompute cubin then produced `w` up to `1.25e38`, 27,771 NaNs in `u`, and all 248,320 final logits NaN. The local FLA source confirms the expected KKT, inverse and recompute contracts, but numerical parity of the two custom stages was not qualified. Isolated source and logs `/home/sean/work/escha-v047-gdn-k1-repair/gdn_bridge_ported.cu`, `w2-logits-custom-{kkt,kktinverse}.log`; r7 package and profile remain GDN-off. | Reject the cached GDN pipeline as a release candidate. It has at least three failing integration stages, so do not time or run a long correctness suite; resume only with matched source/cubin provenance or a separately bounded native CUDA implementation. |
| GDN-12 | Isolated p2048 K1 bridge with seven source-matched SGLang SM120 cubins and their varlen descriptors, including 16×16 solve before 64×64 merge; W2 and E3 focused logits and three-repeat raw KVarN prefill screen | Revisit GDN only after obtaining matched source/cubin provenance | All intermediate stages and 248,320 final logits are finite on both models. Against GDN-off, top choice is identical for both; top-20 overlap is W2 19 and E3 18; full-logit relative RMS is 0.01647/0.01598. W2 p2048 three-repeat medians: 3204.79 on vs 2873.00 off tok/s (+11.55%); E3: 3219.26 vs 2866.13 (+12.32%). Cubin manifest and source/logits are under `/home/sean/work/escha-v047-gdn-k1-repair/`; speed receipts `tuning/v047-sm120-sglang-gdn-k1-screen-20260924/`. | K1 is a viable isolated prefill probe, not release qualified. It does not cover served MTP's K=3 rollback states. The old cached-cubin GDN-11 failure remains valid only for that mismatched pipeline. |
| GDN-13 | Retired 1856+64 K3 split on W2 stateless 8K, followed by actual GDN shape trace and process map audit | Correct a false attribution before promotion | The stateless `-ctxcp 0` server issues a single GDN batch with `n_tokens=2048`, `K=3`; the split dispatch required `n_tokens=1920`, and the bridge library was absent from `/proc/<pid>/maps` after requests. Earlier apparent +0.69% prefill difference and healthy draft acceptance came from the native GDN path in both processes. Old response timings remain under `tuning/v047-sm120-sglang-gdn-k3-screen-20260924/`, but they are **not** GDN bridge evidence. | Retract the earlier positive attribution; no correctness or speed claim for the 1856+64 split. Retarget only the observed 2048-token K3 shape. |
| GDN-14 | Isolated source-matched SM120 GDN cubins with masked 2045-token prefix and three-token native recurrence tail, old v0.4.7 runtime via function interposition for a speed-first screen | Preserve MTP rollback snapshots while recovering the full K1 chunk gain | Focused W2 K1 p2045 final 248,320 logits all finite; top choice and top-20 set equal GDN-off, full-logit rel RMS 0.03697. Independent p1917 masked K1 output also all finite, same top choice/top-20, rel RMS 0.01976; raw p1917 three-repeat prefill median 3122.90 bridge-on vs 2793.00 off tok/s (+11.8%). The actual served stateless GDN call was traced as `n_tokens=2048,K=3,S_v=128,H=48`; exact 2045 bridge library was verified mapped after a request. Isolated source/logs under `/home/sean/work/escha-v047-gdn-k1-repair/`. | Advance to served K3 speed gate; K1 numerical proximity and partial output agreement do not qualify full model correctness. |
| GDN-15 | W2 served 8K stateless KVarN3/2 MTP N2, draft CPU UB256; exact same v0.4.7 runtime, official bridge, requests and five measured responses with only K3 GDN 2045+3 dispatch on/off | Verify the target workload and native-runtime speed floor | Prefill median 2997.13 on vs 2677.43 off tok/s (+11.94%); decode median 84.52 on vs 86.22 off (−1.97%). The on medians are 97.4%/101.5% of native W2 SGLang 3076/83.30. All five prefill responses and five 256-token decode responses match off exactly. Draft acceptance observed 139/230. Receipts `tuning/v047-sm120-sglang-gdn2045-k3-screen-20260924/{w2-on-five,w2-off-five}.json`. | Pass short native-speed and paired-output screens on isolated source; integrated binary, full prompt-to-decode rollback, corpus, long retrieval, and package qualification remain. |
| GDN-16 | E3 served 8K stateless KVarN3/2 MTP N2, GPU draft UB1024, isolated down dequant+cuBLAS and GDN 2045+3; five measured same-session on/off requests with the down route held fixed | Close E3 native-runtime prefill and decode thresholds before correctness investment | Prefill median 3147.75 on vs 2862.55 off tok/s (+9.96%); decode median 83.67 on vs 83.11 off. On medians are 96.0%/101.4% of native E3 SGLang 3278.74/82.53. All five prefill and five 256-token short-prompt decode responses match the GDN-off control and original r7 outputs. A 2048-token prompt followed by 256 speculative tokens matches the same-stack GDN-off control exactly. Exact receipts `tuning/v047-sm120-sglang-gdn2045-k3-screen-20260924/e3-{on,off}-five-ub1024-gpu-down.json` and `e3-on-longdecode-ub1024-gpu-down.json`. | Pass short speed and bounded output screens on isolated source; exact integrated runtime and release artifact remain unqualified. |

| GDN-17 | Exact r7 runtime plus rebuilt SM120 CUDA library with the 2045+3 K3 path, matched GDN wrapper/cubins, and selected E3 down bridge; five measured stateless 8K MTP requests per model | Confirm the isolated gain on integrated installable bits | W2 prefill/decode median 2983.57/86.47 tok/s; E3 3148.47/83.76. Native SGLang same-card references W2 3076/83.30, E3 3278.74/82.53: each integrated cell is within 5%. All five prefill and five 256-token short decode outputs per model match original r7. E3 2048-prompt +256 speculative output matches its GDN-off control. Both E3 and W2 returned `BLUE-FLAMINGO-42` twice on the 4841-token stateless 8K needle. Receipts `tuning/v047-sm120-sglang-gdn2045-k3-screen-20260924/{e3,w2}-integrated-five.json` and `{e3,w2}-integrated-needle8k.json`. CUDA library SHA-256 `08285f7a4c16c42f0c27331c9183f26ff6d2504c9c5c1dd202fabdcaad035a73`; z840 build source and binary restored to original hashes and port 8080 healthy. | Stage a bounded MTP speed profile. DFlash2 depth four code/prose target, broader quality, 32K GDN behavior, and native IQ3 decode parity remain open. |
| DFL-01 | Integrated r7+GDN E3 at DFlash2 depth four, Qwen3.8 Q4_K_M draft, 8K stateless, Club 3090 narrative/code prompts, one warmup and three measured 500-token runs per prompt | Screen the final speculative path before quality work | Mean decode 62.98 narrative / 93.57 code tok/s. The draft reported 45–47% narrative and 76–83% code token acceptance. Receipt `tuning/v047-sm120-sglang-gdn2045-k3-screen-20260924/e3-dflash-n4-initial.log`. This is below the user's 150/200 goal. | Reject the stock E3 head route as a final profile; no correctness suite. |
| DFL-02 | Same DFlash2 screen, with an opt-in E3 head dispatch cap of eight rows so depth-four verification's five rows use the existing direct decode kernel | Reuse the proven SM89 M>4 head fix on SM120 | Mean decode 79.38 narrative / 114.93 code tok/s (+26%/+23% versus DFL-01), still below 150/200. Receipt `tuning/v047-sm120-sglang-gdn2045-k3-screen-20260924/e3-dflash-n4-mrow8.log`; candidate CUDA library SHA-256 `cdf4d363a0dc981393dbb3762601640bcadd250eec62c13a7df1f237828d5102`. z840 source and binary restored to original hashes, port 8080 healthy. | Keep as an isolated speed lead; screen the existing warp-four-row head path before any correctness suite or release profile. |
| DFL-03 | Same candidate with existing `ESCHA_E3_HEAD_WARP4=1`, one warmup and three measured 500-token runs per prompt | Check whether four vocabulary rows per CTA closes the verification-head cost | Mean decode 78.26 narrative / 111.73 code tok/s versus DFL-02 79.38/114.93. Receipt `tuning/v047-sm120-sglang-gdn2045-k3-screen-20260924/e3-dflash-n4-mrow8-warp4.log`. | Reject the warp4 flag on this workload. Do not run a correctness suite. The historical SM89 R=2 multi-row verification head is the next targeted speed candidate. |
| DFL-04 | Port the frozen SM89 R=2 multi-row E3 head kernel, opt in with `ESCHA_E3_HEAD_RT=2` and MROW8, same RTX 5090 DFlash2 N4 500-token code/narrative screen | Reuse the production M>1 head optimization without changing default MTP or raw M1 dispatch | Mean decode 90.22 narrative / 135.54 code tok/s, versus stock 62.98/93.57 and MROW8-only 79.38/114.93. Three measured runs per prompt after one warmup, receipt `tuning/v047-sm120-sglang-gdn2045-k3-screen-20260924/e3-dflash-n4-rt2.log`; integrated CUDA candidate SHA-256 `f1e785ebd27174b7d1d8d665d76da52b28931d3a470a5922b39c57f418a0aa89`, incremental lowgpu patch `dflash-rt2-post-r7.patch`. z840 source/binary restored to original hashes and port8080 healthy. | Positive isolated speed lead, but still short of 150/200; retain as opt-in research candidate. No broad correctness suite or DFlash release profile until a further speed gain closes the gap. |
| PKG-07 | Local r8 experimental MTP stage, opt-in `fast-8k-mtp-gdn`; exact GDN integration and E3 down bridge with pinned cubins/licenses/source patches | Give users an installable review asset while the no-spec and DFlash gates remain open | Archive `dist/escha-beellama-v047-sm120-experimental-r8.tar.zst` SHA-256 `57a36f1add85ddb37629a9d32b5482f8c63db78d8347795f2064cd8ae2a5f9f5`. Fresh extraction passed 338/338 payload hashes. E3 and W2 `doctor --profile fast-8k-mtp-gdn --report` both passed model hash, payload, launcher and control smoke. Exact runtime/bin profile from GDN-17 is staged; r7 archive remains unchanged. | Internal experimental review only. Non-spec native GGUF speed within 5%, DFlash2 N4 150/200, 32K GDN, and broad correctness remain open. No commit/tag/public release. |
| GDN-18 | Added a K=1 2045+3 GDN dispatch as an isolated follow-up candidate, then stopped at user request before GPU speed/correctness testing | Preserve a reproducible next lead without claiming parity | Candidate CUDA library SHA-256 `e7e8f9dfdeef99ccc3452a646dda6336b93c98af26994dba1d05b5c43044f263` saved under `/home/sean/work/escha-v047-gdn-k1-unqualified/`; incremental patch `tuning/v047-sm120-sglang-gdn2045-k3-screen-20260924/gdn-k1-unqualified-post-r8.patch` SHA-256 `bd58d120f1351dc89a4177d90e97490c5fec1204ce1870fe12b6873ef03e1720`. No GPU request was run with it. Local source restored to r8 GDN and r7 lowgpu hashes; z840 source and binary restored and port8080 healthy. | Unqualified research artifact only; no performance or correctness claim. |

The within-5% native GGUF gate is explicitly **non-speculative Escha versus non-speculative native GGUF** at matched GPU, cache, prompt and decode settings. GDN-17's MTP rates meet a separate native SGLang served comparison but do not close the native GGUF gate. Historical corrected raw no-spec KVarN3/2 rates in GDN-06/NAT-03 were E3 2869.16/79.67, W2 2790.75/75.62, native IQ3 GGUF 3104.70/90.50 prefill/decode tok/s; all four Escha cells were more than 5% below native IQ3 at that point. GDN-20/RES-01 supersede these as the current discovery screens. The final speculative goal is DFlash2 depth four at >=2x matched served non-speculative native GGUF decode; 150 narrative and 200 code are historical directional targets.

### PLAN-01 — next-sprint review, 2026-09-24

Documentation review only; no new GPU measurements or qualification. Prioritize
the already-built GDN-18 prefill probe, then profile and remove shared raw M=1
decode cost, then address model-specific residuals. GDN-12 already provides an
isolated positive K=1 prefill signal, but GDN-18 still needs its own validity
and timing evidence. A shared GDN improvement must also be offered to the
native control wherever applicable; otherwise parity could be overstated.

Clarification of DFL-04: retain RT2 as the strongest E3 verification-head lead.
After paired speed confirmation and targeted candidate correctness, promote it
to the DFlash champion even if Gate B remains open. Do not require a single
optimization to close the entire release gap. Broad release qualification
still waits for the speed gates. Raw M=1 and MTP champion choices require their
own evidence; RT2's M=2..5 gain does not establish either.

### GDN-19 — integrated K=1 2045+3 validity screen, 2026-09-24

Hypothesis: GDN-18's saved K=1 dispatch can reproduce GDN-12's isolated
prefill gain. Exact GDN-18 CUDA library SHA-256
`e7e8f9dfdeef99ccc3452a646dda6336b93c98af26994dba1d05b5c43044f263`
was loaded through an isolated r8 runtime overlay with the r8 matched GDN
bridge/cubins. E3, W2 and native IQ3 each processed the same 2048-token
diagnostic batch with bridge on/off; stage traces confirmed actual execution.
All 248,320 final logits were finite and top choice matched, but full-logit
relative RMS was 0.29933 E3, 0.24599 W2, 0.572 native; top-20 overlap was
12/14/6 respectively. Exact logits and logs:
`tuning/v047-sm120-sprint-20260924/gdn18/{e3,w2,native}-{control,candidate}.*`.
Decision: **reject GDN-18 for timing or promotion**. The reused 2045+3 MTP
tail is a possible cause, not an established diagnosis. GDN-12's positive
screen used a full 2048-token K=1 bridge. A separate K=1 full-batch dispatch
is the one bounded follow-up; the K=3 path remains intact.

### PRF-01 — raw W2/native decode API comparison, 2026-09-24

One matched n256 KVarN3/2 llama-bench run per model under Nsight Systems,
exact r8 control runtime. W2 74.35 tok/s, native IQ3 89.07 tok/s under
profiler; these are diagnostic, not speed-gate samples. Nsight captured CUDA
API timing but no GPU kernel timing on this WSL host. W2 showed 11,129
`cudaStreamSynchronize` and 6,105 `cudaMemcpyAsync` calls versus native
7,797/4,062. Counts include benchmark warmup and are not per-token causal
overhead without stack/route attribution. Reports under
`tuning/v047-sm120-sprint-20260924/gdn18/{w2,native}-raw-decode-profile.*`.
Decision: retain the question of extra W2 synchronization/copy calls; no
decode patch is justified from this trace alone.

### GDN-20 — full 2048-token K=1 bridge, 2026-09-24

Hypothesis: GDN-18's 2045+3 tail caused its numerical error; a dedicated
full-batch K=1 route should recover GDN-12's valid prefill gain. New isolated
source worktree `/home/sean/work/escha-sm120-sprint-k1full` replays the pinned
Preview rebase patch and r8 GDN patch, then adds
`tuning/v047-sm120-sprint-20260924/gdn18/gdn-k1-full2048-post-r8.patch`
(SHA-256 `c281dfe9f8d1a874b8dd3666955cb79f172f0dc47237bdeeb86419f70f048c0c`).
The K=1 path loads a distinct full-2048 source-matched bridge and does not
change the K=3 2045+3 rollback route. The bridge is built from the previously
screened GDN-12 source with CUDA 13.0; local library SHA-256
`10e0133861e469f27a55e31dea466689e4ac971efeda054b1fc9d98fdee7d69d`.

For the cheap screen, a temporary `LD_PRELOAD` library linked just the new
`gated_delta_net.cu` object against frozen r8. The r8 exported entry point is
called via PLT; GDN stage traces confirm dispatch. This overlay is a diagnostic
artifact, **not** the final integrated CUDA library. Exact E3/W2 final logits
were finite, top choice matched, top-20 overlap was 18/19, and relative RMS
0.015982/0.016470 versus control, reproducing GDN-12. Both models' 16-token
greedy continuation matched control exactly under both F16 and the release
KVarN3/2 cache (separate `continuation-kvarn` receipts). Native IQ3 failed the same
numerical guard: top choice changed from token 13477 to 13, top-20 overlap 6,
relative RMS 0.647631. Therefore the candidate is currently applicable only
to Escha E3/W2; native uses the correct unoptimized route. Do not enable this
K=1 flag for native IQ3.

One warmup plus three measured KVarN3/2 p2048 repeats per process produced
initial medians E3 3218.98, W2 3201.59, native-off 3214.94 tok/s; E3/W2
control medians 2864.08/2893.78. A second fresh-process screen and bracket
gave E3 candidate 3159.72, W2 3191.38, native-off 3260.54 before and
3004.66 after. The native control drift means these numbers are a strong
prefill lead, **not** a clean Gate A declaration. Raw decode is unchanged by
design and remains below threshold. Receipts (logits, stage traces, 16-token
continuations, sample arrays, source/bridge) are under
`tuning/v047-sm120-sprint-20260924/gdn18/`.

The clean isolated z840 build completed. Its linked `libggml-cuda.so.0.23.0`
SHA-256 is `9d13df6fbf063834ced63a16e6074074654f026c222716fcf188d2f8d71843aa`.
Using that library, the E3/W2 K=1 bridge dispatched and passed the same finite
logit/top-token check (top-20 overlap 18/19; relative RMS 0.015982/0.016470).
Sixteen-token greedy continuations matched control under both F16 and KVarN3/2.
With the bridge disabled, native IQ3 logits were byte-identical to r8. Native
with the full K=1 bridge enabled failed the logit guard, so the matched native
control correctly uses the disabled route.

Clean linked one-warmup/three-measured fresh-process p2048 KVarN3/2 screen:
native-off median 3212.94 tok/s (nine measured samples, max 3271.00), E3
linked 3179.15 (six samples), W2 linked 3213.68 (six samples). E3 and W2
respectively reached 98.95% and 100.02% of the native median; even their
slowest linked samples exceeded 95% of the fastest native sample in the bracket.
Relative to same-library bridge-off controls, gains were 12.57% E3 and 11.46%
W2. Receipt: `tuning/v047-sm120-sprint-20260924/gdn18/linked-prefill-summary.json`.
This passes the raw p2048 prefill microbenchmark gate, not a populated-context
served release gate. The linked library still has a build-path RUNPATH and must
be relinked for a portable artifact.

Same-library raw p0/n256 KVarN3/2 screen: native-off 90.6817 tok/s, E3 linked
80.2438, W2 linked 76.3096. The 95% decode threshold is 86.1476; E3 needs
7.4% and W2 needs 12.9% throughput gain. The K=1 prefill bridge does not
dispatch at single-token decode, as intended. Decision: retain GDN-20 as the
E3/W2 raw-prefill champion and continue the decode critical path. z840's
running port-8080 server and original build were not modified.

### PRF-02 — decode graph-split attribution, 2026-09-24

Hypothesis: Escha raw decode loses significant time in extra backend graph
segments and host-to-device input copies. A temporary `LD_PRELOAD` callsite
interposer in `tuning/v047-sm120-sprint-20260924/gdn18/api_calls.cpp` traced
the linked CUDA library, then `GGML_SCHED_DEBUG=2` identified split inputs.
Matched W2 and native p0/n32 KVarN3/2 traces showed seven CUDA graph splits
per Escha token versus two total native splits (one CPU embedding, one CUDA).
W2 made 330 `cudaMemcpyAsync` tensor uploads versus 231 native, and 221 CUDA
graph launches versus 31 native over the 33 decode calls including warmup.
E3 also had seven splits. The extra Escha split inputs include `escha_lut`
(128 KiB), `escha_dep_k2/k3` (8 KiB each), and cache/attention input views.
These are structural counts; profiler-instrumented speed is diagnostic only.
Next hypothesis: make the shared tables GPU resident to remove repeated
transfers and some graph boundaries.

### SCH-01 — remove scheduler input-capacity split, 2026-09-24

Hypothesis: a scheduler split triggered at input-list capacity causes the
seven Escha segments. An isolated `ggml-base` rebuild removed only that
capacity condition while retaining the separate cross-backend weight rule.
W2 still had seven splits, 90 tensor uploads and 53 CUDA graph launches over
nine p0/n8 decode calls, identical to control. Decision: reject; no speed
screen or correctness qualification. The source change was reverted. The
weight-placement path is the better next experiment.

### RES-01 — GPU resident shared Escha decode tables, 2026-09-24

Hypothesis: the 128 KiB `escha_lut` and two 8 KiB dependency tables are loaded
as input weights on CPU even when all 64 model layers run on CUDA. Placing only
these three immutable tables under the output GPU buffer policy should remove
repeated host-to-device copies and graph boundaries. A nine-line loader change
in the isolated Preview worktree does this only for `LLM_ARCH_ESCHA` and those
three tensor IDs. Incremental patch:
`tuning/v047-sm120-sprint-20260924/gdn18/escha-tables-gpu-resident-post-k1full.patch`
SHA-256 `04a53a58b21cb258bd94e6aabe49fcd9f6fcf1805257b2d397973fda91b10dad`.
The rebuilt `libllama.so.0.4.7` SHA-256 is
`87994d82caab38bdc1f4769c99fc553c9570c6cf0ce513b1a59d67d2886ce973`;
CUDA library and bridges are unchanged from GDN-20. Local source and z840
build source hashes match.

W2 p0/n8 callsite trace: graph splits fell 7→1 per decode call; across nine
calls, graph launches fell 53→7 and tensor uploads 90→63. E3/W2 16-token
greedy continuations under KVarN3/2 matched the prior control exactly and
all logits remained finite. One warmup plus three measured p0/n256 KVarN3/2
samples per process gave first medians native 91.4569, E3 81.7095, W2
77.4795 tok/s. A paired confirmation versus the unchanged linked-off runtime
gave E3 79.1432→81.5741 (+3.07%) and W2 75.2718→76.9280 (+2.20%);
the E3 control varied, so treat its percent gain as provisional. Same library
p2048 prefill medians: native 3160.36, E3 3146.30, W2 3186.76 tok/s, all
above the 95% Gate A prefill threshold in this bracket. Receipts are under
`tuning/v047-sm120-sprint-20260924/gdn18/` with `resident` tags.

Decision: retain the small, source-isolated cumulative improvement as the
local champion; it does **not** close raw decode parity. The matched native
decode 95% threshold is 86.8841 tok/s on this control. E3 and W2 remain
roughly 6.5% and 12.9% short of that threshold in candidate throughput.
Do not run full release qualification yet. The `resident` overlay is local and
still needs a portable `$ORIGIN` relink before any release artifact.

### BND-01 — raw decode bridge contribution, 2026-09-24

Hypothesis: the active direct-F32 raw decode bridge might be adding overhead
on W2. On the RES-01 champion, only `ESCHA_OFFICIAL_RAW_DECODE_BRIDGE` was
disabled after profile load. The W2 16-token KVarN3/2 greedy continuation
remained identical and finite. One warmup plus three measured p0/n256 samples
gave median **56.6452** tok/s off versus approximately **77** tok/s with the
bridge enabled in the neighboring RES-01 screens. The loss is far beyond run
noise; decision: **reject bridge-off** without further qualification. The
existing direct-F32 bridge is material to W2 decode speed. Receipt:
`tuning/v047-sm120-sprint-20260924/gdn18/w2-resident-full-decode-bridge-off-n256.json`.

### PRF-03 — same-method output-head phase timing, 2026-09-24

To decide whether the W2 output head can account for the remaining raw decode
gap, a diagnostic `ggml_backend_sched_set_eval_callback` interposer synchronized
after `result_norm` and `result_output`. Each `result_output` graph segment
contained one CUDA `MUL_MAT` node; the earlier segment ended at `result_norm`.
The same callback method on the RTX 5090, KVarN3/2 p0/n8, gave median
output-head wall time over nine calls: W2 **1.534 ms**, E3 **0.622 ms**, native
IQ3 **0.471 ms**. The callback changes graph segmentation and adds sync, so
these are *diagnostic phase timings*, not Gate A throughput numbers. The W2
head exceeds the entire 1.49 ms/token latency reduction needed to reach the
current 95% threshold; its difference from native is about 1.06 ms. E3's
head difference is only about 0.15 ms against a ~0.75 ms total gap. Receipts:
`tuning/v047-sm120-sprint-20260924/gdn18/{w2,e3,native}-head-probe-n8.log`.

Sean clarified that E3 and W2 model weights must remain untouched, with no
requantization. A proposed W2 Q8_0 in-memory head repack was **canceled before
any GPU test** because it violates that constraint. Its isolated build was
interrupted and the candidate is discarded. Subsequent work must optimize the
existing weight representation and executed kernels/dispatch. The historical
SM120 exact-I8 head's 7.56 tok/s result remains rejected.

### SPL-01 — bounded raw decode split selectors, 2026-09-24

Hypothesis: the existing SM89-derived K2/K3 split selectors might improve the
SM120 direct-F32 coded GEMV without changing E3/W2 weights or bridge kernels.
This screen used the RES-01/GDN-20 W2 champion, RTX 5090, KVarN3/2 p0/n256,
one warmup and three measured `llama-bench` samples. The unchanged control
median was **77.0486 tok/s**. The K2 selector
`ESCHA_OFFICIAL_RAW_DECODE_K2_SHAPE_SPLITS=1` produced **76.4651 tok/s**
(−0.76%); its finite 16-token greedy continuation exactly matched the
champion. Reject the K2 selector for this configuration.

The profile sets `ESCHA_OFFICIAL_RAW_DECODE_K3_SPLIT_DIV2=0`, so an initial
environment assignment before the launcher was overridden and measured only
the control path. Applying the override *after* profile loading produced
**76.7569 tok/s** (−0.38% against the same control) with an identical finite
16-token continuation. Reject/defer the K3 selector: no repeatable positive
signal that justifies a larger screen. The ineffective first run is retained
as a receipt, not counted as candidate evidence. Receipts:
`tuning/v047-sm120-sprint-20260924/gdn18/w2-resident-full-n256-r4-k2split-{control,all}.json`,
`w2-resident-full-n256-k3split-half-real.json`, and
`w2-k3split-half-real-continuation.{txt,log}` in the same directory. Both GGUF
SHA-256 hashes remained the pinned E3/W2 values; no model or runtime binary
was edited. The next decode attempt needs a measured bottleneck or a different
mechanism rather than more unguided split sweeps.

### OVL-01 — existing gate/up overlap selector, 2026-09-24

The old SM89 isolated K2-gate/K3-up concurrent pair suggested a possible
body-speed gain. On the RES-01/GDN-20 W2 champion, enabling the existing
`ESCHA_OFFICIAL_RAW_DECODE_GATE_UP_OVERLAP=1` *after* profile load failed the
cheap KVarN3/2 continuation guard at `escha-moe.cu:2958`: `gate/up overlap
expected one pending K2 gate`. Exit 134; no throughput was measured. The
selector assumes strict one-gate/one-up adjacency that the current graph does
not provide. Reject the flag-only experiment. Revisit only with an explicit
graph-order diagnosis and a safe multi-pending design whose projected benefit
justifies the code and qualification cost. Receipt:
`tuning/v047-sm120-sprint-20260924/gdn18/w2-gate-up-overlap-continuation.log`.
The existing champion and GGUF weights remain unchanged.

### GDN-21 — fused decode GDN column-warps screen, 2026-09-24

Hypothesis: Escha's single-token fused GDN kernel redundantly computes the
gate/beta preparation once per column CTA. A same-method scheduler callback
isolated the first `GGML_OP_GATED_DELTA_NET` node on KVarN3/2 p0/n8. The
diagnostic median over nine calls was W2 **0.042420 ms**, E3 **0.047640 ms**,
native IQ3 **0.028494 ms**. This method synchronizes at the node boundary and
is for attribution, not a Gate A throughput measurement. Receipts:
`tuning/v047-sm120-sprint-20260924/gdn18/{w2,e3,native}-gdn-probe-n8.log`.

The isolated `gdn8-fused8-candidate.patch` changes only the fused decode
specialization from four to eight column warps per CTA; native and all other
GDN paths retain four. It changes no model tensors. Candidate source:
`/home/sean/work/escha-sm120-sprint-gdn8/`; patch SHA-256
`dc2d4b2cec01a336432f38973011f1241f8c9d3b4dafb7bb38760e5733a19cca`.
The z840 diagnostic build reused the champion's compiled objects and rebuilt
only `gated_delta_net.cu` before relinking. Candidate CUDA library SHA-256
`a9a76d31a1906366a83b69399da5c9858f8643d12817b63d3c236ca1ab532924`;
the champion CUDA library remains
`9d13df6fbf063834ced63a16e6074074654f026c222716fcf188d2f8d71843aa`.
The local overlay's load path was confirmed by `LD_DEBUG=libs`. A 16-token W2
KVarN3/2 greedy continuation was finite and identical to the champion.

The diagnostic GDN median fell from **0.042420** to **0.039615 ms** at the
first node, but whole-model W2 p0/n256 speed did not improve repeatably.
One-warmup/three-measured medians in the first control/candidate/control
bracket were **77.5809 / 78.0543 / 77.1890 tok/s**. An order-alternated
repeat measured candidate **77.0546** versus control **78.4114 tok/s**;
individual samples overlapped substantially. Decision: **reject**, do not
promote or run full correctness/other-model qualification. The effect on one
GDN node is too small to close the decode gap. Receipts are the `gdn8` logs,
JSON files and patch under the same `gdn18/` directory. The RTX 5090 returned
to 16 MiB idle; the champion, GGUF weights and z840 port-8080 service were not
changed.

### GDN-22 — per-head preparation prepass, 2026-09-24

Hypothesis: compute Escha's fused GDN gate/beta transcendentals once per head,
then reuse those results across all column CTAs. An isolated candidate adds
one CUDA preparation kernel and a small pooled output buffer within the GDN
op; the recurrent kernel reads the prepared values. No GGUF weight or model
tensor is changed. Source patch:
`tuning/v047-sm120-sprint-20260924/gdn18/gdn-prep-per-head-candidate.patch`,
SHA-256 `5f357fccdab5ffb3923cfd6eb3d9a14897341511ed6c7a28addbe50fdb0b10da`.
Candidate CUDA library SHA-256
`e4255b4c2c366e4c2cb3722d0ea3a777cea69fd8b940db179036414e6a878b16`;
the champion library was not replaced. The diagnostic overlay was confirmed
loaded, and the W2 KVarN3/2 16-token greedy continuation was finite and
identical to the champion.

The same first-GDN-node p0/n8 probe **regressed** from **0.042420** to
**0.063270 ms** median over nine calls. The extra launch/buffer path costs more
than it saves at this shape. Decision: **reject before whole-model speed or
broader correctness testing**, and do not add a per-head prepass to the
champion. Receipt: `gdn18/w2-gdn-prep-probe-n8.log` and the candidate patch.

### PRF-04 — first-layer FFN diagnostic comparison, 2026-09-24

The same scheduler callback timed isolated `ffn_gate-0`, `ffn_up-0` and
`ffn_out-0` nodes under native IQ3 KVarN3/2 p0/n8. Native medians were
**0.053602 / 0.076796 / 0.066967 ms**, compared with earlier W2 medians
**0.065555 / 0.071325 / 0.069021 ms**. Each target was one CUDA graph node.
These are separate short diagnostic sessions, with graph segmentation and
sync, so neither the sum nor the small differences establish a whole-model
attribution. They do rule out an obviously dominant first-layer FFN node;
next FFN work needs a shape-wide or direct-kernel measurement. Receipts:
`gdn18/native-node-ffn_{gate,up,out}-0-n8.log` (W2 gate receipt uses its
`w2-node-ffn-gate0-n8.log` spelling).

### PRF-05 — direct-F32 bridge shape trace and timing limit, 2026-09-24

A diagnostic `cuLaunchKernel` interposer recorded the executed direct-F32
bridge shapes on RTX 5090 KVarN3/2 p0 decode. Across three W2 decode calls,
the bridge launched 64 each of the K2 gate (`5120→17408`, 8 splits), K3 up
(`5120→17408`, 8 splits), and K3 down (`17408→5120`, **12 splits**) per call.
The K3 down kernel is used by the fused down/add/RMS path at
`escha-moe.cu:4014`, which hardcoded 12 splits. The older environment selector
in the ordinary projection function did **not** control this executed path.
Receipts: `gdn18/bridge_events.cpp` and `w2-bridge-events-n2.log`.

CUDA events under `GGML_CUDA_DISABLE_GRAPHS=1` are unsuitable for a release
critical-path attribution: graph-off W2 p0/n8 was **38.83 tok/s** without the
interposer versus approximately 25–26 tok/s with event instrumentation, while
the graph-on champion is around 77 tok/s. Warm-call per-shape event times also
varied substantially. Do not use their summed GPU time as a Gate A latency
breakdown. Their launch counts and shape/split identification were used to
choose one targeted SM120 experiment.

### DOWN-01 — SM120 W2 fused K3 down projection, 2026-09-24

Hypothesis: the SM89 12-split fused K3 down projection underfills the RTX
5090. CUDA reports **170 SMs** on RTX 5090 versus **128** on z840 RTX 4090.
For `OC=5120`, 12 splits launch 480 CTAs, or 2.82 per 5090 SM; 16 splits
launch 640 CTAs, or 3.76 per SM, matching the SM89 launch density. The
power-of-two count also avoids the non-power-of-two division branch visible
in the retargeted PTX. These are hypotheses; the end-to-end screen decides.

An isolated source copy `/home/sean/work/escha-sm120-sprint-down16` changes
the **executed fused down/add/RMS** path to use 16 splits only when
`ESCHA_OFFICIAL_RAW_DECODE_K3_DOWN_SPLITS=16` is set. The default remains 12,
and the ordinary projection path remains 12. Patch:
`tuning/v047-sm120-sprint-20260924/gdn18/w2-down16-post-res01.patch`
SHA-256 `1891c7aec22da05060f753721d9123e07a03e98463213c0ac03753e5b0a1ea9d`.
The isolated CUDA library SHA-256 is
`c737d259d9cce61e798dd54fb2b09fd2e23b1ca5fcb5d18df5822fb1d8426a4d`.
It is a local build-path-RUNPATH overlay, not a portable release library.
`run.sh down16 w2` selects it and applies the 16-split flag after loading the
r8 base profile; E3 remains on `resident-full`. The frozen r8 artifact and
RES-01/GDN-20 champion binaries were not replaced.

The initial patch to the ordinary projection selector still dispatched 12
splits and was **not timed**. The corrected candidate's route trace showed
**128 K3 down launches at 16 splits over two decode calls**. `LD_DEBUG=libs`
confirmed the candidate CUDA library mapped. W2 and E3 16-token KVarN3/2
greedy continuations after a 2048-token prompt were finite and exactly matched
their prior champion outputs.

On graph-on W2 p0/n256, one warmup plus three measured samples per process,
the first control/candidate/control medians were **77.2564 / 78.4415 /
77.2509 tok/s**. An order-alternated candidate/control repeat gave **78.4315 /
77.2401 tok/s**. The roughly **+1.5%** W2 gain repeated. The same E3 bracket
gave control/candidate/control **83.5263 / 83.1935 / 83.5495 tok/s**; E3 has
no gain, so retain its prior champion. W2 candidate p2048 prefill median was
**3218.65 tok/s**, bracketed by correct native IQ3 controls **3241.46** and
**3161.33 tok/s**. Even the slowest W2 candidate measured sample exceeded 95%
of the fastest native measured sample. This confirms only the raw p2048
microbenchmark prefill cell, not served 8K qualification.

Contemporary native IQ3 graph-on p0/n256 median was **90.8622 tok/s**, so its
95% decode threshold is **86.3191 tok/s**. W2's 78.44 remains about **10.0%**
short in throughput; E3's unchanged roughly 83.54 remains about **3.3%** short
against this reference. **Decision: retain DOWN-01 as the W2-only local raw
decode champion**, below the release threshold. This SM120 raw M=1 change has
not been shown to speed DFlash2's multi-row verification. Full correctness,
served speed, installation and release artifact qualification remain open.
Receipts: `gdn18/{w2,e3}-down16-*`, `native-resident-*-down16-*`, the source
patch and build log on z840. No E3/W2 model weights were changed.

### GDN-WARP — reject warp-local fused decode preparation

Hypothesis: in the fused K=1 GDN kernel, compute gate and beta once per warp
and broadcast with `__shfl_sync`, eliminating the block-wide shared-memory
barrier. This keeps the same math and adds no kernel launch. The isolated
source `/home/sean/work/escha-sm120-sprint-gdn-warp` builds on DOWN-01;
`gdn18/gdn-warp-post-down16.patch` SHA-256
`d8d103e6b7120911f8b4f957006f49e49d68d6cdac7eba191e14ee547515c18e`.
The CUDA overlay SHA-256 is
`24f55b65b879f6ea8570ff1e34fabc1d3f74478ec7410739afd2468249b76c12`.
The `gdn-warp` tuning wrapper uses W2's 16-split down path and E3's ordinary
12-split path. `LD_DEBUG=libs` confirmed the candidate was loaded. Both models
completed the 2048-prompt plus 16-token KVarN3/2 greedy continuation with
finite logits and exact token agreement against their respective champions.

Graph-on p0/n256, one warmup plus three measured samples, yielded W2 DOWN-01
control **77.5853** vs candidate **77.1649 tok/s** (−0.54%). E3 candidate
**82.3800** vs same-session RES-01 control **83.5930 tok/s** (−1.45%). The W2
result lacks a positive signal, and E3 is clearly slower. **Reject:** retain
DOWN-01 for W2 and RES-01 for E3; do not run prefill or release correctness on
this regression. Receipts: `gdn18/*-gdn-warp-*`. The frozen r8 artifact and
model GGUF weights are unchanged.

### HEAD-01 — current W2 direct INT8 head screen

The earlier direct INT8 W2 head was extremely slow on SM120, but the current
`lowgpu.cu` has a four-row-per-CTA FP32-accumulator implementation. A single
graph-on p0/n8 KVarN3/2 raw decode screen on the W2 DOWN-01 champion, with
`ESCHA_W2_I8_HEAD=1` set **after** profile loading, measured **7.65025
tok/s**. This is the same severe regression class as the older 7.56 tok/s
result. **Reject the existing direct INT8 head route** and stop before a
correctness suite or long throughput screen. Receipt:
`gdn18/w2-direct-i8-current-n8.{json,log}`. This read the original W2 head
format at load time; the GGUF weights and champion artifacts were unchanged.

### PRF-06 — sparse bridge event probe remains too disruptive

To test whether PRF-05's CUDA-event shape timing could be made usable, the
existing bridge interposer sampled only every 64th launch on the W2 DOWN-01
champion. Matched graph-off KVarN3/2 p0/n16 single-run throughput was **41.3799
tok/s** uninstrumented and **23.5499 tok/s** with sparse interposition. Even
sparse sampling perturbs the run by about **43%**. Retain only executed shape
and launch-count evidence from this tool; do not use its event sums as decode
critical-path attribution or pursue more sampling sweeps. Receipts:
`gdn18/w2-prf06-{graphoff-control,sparse64}-n16.{json,log}`.

### HEAD-02 — isolate W2 direct-INT8 placement defect, short-screen hold

GBrain's prior SM89 work identified a loader probe defect behind the old
~8 tok/s INT8-head result: the full W2 output matrix was being copied from
host to GPU each token. Current v0.4.7 loader source had the same structural
gap: `create_tensor` selected generic `MUL_MAT` for W2 `output.weight` and
`output.weight_scale`, while its `LOWGPU_MUL_MAT` buffer probe knew only E3's
packed sidecars. An isolated loader-only patch in
`/home/sean/work/escha-sm120-sprint-i8res` selects the real W2 lowgpu op
and probes the original row-scaled I8 code/F16 scale pair. It is gated by
`ESCHA_W2_I8_HEAD=1`, preserving the default dense-head route. Patch
`gdn18/w2-i8res-post-down16.patch` SHA-256
`cf2247270e9d0a9f8d8ca4ea13d96265cff53d4688d3410b9c02aa8a2d55ffca`;
isolated libllama SHA-256
`4c26a2a4648916870921cb3a2a1d6dea719ece1432edda371f9d3868cca43ef6`.

With the W2 DOWN-01 CUDA champion still selected, the isolated opt-in p0/n8
screen rose from HEAD-01's **7.65025** to **56.8587 tok/s**, and `LD_DEBUG`
confirmed the candidate libllama loaded. The same-session dense-head control
was **67.414 tok/s** on the same short method: the repaired direct-I8 route is
still **15.66% slower**. This initial short-screen hold was **superseded by
HEAD-03** after copy counting, a longer paired speed screen, and a targeted
logit guard. The recovery is consistent with removing
the historical placement defect, but this run did not directly count H2D
bytes. The frozen r8 artifact, W2 champion and E3/W2 GGUF weights are unchanged.
Receipts: `gdn18/w2-i8res-{n8,control-n8}.{json,log}` and isolated patch/lib.

### PRF-07 / HEAD-03 — original-INT8 W2 raw local champion

The HEAD-02 p0/n8 result was too short to decide the route. A focused
`copy_bytes.cpp` interposer counted transfers through `cudaMemcpyAsync` and
`cuMemcpyHtoDAsync_v2` during `llama_decode`. Both repaired I8 and dense
controls had **63 runtime H2D calls totaling 5,004 bytes**, largest 512 bytes,
on p0/n8; neither issued a large transfer through those APIs. Instrumented
short speeds were **60.6258 / 61.6339 tok/s** (I8/dense). This is copy-count
evidence, not comprehensive transfer coverage or GPU kernel timing. Receipts:
`gdn18/w2-{i8res,dense}-copy-n8.{json,log}` and `gdn18/copy_bytes.cpp`.

Graph-on p0/n256, one warmup plus three measured samples per fresh process,
gave first dense/I8/dense medians **77.9061 / 82.3808 / 78.8447 tok/s**.
An order-alternated I8/dense repeat gave **83.5961 / 78.9640 tok/s**. The
roughly **+5–6%** W2 gain repeats beyond ordinary control drift. Matched
native IQ3 median **90.7952 tok/s** sets a current 95% threshold of
**86.2554 tok/s**; W2 I8 remains **3.18% short** in throughput. This is a
local champion, not a Gate A decode pass.

The candidate completed p2048 prefill: this prompt graph supplied a one-token
head output despite the lowgpu source's multi-token guard. Candidate median
**3212.32 tok/s** versus dense **3210.88** and native IQ3 brackets
**3138.05 / 3232.43**. All three measured candidate samples exceeded 95%
of the fastest measured native sample. Its p2048+16 greedy continuation was
finite and token-identical to dense. At steps 0 and 15, full 248,320-vocabulary
logit snapshots had identical argmax and 20/20 top-20 overlap; relative RMS
deltas were **0.0002305 / 0.0002356**, max absolute deltas **0.00785 /
0.00657**. This is numerical, not bitwise, equivalence; broader quality
qualification remains open.

**Keep HEAD-03 as the W2 raw M=1 local champion** on top of DOWN-01 and
RES-01. `run.sh i8res w2` loads the isolated libllama and DOWN-01 CUDA lib,
setting original-I8 head and 16-split down flags after the profile. `LD_DEBUG`
confirmed both libraries mapped; the wrapper's short continuation matched
the manual candidate. E3 remains on RES-01. The W2 I8 head's `n_tokens != 1`
guard means DFlash2 multi-row verification and other served shapes remain
unqualified. Frozen r8, dense fallback and model GGUF weights are unchanged.
Receipts: `gdn18/*-i8res-*`, the two `*-logits-snapshot.bin` files,
`gdn18/logits-snapshot.cpp`, and HEAD-02's patch/artifact hashes.

### BRG-01 — SM89 K2 BFE resource port rejected on SM120

Hypothesis: transfer the SM89 R251 K2 bitfield-extract rewrite/resource
schedule to the SM120 direct-F32 decode cubin, reducing K2 instructions
and closing a portion of the remaining raw decode gap without changing model
weights. The pinned SM120 PTX contains the same 56 K2 shift-plus-mask pairs
that the retained SM89 `make_bfe_ptx.py` rewrites. The isolated artifacts live
in `gdn18/r251-k2min5/`; frozen r8 cubin is unchanged.

Static guards: bare `.minnctapersm 5` reduced K2 from 64 to 48 registers but
introduced a 16-byte stack frame and 16-byte spill loads/stores; no GPU timing
was spent on that weak candidate. The full R251-style BFE + min5 variant still
spilled. BFE at the existing min4 bound compiled at 64 K2 registers with
**zero spills** and left K3's 64-register source branch unchanged. Its PTX
patch SHA-256 is `8445b003e5063d0fe18f4648c7575526f8f7f6968252ec311d295d1477fa90df`;
candidate cubin SHA-256 is
`52f9695b5a899312581104b239d2a343111a38ae975c1ff9ceaeadb10a4ba9fc`.
`cuModuleLoad` tracing confirmed that exact cubin loaded. On W2 HEAD-03,
the p2048+16 checked full-vocabulary logit snapshots and continuation were
**byte-identical** to control.

Graph-on W2 p0/n256 one warmup plus three measured samples yielded
control/candidate/control medians **82.1020 / 80.6851 / 82.4959 tok/s**,
roughly a **2% regression**. **Reject** the K2 BFE cubin; retain HEAD-03 W2
and RES-01 E3. Per the lean stop rule, no E3 speed screen, broad correctness,
or release profiling was run on this loser. Receipts:
`gdn18/{w2-k2bfe-*,w2-i8res-n256-r4-k2bfe-*}` and the ptxas/resource logs.

### GDN-23 — shared Q/K norm in fused single-token GDN rejected

Hypothesis: four column warps in each fused GDN CTA recompute the same
head-wide Q/K L2 norms. Compute them in warp 1 while warp 0 prepares gate and
beta, then broadcast both factors through the existing block barrier. This
adds no launch, changes no model weights, and leaves other GDN shapes on their
existing path. The isolated source is
`/home/sean/work/escha-sm120-sprint-gdn-norm-once`; its incremental patch is
`gdn18/gdn-norm-once-post-down16.patch` SHA-256
`cb9986c4e795edf6790414a14d0690b75d71c7117c844c31c49cdba165e75b1c`.
The CUDA library SHA-256 is
`e25213ef26c59adbb3611a23f9f080f91c0a461a9f36ae868e47b2a4cfb60306`.
It was built from the DOWN-01 CUDA base with the pinned CUDA 13.0 SM120 build;
the remote source and DOWN-01 library were restored to their original hashes
after extracting the candidate. `LD_DEBUG=libs` confirmed that W2 used this
library and the HEAD-03 I8 `libllama` overlay.

The W2 KVarN3/2 p2048+16 greedy continuation matched HEAD-03 exactly. All
checked logits were finite, and full-vocabulary snapshots at steps 0 and 15
were byte-identical to HEAD-03. Graph-on W2 p0/n256, one warmup plus three
measured samples per process, yielded control/candidate/control medians
**82.4204 / 81.8604 / 82.1011 tok/s**. Candidate samples were
83.7743/81.8604/80.7095 tok/s: no repeatable speed signal and still below
the approximately 86.3 tok/s raw gate. **Reject** GDN-23; retain HEAD-03 W2
and RES-01 E3. No E3 speed screen or broader correctness was spent on this
losing route. Receipts in `gdn18/`: `w2-gdn-norm-once-*`,
`w2-i8res-n256-r4-normonce-*`, and `gdn-norm-once-post-down16.patch`.

### HEAD-04 — eight-row original-I8 W2 head neutral

Hypothesis: HEAD-03's W2 head launches one CTA for four independent vocab
rows; eight rows per 256-thread CTA halves the block count and might save a
meaningful portion of the remaining approximately 0.37 ms/token raw gap. The
original W2 I8 weights and F16 roundtrip math remain unchanged. An opt-in
`ESCHA_W2_I8_HEAD_WARP8=1` selects the eight-row launch in isolated source
`/home/sean/work/escha-sm120-sprint-head8`, layered on DOWN-01's CUDA base
and HEAD-03's loader overlay. Incremental patch
`gdn18/head8-post-down16.patch` SHA-256
`88b875c3d232f4d2de175dfcc9d7be6081511197c8ccb0887c260d1b4afdb81f`;
CUDA library SHA-256
`5b31e88f12204267ca77f492a189a19a22c53c16dedd47674efff1ba5ef62c6a`.
The source, object and library in the z840 DOWN-01 build were restored; the
final library SHA-256 again matched its original `c737d259...` value.

`LD_DEBUG=libs` confirmed the candidate CUDA library and HEAD-03 `libllama`
loaded. W2 KVarN3/2 p2048+16 greedy continuation and checked full-vocabulary
logits at steps 0 and 15 were byte-identical to HEAD-03. Graph-on p0/n256,
one warmup plus three measured samples per process, gave bracketing
control/candidate/control medians **83.1393 / 83.1117 / 82.6774 tok/s**.
The launch change is neutral within run drift and far short of the raw gate.
**Reject** HEAD-04 and retain HEAD-03. No E3 test applies, and no broad
correctness work was spent. Receipts in `gdn18/`: `w2-head8-*`,
`w2-i8res-n256-r4-head8-*`, and `head8-post-down16.patch`.

### HEAD-05 — preconvert W2 head activation once rejected

Hypothesis: HEAD-03 re-rounds each F32 input activation through F16 for every
original-I8 vocabulary row. Convert the approximately 5K-element activation
vector to F16 once, then reuse it in the same four-row head kernel; preserve
the original I8 weights and the exact F16 arithmetic boundary. The isolated
opt-in `ESCHA_W2_I8_HEAD_PRECONVERT_X=1` source is
`/home/sean/work/escha-sm120-sprint-head-xhalf`, patch
`gdn18/head-xhalf-post-down16.patch` SHA-256
`f92ddec14aae52281836d48a66e1761526ecc41cdce9492bd80b41520c95be62`,
CUDA library SHA-256
`ea68f530a2ace1b3adee86bdd34a2a1b69cde8f02d9ccc63e036bd0a5bc34342`.
The candidate library and HEAD-03 loader mapped as intended; W2 KVarN3/2
p2048+16 continuation and checked logits were byte-identical to HEAD-03.

Graph-on W2 p0/n256 one warmup plus three measured samples yielded candidate
median **81.6173 tok/s** versus nearby HEAD-03 controls **82.6774** before
and **82.6855 tok/s** after (about **−1.3%**). The extra conversion launch
and scratch path do not pay back in the tested workload. **Reject** HEAD-05;
keep HEAD-03 W2 and RES-01 E3. No broad correctness, E3 or prefill run was
spent on this regression. The z840 source/object/library were rebuilt back to
the original DOWN-01 library SHA-256 `c737d259...`; model weights remained
untouched. Receipts in `gdn18/`: `w2-head-xhalf-*`,
`w2-i8res-n256-r4-xhalf-control-after.*`, and
`head-xhalf-post-down16.patch`.

### PRF-08 — current-champion output-head budget and W2 traffic floor

The existing `libapi-calls.so` scheduler callback was rerun on current
W2 HEAD-03, E3 RES-01, and native IQ3, with the same KVarN3/2 p0/n8
method as PRF-03. Each head was a one-node `result_output` CUDA graph segment
following `result_norm`. Nine-call median wall times were **0.817410 ms W2**,
**0.637950 ms E3**, and **0.475592 ms native IQ3**. The earlier W2 dense-head
probe was 1.534417 ms by the same method; HEAD-03 removed about 0.72 ms
there, consistent with its whole-model gain. This probe changes graph
segmentation and synchronizes, so phase times are diagnostic, not additive
Gate A throughput measurements. Receipts:
`gdn18/{w2-i8res,native,e3}-head-probe-n8-v2.{json,log}`.

W2's unchanged original-I8 vocabulary matrix plus F16 row scales requires
at least `248320*(5120+2) = 1,271,895,040` source bytes per one-token head.
The local CUDA driver reports 14,001,000 kHz max memory clock and 512-bit bus,
giving an idealized 1.792 TB/s peak and **0.710 ms** streaming floor. The
observed 0.817 ms diagnostic corresponds to 1.556 TB/s of source bytes if
streamed from DRAM, about 87% of that ideal peak. This is a roofline
*inference*, not a measured DRAM counter or guaranteed kernel lower bound;
the callback includes launch/sync cost and any cache effects. Under the
streaming assumption, even a perfect I8 head would save only approximately
0.108 ms from HEAD-03. It cannot alone explain the current raw deficit.

A nearby native IQ3 p0/n256 control median was **91.3812 tok/s**, giving a
95% threshold of **86.8121 tok/s**; nearby W2 HEAD-03/E3 RES-01 discovery
controls were about **82.69/83.68 tok/s**. Their approximate median latency
deficits are **0.575/0.431 ms/token**, respectively. This is prioritization
evidence, not Gate A confirmation: the controls are not a fully bracketed
five-sample crossing set, and native samples spread 88.56–92.16 tok/s.
**Next:** measure the recurrent/coded-projection body with a bounded method
that retains graph-on behavior or a validated isolated-kernel control. Avoid
further W2 head row/conversion sweeps without a new traffic-reducing mechanism
that preserves the original model weights.

### PRF-09 — whole-body scheduler span probe invalid

An isolated diagnostic copy `gdn18/api_calls_span.cpp` added a two-boundary
callback from `ffn_out-0` to `result_norm` to estimate the coded/recurrent
body on W2 HEAD-03 without touching production source. The expected nodes
matched, but the callback split the CUDA graph into 54-node and 3088-node
segments. Nine p0/n8 W2 span samples ranged **15.615–40.053 ms**, median
**18.703 ms**, above the approximately 12 ms/token graph-on raw decode
latency and far too variable to attribute a submillisecond deficit. This is
instrumentation perturbation, not evidence that the body regressed.
**Reject the span timing as a patch selector.** Do not compare it to native
or tune from its absolute value. Retain PRF-08 head phase estimates with
their stated limits; the next body measurement must preserve graph execution
or use a validated isolated kernel with matched inputs. Receipt:
`gdn18/w2-body-span-n8.{json,log}`. No E3/native span runs or runtime/model
changes were made.

### OVL-02 — existing overlap abort resolved to a missing up dependency

Hypothesis: the old R224 gate/up pair could reduce the coded FFN body, but
OVL-01 aborted because its thread-local pending gate was not consumed before
the next layer. An isolated trace of the opt-in on W2 HEAD-03 showed
`ffn_gate-0` K2 followed by `ffn_gate-1` K2 with gate 0 still pending. GGUF
metadata confirmed both W2 and E3 gate/up widths are K2/K3. A second trace
of the CUDA graph saw four consecutive nodes per layer: gate, SiLU, up, MUL.
The current graph's `ffn_up-0->src[6]` is null, so the default fused-up path
cannot consume the deferred gate. This is the specific missing dependency;
adding a second pending slot would not fix it. Receipts:
`gdn18/w2-overlaptrace-n8.log` and `gdn18/w2-fusetrace2-n8.log`.

### OVL-03 — atomic gate/up overlap, same-weight local champion

The isolated `overlap-atomic-post-down16.patch` matches those four graph nodes
under the existing opt-in and dispatches the gate SiLU and up/MUL inside one
graph visit. It checks the shared activation, shapes, output alias and the two
existing fusion memory guards. A first trace confirmed each K2 gate is paired
with its own K3 up and pending state is cleared before the next layer. No
GGUF or weight representation changed. Patch SHA256
`0e1cf9000443907d61f4a8e84f1e9bd8fa354d3595f9f2796226d971b1e70f10`;
clean CUDA library SHA256
`2d8ff1f1bdf8290a39e5f3e27df3f3e8d215c9fd6917ed9f970c316b485cb8a4`.
The patch dry-runs cleanly against the isolated DOWN-01 source. z840 build
sources, objects and library were restored after copying the candidate.

W2 HEAD-03 and E3 RES-01 each had an identical 2048-prompt plus 16-token
greedy KVarN3/2 continuation on the clean candidate; all checked logits were
finite. Graph-on p0/n256, one warmup plus three measured runs per process:

| Model | First control / candidate / control | Reverse-order candidate / control | Decision |
| --- | --- | --- | --- |
| W2 | 82.7279 / 84.8120 / 82.0114 tok/s | 85.5301 / 83.1023 tok/s | retain, about +2.9% in both brackets |
| E3 | 82.5664 / 85.6342 / 82.3473 tok/s | 86.1514 / 82.4568 tok/s | retain, about +3.9–4.5% |

These are local raw M=1 champions, not Gate A passes. The historical native
IQ3 control near 90.8 tok/s implies a provisional 95% threshold near
86.3 tok/s, so both remain short and require a fresh matched native control.
The four-node atomic matcher also selects the matching 2048-row prefill graph;
the CUDA gate/up concurrency branch itself requires single-row decode.
Intermediate speculative M>1 behavior remains unqualified. Receipts:
`gdn18/{w2,e3}-overlapatomic-continuation.*` and the tagged `ovl1`/`ovl2`
benchmark JSON/log/GPU samples. Diagnostic traces are not speed controls.

Targeted two-step full-vocabulary snapshots (steps 0 and 15) compared to
current controls: both models had identical 16-token continuations, finite
logits, 20/20 top-token overlap, and identical step-0 logits. At step 15,
W2 max absolute difference was 0.004146 and relative RMS 0.0002012; E3
was 0.022378 and 0.0009137. Each candidate's repeat snapshot was byte
identical to its own first run. This is candidate correctness, not release
qualification. Receipts: `gdn18/{w2,e3}-ovl03-{control,candidate}.*` and
`*-candidate-repeat.*`.

After the paired screens, a native IQ3 p0/n256 control median was 90.4497
tok/s, implying a **provisional 85.9272 tok/s** raw decode threshold. W2
OVL-03 repeated medians 84.8120 and 85.5301 remain below it. E3 repeated
85.6342 and 86.1514 straddle it; neither a single crossing nor a changing
native baseline suffices for Gate A. Confirm with the gate's longer matched
sample protocol only when both models have a clear speed margin. Receipt:
`gdn18/native-resident-n256-r4-ovl-native.json`.

### OVL-04 — overlap launch-order and split screen

On W2 OVL-03, the default gate-aux/up-main 8/8 splits remained fastest within
the small screen's noise. Default bracket medians were 86.1290 and 85.0699
tok/s. Up-first returned 85.4406; up-aux 85.6187. Both are neutral against
the bracket, so do not promote. Both projections at 12 splits returned
84.2393, and up-only 12 splits returned 84.5088; reject the larger split
settings. Each used p0/n256, one warmup plus three measured runs; no broad
correctness or E3 speed run was spent on the losing settings. Receipts are
tagged `orderpre`, `orderpost`, `upfirst1`, `upaux1`, `split12`, `up12` under
`gdn18/`. Next change must target a different measured cost.

### OVL-05 — longer matched raw decode gate screen

With the clean default OVL-03 binary and no speculative decoding, the same
RTX 5090, KVarN3/2 p0/n256 configuration used one warmup plus five measured
samples per fresh process. Native IQ3 medians before/after the two Escha runs
were **91.5999 / 89.6178 tok/s**, yielding 95% thresholds **87.0199 /
85.1369 tok/s**. Their sample ranges were 90.3984–91.6438 and
87.0291–92.3030 tok/s. W2 OVL-03 median **85.5741** (range
83.7493–86.3153), E3 OVL-03 **84.8767** (84.4997–86.7143).
E3 missed both bracket thresholds. W2 missed the before control and exceeded
the after control, so clock/session drift prevents a pass. **Gate A decode
remains open for both models.** Do not use the favorable short `ovl2` sample
as a release claim. Receipts: `gdn18/{native-resident,w2-overlapatomic,
e3-overlapatomic}-n256-r6-gate1*.{json,log,gpu-before.txt,gpu-after.txt}`.
The 5090 returned to 16 MiB idle after the screen. Continue on a mechanism
with a clearer margin before repeating a long gate confirmation.

### OVL-06 — check the atomic matcher's prefill interaction

The profile also enables 2048-row gate/SiLU/up/MUL finalization, so OVL-03's
four-node match can dispatch during prefill even though its concurrent CUDA
gate/up branch is single-row only. A paired p2048 KVarN3/2 screen (one warmup,
three measured) found W2 HEAD-03 3220.51 versus OVL-03 3239.88 tok/s, and
E3 RES-01 3148.41 versus OVL-03 3184.00 tok/s. No measured prefill
regression; both also retained the identical 2048-prompt plus 16-token
continuation and finite logits. Nearby native IQ3 median was 2853.30 tok/s,
unusually low versus earlier ~3.2K controls. Do not use this favorable native
session alone to declare the prefill gate; the direct paired Escha controls
are the reliable check that OVL-03 did not cost prefill speed. Receipts:
`gdn18/{w2-i8res,w2-overlapatomic,e3-resident-full,e3-overlapatomic,
native-resident}-p2048-r4-ovlpre.{json,log}`.

### R258CONV-01 — direct convolution-state view, retained local raw champion

SM89 R258/R260 history reported a larger same-weight recurrent-state and
fused-convolution gain. Current v0.4.7 enables both flags, but
`build_conv_state()` still calls generic `build_rs()` for convolution state.
Turning R258 off on W2 OVL-03 reduced p0/n256 median **85.489 / 85.306**
bracketing controls to **84.3363 tok/s**, showing the retained direct SSM
state path is active while leaving open whether the convolution gather and
R260 fusion execute. Hypothesis: sharing R258's exact M=1/one-sequence/
one-slot/source=destination guard with the convolution-state view will
remove that gather and enable existing R260. The isolated patch
`gdn18/r258-conv-direct-post-i8res.patch` SHA256
`0e79f306634f0676fddfeeb1a77eb32bb1349561a9b0f56fe78a302aec9a39d7`
changes only graph construction and guards the now-dead recurrent index
input's null allocator buffer. Other models, prefill and multi-row paths
retain the generic builder. The patch dry-runs against the HEAD-03 source.

The first executable build lacked the dead-input guard and aborted in
`llm_graph_input_mem_hybrid::set_input` at `GGML_ASSERT(buffer)` during the
short W2 continuation guard. **No speed timing or broad correctness was run
on that invalid binary.** The corrected build reported a libllama SHA256 of
`45aaa59aab31c5e9add2093d85538230e90cbf19fbf3eafef413cfbd82834509`,
but could not be copied off z840 before a storage fault. The invalid first
binary was moved out of the selectable `bin` directory. The patch was rebuilt
independently on healthy local storage as a CPU-backend `libllama` overlay;
`LD_LIBRARY_PATH` selects the validated OVL-03 CUDA library and resident
ggml libraries. Loader resolution was checked before execution. The local
library SHA256 is
`3b19f2fc018cecbf99da406379cc3c49411da526c7cf5dc3cf37a189c8af0318`.
This is a lab overlay, not a packaged release binary.

During the corrected build's restore, z840 `/dev/sdb1` reported buffer I/O
errors, an aborted ext4 journal, and a disconnected SAS device. The
`/mnt/storage` mount then disappeared; do not write to or rely on that build
tree until the host storage is repaired and checked. Source files under
z840 `/home/sean` were restored to local baseline SHA256 values; the build
tree on the missing mount could not be restored and must be treated as
untrusted when remounted. At last check port 8080 returned `{"status":"ok"}`;
its running binary and models reside under `/home/sean`, not the missing
mount. The local 5090 was idle at 16 MiB, and the E3/W2 GGUF full hashes
remained unchanged. This was an infrastructure interruption, not a rejected
performance hypothesis.

Both E3 and W2 completed the KVarN3/2 2048-prompt plus 16-token continuation
with finite logits and exact greedy token agreement against OVL-03. The raw
p0/n256 one-warmup/three-measured bracket found W2 **87.6076 tok/s** versus
OVL-03 **84.6177 / 84.8056** (+3.3–3.5%) and E3 **87.6214** versus
**86.1359 / 84.5384** (+1.7–3.6%). Paired p2048 prefill medians were W2
**3220.97** versus OVL-03 **3214.22**, E3 **3218.31** versus **3216.55**
tok/s. Retain the direct convolution-state view as the local M=1 champion for
both models. No E3/W2 weight, model file or representation changed.

The longer one-warmup/five-measured decode bracket used native IQ3 before
and after the E3/W2 candidates. Native medians were **89.0591 / 91.0607**
tok/s, giving 95% thresholds **84.6061 / 86.5077**. W2 was **86.4049**,
E3 **87.4367** tok/s. E3 exceeds both controls; W2 misses the later
threshold by **0.1028 tok/s** (about 0.12%) amid native/session drift.
**Gate A decode remains open**, pending a stable repeat with margin. A
matched p2048 one-warmup/five-measured prefill bracket gave native
**3129.44 / 3183.50**, W2 **3238.58**, E3 **3188.93** tok/s; both Escha
models exceed 95% of both controls. This supports the raw p2048 prefill
cell, not the populated-context serving cell. Receipts are tagged
`r258conv*` under `gdn18/` and include benchmark JSON, stderr and GPU samples.

Turning R260 off inside the W2 candidate gave **88.0828** tok/s, followed
by R260-on **88.0734** tok/s (one warmup, three measured each). A separate
launch-count-only diagnostic proved the route: R260-on captured **96**
`escha_ssm_conv4_update_silu_f32` launches in a p1+16 W2 continuation;
R260-off captured zero fused launches and **96** each of the old concat and
SSM-conv launches. Both paths produced identical 16-token continuations.
The diagnostic intercepted CUDA launches and was not used for speed timing.
Thus R260 **does dispatch**, but its end-to-end speed contribution was below
the short screen's resolution. Attribute the measured R258CONV-01 gain to
the direct convolution-state view and avoid further R260 flag sweeps.
Receipts: `gdn18/r260-route-probe.{cpp,so}` and
`gdn18/w2-r258conv-r260{,off}-route.{txt,log}`. DFlash2 M>1 remains
unqualified.

A second W2 one-warmup/five-measured native bracket confirmed that a raw
decode pass would be premature: native **91.7664 / 89.5102** tok/s around
W2 **86.6630**. The 95% thresholds are **87.1781 / 85.0347** tok/s. W2
again misses the higher control and exceeds the lower one. Stop repeating
this timing pattern for a favorable denominator; seek a measurable W2
gain or a method that reduces the observed native/session drift. Receipts:
`gdn18/{native-resident,w2-r258conv-local}-n256-r6-r258convgate2*.{json,log}`.

### PRF-10 — W2 versus native operating-state diagnostic

The second W2 long bracket slowed within its five measured repeats while
the native controls moved in the opposite direction. A bounded 200-ms
`nvidia-smi` trace during separate short p0/n256 diagnostic runs found
active W2 median **2692 MHz / 538 W / 66 C** versus native IQ3
**2797 MHz / 487 W / 64 C**; both had about 93% sampled utilization.
The card's enforced power limit was **575 W**. W2 active clock medians were
2700 MHz in the first half and 2685 MHz in the second; native stayed near
2797 MHz. W2 power samples reached 577 W. This is evidence that W2 operates
closer to the power ceiling and at a lower clock on this card; it does not
prove the entire throughput gap is caused by power or justify lowering the
native clock/raising the power limit to claim parity. These monitored runs
are diagnostics, not Gate A controls. Receipts:
`gdn18/{w2-r258conv,native-resident}-clockdiag.csv` and tagged `clockdiag`
benchmark JSON/logs. The next W2 patch should reduce real runtime work or
weight traffic; no further ratio reruns without a mechanism.

### DFL-05 — RT2 E3 head on the v0.4.7 raw champion

The proven SM89 E3 R=2 multi-row head patch applies cleanly on top of
R258CONV-01 and OVL-03 in isolated source
`/home/sean/work/escha-sm120-sprint-r258conv-rt2`. The RT2 patch SHA256 is
`d75c911d196b6bc8426230983342a9d5686fa453404f166029a8b74ad0b1b6e9`.
For the lean screen, only its LowGPU CUDA module was compiled as a lab
interposition library, SHA256
`fbb251369c9c6e90db6f7d870823f7a272154ea69fa4e349b2117e7adcc052ff`,
over the validated OVL-03 CUDA library and R258CONV-01 `libllama`. A short
depth-four E3 request returned three sane tokens; the launch-count probe
captured `lowgpu_packed_gemv_rt_kernel<2,5>` twice, proving five-row RT2
dispatch. No model weights or GGUF bytes changed.

On the same RTX 5090, 8K/one-slot KVarN3/2 target, Q4_K_M DFlash2 draft
with q4_0/q4_0 draft cache, depth four, reasoning off, 500 output tokens,
Club-3090 narrative and code prompts, one warmup plus three measured served
requests per class, E3 RT2 versus an adjacent identical-config control:

| E3 DFlash2 N4 | Narrative decode mean | Code decode mean | Decision |
| --- | ---: | ---: | --- |
| OVL-03/R258CONV without RT2 | 65.86 tok/s | 94.91 tok/s | control |
| RT2 interposed | **86.61 tok/s** | **120.37 tok/s** | retain isolated lead; +31.5% / +26.8% |

All measured requests produced 500 tokens. The RT2 result is a meaningful
same-weight speed lead, but is below the 2x native served gate and has only
a short output/route guard. It is **not** release correctness qualified.
Receipts: `gdn18/e3-r258conv-local-dflash-n4-{rt2probe1,rt2screen1,
rt2control1}-{server,bench}.log`, short JSON, and `dflash-screen.sh`.
The integrated full CUDA build on z840's healthy `/home/sean` root was
stopped after 131/248 objects because KVarN attention templates were taking
minutes per object. Compiled objects are retained; no full integrated RT2
library or release package exists yet. z840 port 8080 stayed healthy.

### DFL-06 — W2 DFlash2 N4 baseline and native served denominator

W2 initially aborted during the cheap short request because the raw-only
original-I8 head flag reached a five-row DFlash verification batch. The
served W2 selector now unsets that flag after applying the raw profile,
allowing its existing multi-row head path. A fresh short request passed,
then the identical 500-token served screen measured **86.11** narrative
and **125.55** code tok/s (three measured means, after one warmup). Mean
draft acceptance was about 45–48% narrative and 75–81% code; useful lengths
were about 2.7–2.9 and 4.0–4.2 tokens per cycle. This is the first valid
depth-four W2 speed screen on the current v0.4.7 champion, not a broad
correctness pass. Receipts: `gdn18/w2-r258conv-local-dflash-n4-{w2base1,
w2guard2,w2screen2}-*`.

A matched non-speculative native IQ3 GGUF server using the same 5090,
one-slot 8K context, KVarN3/2, batch/microbatch, reasoning-off prompts
and 500-token output measured **85.15** narrative and **85.42** code
tok/s. Thus provisional 2x served thresholds are **170.30 / 170.84**
tok/s. E3 RT2 is about **1.02x / 1.41x** native; W2 is about
**1.01x / 1.47x**. Gate B remains far open, especially narrative. These
are discovery screens, not the final multi-prompt release pack or a gate
pass. Receipt: `gdn18/native-resident-dflash-n4-nativecontrol1-*` (the
native arm has speculation disabled despite the common receipt stem).

### PRF-11 — E3 depth-four cycle budget, target verification dominates

An opt-in `LLAMA_TRACE=2` and trace log level 5 captured 45 E3 narrative
speculative cycles during a separate 128-token diagnostic request. Median
cycle **26.517 ms** comprised target decode **16.992 ms**, draft
**6.483 ms**, process **0.969 ms**, and verify **0.649 ms**; sampling
**0.511 ms** is a subset of verify, not an extra phase. Mean useful output
was **2.82 tokens/cycle**. At that acceptance, 2x the matched native
narrative 85.15 tok/s requires at most about **16.6 ms/cycle**, below the
current median target decode phase alone. This budget shows that a small
isolated patch cannot plausibly close the narrative gate. PRF-12 below
supersedes the provisional priority to attack the target head again: RT2
already removes most measured M=5 target overhead. Trace logging perturbs
timing, so use these as
a phase budget, not a served speed result. Receipt:
`gdn18/e3-r258conv-local-dflash-n4-trace2-server.log`.

### PRF-12 — target-only M=5 curve moves the speculative priority

The copied SM89 M-scale probe was compiled against the current v0.4.7
candidate and run on the RTX 5090 with 8K context, KVarN3/2, UB512,
p32 no-reset, all logits, two warmups and eight measured forwards. These
are synthetic target-only timings, not Gate B served results.

| Runtime | M=1 median | M=5 median | M=5 / M=1 |
| --- | ---: | ---: | ---: |
| Native IQ3 GGUF | 9.8903 ms | 15.3860 ms | 1.556 |
| E3 without RT2 | 10.1262 ms | 22.2158 ms | 2.194 |
| E3 with RT2 enabled | 10.4849 ms | 16.4804 ms | 1.572 |
| W2 served selector | 10.9606 ms | 16.7794 ms | 1.531 |

RT2 saves **5.7354 ms (25.8%)** at E3 M=5 and leaves E3 only **1.0944
ms** behind native; W2 M=5 is **1.3934 ms** behind native. The first RT2
probe lacked `ESCHA_E3_HEAD_RT=2` and is invalid; the table uses the
corrected enabled arm. The target-only M=5 overhead is now a smaller
opportunity than the serial draft/cycle costs and narrative useful-token
yield seen in PRF-11. Next measure draft graph/cache behavior and target
delivery/rollback costs, then patch only a demonstrated removable phase.
Keep E3/W2 GGUF weights unchanged. Receipts: `gdn18/*-mcurve32.jsonl`;
`e3-rt2-enabled-mcurve32.jsonl` is the valid RT2 arm.

### PRF-13 — bounded code-cycle diagnostic and raw W2 repeat

An E3 RT2 128-token code request under `LLAMA_TRACE=2`, log level 5,
depth four, returned 128 tokens and 31 reused graphs. Across 32 cycles,
median target decode/draft/process/verify/full-cycle times were
**17.062 / 7.920 / 1.087 / 0.963 / 28.637 ms**. Mean useful output was
**3.969 tokens/cycle** and mean cycle time **29.957 ms**. After two warm
cycles, draft median was 7.964 ms; in the last half it rose to 10.858 ms,
without an observed graph warmup reset. This is a short, trace-perturbed
diagnostic; it cannot explain the increase causally or stand in for a
500-token gate result. At the observed yield and native code control
85.42 tok/s, the 2x budget is ~23.2 ms/cycle, roughly 6.7 ms below this
trace's mean. The 500-token code screen remains 120.37 tok/s E3 RT2.
Receipt: `gdn18/e3-r258conv-local-dflash-n4-codetrace1-{server,bench}.log`.

A fresh native/W2/native/W2/native raw p0/n256, KVarN3/2 bracket used
one warmup plus five measured samples per fresh process. Native medians
were **89.5859 / 90.4090 / 90.4191** tok/s; W2 R258CONV-01 medians
were **86.3711 / 86.4297** tok/s. Both W2 medians exceed 95% of the
highest native median (**85.8981 tok/s**) by only 0.47–0.53 tok/s;
individual W2 samples fell below that threshold (minimum 84.8462).
This supports a *near-parity median* but does not meet the sprint's
no-overlapping-uncertainty promotion rule. Gate A decode remains open.
Do not repeat brackets in search of a favorable denominator; seek a
measured same-weight W2 gain or a more stable controlled method. Receipts:
`gdn18/{native-resident,w2-r258conv-local}-n256-r6-prf13-*.{json,log}`.
No E3/W2 GGUF weights or representations changed.

### PRF-14 — native speculative control isolates the draft penalty

The same RTX 5090, 8K one-slot KVarN3/2 server and Q4_K_M DFlash2 drafter
were run with the native IQ3 GGUF target via an explicit opt-in in
`gdn18/dflash-screen.sh`. This is a diagnostic speculative control;
the default native parity denominator still has speculation disabled.
The native speculative 500-token narrative/code decode means were
**128.58 / 177.78 tok/s** (one warmup, three measured). Native code
exceeded its provisional 2x non-spec denominator of 170.84 tok/s;
native narrative remained below 170.30. E3 RT2 scored 86.61/120.37
and W2 scored 86.11/125.55 under the earlier same-shape screens.
This native speculative result is not an Escha Gate B pass and the runs
are not a final interleaved release pack.

The native and E3 code acceptance on the 500-token screens was similar:
roughly 3.8–4.2 useful tokens/cycle for native and 3.9–4.0 for E3.
Separate 128-token `LLAMA_TRACE=2` diagnostics gave these last-half
medians (ms), with mean useful tokens/cycle shown separately:

| Target | target | process | draft | verify | full cycle | useful |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native IQ3 | 15.750 | 0.581 | 3.017 | 0.883 | 20.300 | 4.00 |
| E3 RT2 | 17.143 | 1.655 | 10.858 | 0.971 | 30.388 | 4.00 |
| W2 served selector | 17.212 | 1.691 | 8.038 | 1.050 | 29.398 | 3.71 |

The matched code trace points to the **draft step** as the largest
Escha-specific phase gap: +7.84 ms E3 and +5.02 ms W2 in the last half,
versus target decode +1.39/+1.46 ms. E3's all-cycle median draft gap
was +4.89 ms, so the last-half difference should not be generalized
without a longer controlled trace. Graphs were reused (native 1189,
E3 1181 in the final 500-token requests), and no repeated warmup reset
appeared in the short code trace. This lowers the expected value of a
graph-cache patch. The next concrete question is which same-weight
drafter operation—target head projection, target embedding, selector,
or synchronization—accounts for the phase gap. Profile that boundary
before patching. Do not substitute a different GGUF head or alter the
E3/W2 weights. Receipts: `gdn18/native-resident-dflash-n4-natspec*`,
`gdn18/{e3,w2}-r258conv-local-dflash-n4-codetrace1-*`.
The original E3/W2 GGUF SHA-256 hashes were rechecked after these runs:
`746bd40841fb18b9c1918923e89c007df70b5e4f3de64290d1399593e70c96b0`
and `3f93cbe77a20f1fa7272741757596cac66a66457d7ecaed1e5a6e4baa409535e`.

### PRF-15 — split-graph draft-stage probe rejected for body attribution

Hypothesis: the E3/W2 draft head, body, or DFlash2 selector explains the
PRF-14 draft phase gap. An isolated LD_PRELOAD scheduler callback probe
(`gdn18/dflash-stage-probe.cpp/.so`) marked `inp_noise_embd`,
`result_norm`, `result_output`, and `dflash2_lattice` only in the DFlash2
draft graph. Its first version had an out-of-range diagnostic sample index
and crashed before a timing result; corrected version completed 128-token
code requests for E3 RT2, W2, and native IQ3. The diagnostic made native
served decode fall to **123.60 tok/s** from **189.95 tok/s** in the
earlier short trace and reported native body **10.13 ms** versus E3
**4.99 ms** and W2 **5.15 ms**—the reverse of the graph-preserving
PRF-14 draft ordering. Thus the callback's graph splits/CPU placement
materially changed execution; **reject its body/selector timings** and
do not choose a kernel patch from them. It is not a Gate B screen.

The narrower `result_norm`→`result_output` diagnostic was repeatable
under the same probe method: E3 RT2 **1.50 ms**, W2 **1.59 ms**, native
**0.67 ms** median (30/30/33 calls). Earlier two-boundary head probes
gave comparable E3/native values, but the callback still changes graph
execution. The head may explain roughly 0.8–0.9 ms of the draft gap;
that is insufficient to account for the 5–8 ms PRF-14 phase difference.
Next use a graph-preserving draft measurement or a controlled same-weight
runtime candidate that isolates a larger mechanism. No E3/W2 model or
weight representation was changed. Receipts:
`gdn18/{e3,w2}-r258conv-local-dflash-n4-stageprobe2-*`,
`gdn18/native-resident-dflash-n4-stageprobe2-*`, and matching `headprobe1`
logs. The local GPU returned to 16 MiB idle after the screens.

### DFL-07 — DFlash graph-key collision isolated and removed in a lab overlay

**Hypothesis:** the Escha DFlash feature-injection graph and five-row
draft-block graph alternate under the same CUDA graph-cache key, preventing
either from reaching replay. A CUDA-event interposer wrapped whole graph
launches without splitting their nodes. In 128-token code diagnostics,
E3 RT2 replayed the target graph 31 times but **zero** draft graphs;
native IQ3 replayed target 29 times plus draft injection/block 29/31
times. Native draft GPU graph medians were **0.26/1.99 ms**. The shape
probe traced E3 draft injection (88 nodes) and draft block (772 nodes)
to the **same first-node address**. Native's 88/771-node draft graphs
had distinct first-node addresses. Bee's CUDA backend keys captures only
by that address, so E3's alternating shapes share a warmup slot.

A lab-only `dflash-key-split-interpose.so` gives the Escha block graph a
stable distinct first-node descriptor with unchanged sources and output
buffer. E3 RT2 then replayed the injection/block graphs 31/32 times,
at **0.27/2.87 ms** median GPU time; target M=5 stayed near **16.26 ms**.
W2 uses the same graph shape and also dispatched the key-split route.
This is a mechanism screen, not the release implementation: the overlay
clones a graph descriptor and has not had long-run state/lifetime review.

The ordinary 8K one-slot, KVarN3/2, depth-four, 500-token Club screen
used one warmup and three measured requests per prompt. Means are shown
only for discovery; same-session off controls ran after the on arms:

| Target | key split | narrative decode | code decode | on/off gain |
| --- | --- | ---: | ---: | ---: |
| E3 RT2 | off | 93.27 tok/s | 120.13 tok/s | — |
| E3 RT2 | on | **112.87 tok/s** | **158.99 tok/s** | **+21.0% / +32.3%** |
| W2 served selector | off | 94.39 tok/s | 127.83 tok/s | — |
| W2 served selector | on | **110.37 tok/s** | **161.68 tok/s** | **+16.9% / +26.5%** |

All on-arm measured requests returned 500 tokens. E3 off code run 1
ended at 395 tokens; W2 off code run 3 at 408; the displayed off means
include their actual-length throughput and need a fixed-length repeat
for release claims. The gain is large enough to retain the key split as
the current **experimental DFlash champion**, below Gate B. Relative to
the provisional native non-spec served 2x thresholds 170.30/170.84,
E3/W2 code still need about +7.5%/+5.7%; narrative remains much farther
away. These are not matched release-pack gates.

The 500-token narrative receipts report mean useful lengths of about
**2.74–2.86 tokens/cycle** for E3 and **2.62–2.94** for W2, similar to the
native IQ3 drafter's **2.65–2.96** on the same prompt. At 170.3 tok/s,
2.8 useful tokens permit only about **16.4 ms per complete cycle**. The
post-fix E3 short trace still spent about **17.1 ms in target verification
alone**, before drafting or scheduling. This is a directional bound from
different-length diagnostic requests, not a matched gate measurement: a
draft-only micro-optimization cannot by itself close the narrative gap.
Next seek an actual reduction in target verification cost and/or a larger
useful-token yield without changing weights or degrading output quality.

For cheap correctness, independent 64-token temperature-zero, seed-1234
served continuations were text-identical with the key split off/on for
both E3 and W2. A further A/B/A sequence in one server process used two
different 64-token prompts, revisited A after B, and matched all three
texts between off/on on both models; each A repeat also matched its
first response. This is targeted state/rollback evidence, not full release
qualification. The E3/W2 model GGUFs were untouched. A clean source
version keys CUDA captures by `(first_node_ptr, n_nodes)` in isolated
`/home/sean/work/escha-sm120-sprint-r258conv-rt2`; the z840 root-disk
CUDA build is in progress. Do not package the lab interposer. Receipts:
`gdn18/*-graph{event,key}*`, `gdn18/*-keysplit{serve,control,guard}1-*`,
`gdn18/*-keysplitstate{off,on}1-state.json`, and
`gdn18/dflash-key-split-interpose.cpp/.so`. The incremental source patch
`gdn18/dflash-key-shape-post-ovl03.patch` has SHA-256
`8e8c11d42005b8a38df241818b8365eb42904a78f75ace163bed8baa4b3f6d2a`;
reverse dry-run on the isolated source succeeds.
The full CUDA rebuild remains active on z840 root storage as PID
`2401868`, log
`/home/sean/work/escha-sm120-sprint-r258conv-rt2-build2/keysplit-build-j28.log`.
Ninja was restarted from completed objects with 28 compiler jobs to
shorten the CUDA template rebuild; the port-8080 service remained healthy.
No integrated speed or correctness result exists yet.
DeepSeek 4.1's bounded read-only review found no ABI, capture-warmup,
eviction, or NVCC compile blocker for the pair key; it compiled the
changed CUDA translation unit under graph-on/off definitions. It noted
one residual performance risk: two distinct graphs with the same first
node and node count would still share a key, but the full property
snapshot forces direct execution on mismatch rather than replaying
wrong graph data. Do not expand the key without evidence that this
remaining collision occurs; extra identity fields can themselves make
stable shapes appear unique across calls.

A separate **unrun CPU implementation** of the selector experiment is saved as
`gdn18/dflash-selector-dp-experimental.patch` (SHA-256
`19e8979211c7f05d991fb3f1d4796b8e8e219349c1b8aedc6bf8d630593ebb2d`).
It opt-in selects the highest-probability four-step path from the
existing DFlash2 top-16 transition lattice; no weights or GGUFs change.
The source copy was restored and the patch passes a dry-run. The same
mechanism was subsequently screened in DFL-08 and rejected. It is not
part of the runtime champion or a release artifact.

### DFL-08 — same-lattice selector best-path screen, rejected

**Hypothesis:** a four-step highest-probability path through DFlash2's
existing top-16 transition lattice would raise narrative accepted length
without changing weights. A lab-only `dflash-selector-dp-interpose.so`
made the default greedy walk follow that path. Its 64-token E3 A/B/A
guard was text-identical to the key-split control; 3/50 short-guard
draft paths changed. In a same-session E3 8K, one-slot, KVarN3/2,
depth-four, 500-token narrative screen (one warmup, three measured),
94/718 draft paths changed. All measured requests returned 500 tokens.

| Selector | Mean decode | Mean useful narrative length |
| --- | ---: | ---: |
| Best path lab overlay | 109.99 tok/s | 2.79 tokens/cycle |
| Existing greedy, matched control | 110.96 tok/s | 2.82 tokens/cycle |

The 0.9% slower speed is within screen noise and acceptance did not
improve. **Reject** the selector patch and do not spend a W2 or longer
correctness run on it. The saved CPU patch remains an unrun archive of
the same mechanism, not a candidate. Receipts:
`gdn18/e3-r258conv-local-dflash-n4-selectordp{guard,narr,control}1-*`;
source `gdn18/dflash-selector-dp-interpose.cpp/.so`.

### DFL-09 — integrated graph-key library, E3/W2 raw Gate A microbenchmarks

The isolated SM120 CUDA build linked successfully on z840 root storage.
The integrated `libggml-cuda.so.0.23.0` SHA-256 is
`3e545ad7b6525e73903db361146f6d18d8f73b417d9e65ec67778a5072deaefc`;
the copied local library and remote build hashes match. `run.sh
r258conv-rt2` loads this library with the R258CONV-01 `libllama` overlay;
`ldd` confirmed both selected paths. The E3 and W2 GGUF SHA-256 values
remain `746bd40841fb18b9c1918923e89c007df70b5e4f3de64290d1399593e70c96b0`
and `3f93cbe77a20f1fa7272741757596cac66a66457d7ecaed1e5a6e4baa409535e`.

E3/W2 64-token A/B/A guards returned identical text to the lab
key-split control, including the repeated A after B. An instrumented
128-token E3 code request recorded 33 target, 33 draft-block and 32
draft-injection CUDA graph replays; draft block/injection still have the
same first-node pointer but distinct node counts (772/88). Whole-graph
GPU medians were 16.44/2.86/0.26 ms respectively. The instrumented
throughput is diagnostic, not a gate sample. Receipts:
`gdn18/*-r258conv-rt2-dflash-n4-keysplitintegrated{guard,graph}1-*`.

Uninstrumented 8K one-slot, KVarN3/2, depth-four, 500-token served
screen, one warmup and three measured per prompt; all measured requests
returned 500 tokens:

| Model | Narrative mean decode | Code mean decode |
| --- | ---: | ---: |
| E3 | 108.76 tok/s | 161.58 tok/s |
| W2 | 114.35 tok/s | 158.01 tok/s |

This reproduces the large lab gain, with ordinary request-to-request
spread. Gate B remains open; the provisional 2x native served targets
are about 170 tok/s for each class. Receipt:
`gdn18/{e3,w2}-r258conv-rt2-dflash-n4-keysplitintegratedserve1-*`.

For raw no-spec parity, all arms used the same CUDA backend, GPU,
KVarN3/2, FA on, batch 2048 and three `llama-bench` repetitions.
The p0/n256 decode bracket used UB512; two W2/native pairs measured
87.25/89.53 and 87.28/89.05 tok/s (97.46% and 98.02% of native).
E3 measured 88.52 tok/s against the first 89.53 native control
(98.88%). The p2048/n0 prefill bracket used UB2048: E3 3222.77,
W2 3222.63 and native 3211.24 tok/s, both Escha models about
100.35% of native. **Gate A's matched raw microbenchmark cells pass
on this integrated candidate.** This is not a served or release
qualification claim; retain the exact config in future checks.
An initial UB512 prefill screen measured ~2504/2506 versus native
3069.90 tok/s and was invalid for comparison with prior UB2048
prefill data; it remains a separate UB512 observation. Receipts:
`gdn18/{e3,w2,native}-r258conv-rt2-{n256,p2048-ub2048}-keysplitintegrated*.json`.

### PRF-16 — graph-preserving M=5 kernel differential on integrated champion

A lab-only NVTX interposer tagged only the `llama_decode` call with five
verification rows. Nsight Compute collected `gpu__time_duration.sum`
for that range, with one deterministic p32, KVarN3/2, UB512 forward per
model. The first W2 probe aborted because `run.sh` sets its **raw-only**
I8 head; the valid rerun unset that flag exactly as the served DFlash
launcher does. These kernel replay durations and their sums are
**instrumented attribution**, not wall latency or speed-gate evidence.

| M=5 target | kernel launches | summed profiled GPU time | largest families |
| --- | ---: | ---: | --- |
| E3 RT2 | 2789 | 25.29 ms | coded GEMV 10.86 ms/400; finalization 2.70 ms/400; CUTLASS 3.03 ms/96; head 1.92 ms/1 |
| W2 served head | 2788 | 25.17 ms | coded GEMV 10.83 ms/400; finalization 2.70 ms/400; CUTLASS 3.04 ms/96; head 1.91 ms/1 |
| Native IQ3 | 2436 | 21.92 ms | quantized matvec spread over types; input quantization 1.18 ms/497 |

E3/W2 share essentially the same expensive coded projection and
finalization families, so a shared same-weight bridge/kernel improvement
has more expected value than another model-specific head flag. The
official bridge consumes the existing code/rin/rout GPU pointers and
launches a vendor cubin; there is no evidence yet that its 400
projection/finalization pairs can be fused safely. Do not equate the
25-ms profiled sum to the uninstrumented ~16.5-ms graph time. Next
inspect exact bridge launch geometry and the 400 finalization boundaries;
make one bounded patch only if it plausibly removes multiple milliseconds
from M=5 without touching model weights. Receipts:
`gdn18/prf16-{e3,w2,native}-m5-all-range-ncu*`,
`gdn18/m5-nvtx-range.cpp/.so`.

### DFL-10 — draft KV precision and selector confidence screens

The screen script now accepts optional draft KV type and DFlash2
confidence threshold environment variables; defaults remain Q4_0/Q4_0
and `p_min=0`. This changes only runtime cache precision/selection,
not E3/W2 or draft GGUF weights. Same integrated 8K one-slot depth-four
500-token narrative screen, one warmup and three measured:

| Model | Draft cache / p_min | narrative mean decode | mean useful length | GPU memory at end |
| --- | --- | ---: | ---: | ---: |
| E3 | Q4 / 0, fresh control | 108.23 tok/s | 2.77 | 11278 MiB |
| E3 | F16 / 0 | 116.71 tok/s | 2.87 | 11308 MiB |
| E3 | Q4 / 0.20 | 115.11 tok/s | 2.91 | 11284 MiB |
| E3 | Q4 / 0.35 | 69.99 tok/s | — | 11284 MiB |
| W2 | Q4 / 0, fresh control | 111.60 tok/s | — | 15168 MiB |
| W2 | F16 / 0 | 115.82 tok/s | 2.92 | 15198 MiB |
| W2 | Q4 / 0.20 | 105.10 tok/s | 2.83 | 15168 MiB |

E3 F16 code was 155.09 tok/s against a nearby Q4 code mean of 160.67;
one Q4 measured code request ended at 432 tokens, so that code
comparison cannot support promotion. E3 F16 and 0.20 each show only a
small narrative lead, far short of Gate B; W2 0.20 regressed against
its fresh control. Reject 0.35, avoid a shared 0.20 profile, and keep
Q4/0 as the current default. F16 remains an opt-in experimental lead
until a repeat with equal output lengths and targeted correctness
supports a model-specific choice. All arms passed short output guards;
E3 F16 and 0.35 plus W2 0.20 passed 64-token A/B/A state guards.
Receipts:
`gdn18/{e3,w2}-r258conv-rt2-dflash-n4-{draft*,pmin*}-*`.

### DFL-11 — SM120 M=5 split-target screen, rejected

2026-09-24. Isolated worktree
`/home/sean/work/escha-sm120-sprint-r258conv-rt2`; clean integrated
library `3e545ad7…` stayed frozen. Hypothesis: doubling the inherited
1024-CTA split target to 2048 for five-row coded GEMV would use the
RTX 5090 more fully. The opt-in `ESCHA_SM120_M5_SPLIT_TARGET=2048`
candidate was built separately (`68305cdb…`); the source change is saved
as `gdn18/m5-split-target-post-dfl09.patch`. It touched only the runtime
split decision, never the E3/W2 weights. A deterministic p32/UB512,
KVarN3/2 all-row M=5 logits dump was finite with the same argmax and
20/20 top-20 overlap in every row for E3 and W2 (relative RMS
0.000163/0.000139). The first paired E3 target-only screen used two
warmups and eight measured forwards: **16.8423 ms candidate versus
16.6269 ms champion median, 1.30% slower**. Reject the candidate at
Level 1. No W2 throughput, served speed, or release correctness run is
warranted. Both local and z840 source copies were restored from
`gdn18/m5-split-target-base.cu`; the clean champion library remains
available at `gdn18/r258conv-rt2/bin`. Receipts:
`gdn18/m5split2-{candidate,control}-{e3,w2}/m5.f32`,
`gdn18/m5split2-e3-{candidate,control}-screen.jsonl`.

### REV-01 — independent review of the next experiment

DeepSeek 4.1's read-only review of PRF-16/17 and DFL-11 identified 96
five-row F16 cuBLASLt matmuls with a two-CTA grid per M=5 verifier,
about 3.03 ms of **profiled** GPU time in both E3 and W2. The normal
M=1 dispatch takes MMVF; M=5 fails its F16 Ampere-MMA heuristic and
falls through because MMF requires a larger output row multiple. The
same review pointed to 400 finalization launches as a second candidate,
but neither profiled family time is a wall-time saving. It also noted
that the same-drafter native IQ3 narrative screen reached only
128.58 tok/s against 85.15 non-spec, so the 170.30 tok/s narrative
gate likely needs more useful tokens per speculation cycle as well as
cheaper verification.

The already-running z840 production BeeLLaMA server reported Qwen3.8-27B
Q4_K_M on port 8080; it was not restarted or reconfigured. Qwen's
bounded direct review favored the small F16 draft-cache prose lead and
warned against treating NCU bandwidth as guaranteed headroom. It also
suggested top-k pruning; that is **not promoted** because it can alter
the speculative output distribution and has no matched speed/quality
evidence. Review receipts: `gdn18/qwen-review-{request-direct,response-direct}.json`.
The next bounded experiment is an opt-in M=5 small-F16 MMVF dispatch
screen, followed by a target-only latency check. Keep it only if it
routes the intended 96 calls and produces a repeatable gain.

### DFL-12 — five-row F16 projection dispatch, experimental champion

2026-09-24. Hypothesis: the 96 tiny two-CTA cuBLASLt F16 matmuls in
M=5 verification are slower than the existing five-column MMVF kernel.
In isolated `/home/sean/work/escha-sm120-sprint-r258conv-rt2`, the
opt-in `ESCHA_SM120_M5_F16_MMVF=1` route permits MMVF only on SM120+
for F16, five input rows and at most 128 output rows, after the usual
alignment guards. Raw M=1 and p2048 routes remain unchanged. The source
patch `gdn18/m5-f16-mmvf-post-dfl09.patch` has SHA-256
`0fae90a05206dd4fbc94046947a5540145166be794c59f85112c28e3bf4b3a56`;
the separate experimental CUDA library at `gdn18/r258conv-rt2-mmvf5/bin`
has SHA-256 `84889e2dd63d5081ecd4b533d2e42c7229921d86ceb868153f19477012602be1`.
The frozen DFL-09 library remains untouched. No GGUF weights or draft
model weights were changed.

Same-library env-off/on deterministic p32, UB512, KVarN3/2, five-row
all-logits probes were finite. Every row retained the same argmax and
20/20 top-20 membership for E3 and W2; worst absolute logit delta
was 0.00419/0.00469. With two warmups and eight timed forwards,
E3 M=5 verification median fell **16.7196→14.4321 ms (−13.68%)**;
W2 fell **17.1838→14.9758 ms (−12.85%)**. A matched NVTX/Nsight
Compute one-forward E3 route trace counted 96 cuBLASLt plus 192
conversion launches in the env-off control and 96 five-column MMVF
launches with no cuBLASLt/conversion in the candidate. Other kernel
families matched, so this patch removed exactly those extra launches.
NCU time is attribution only; the target-only medians are the latency
evidence. Receipts: `gdn18/mmvf5-{e3,w2}-{candidate,control}-{guard,bench}/`,
`gdn18/mmvf5-e3-{control-,}route-ncu.csv`.

Same-library, same 8K one-slot Q4 draft-cache DFlash2 depth-four served
screens (one warmup, three measured requests, 500-token limit):

| Model / class | Env off | MMVF5 on | Delta | Length status |
| --- | ---: | ---: | ---: | --- |
| E3 narrative | 109.28 | 123.31 tok/s | +12.84% | all six measured requests 500 |
| W2 narrative | 114.25 | 125.06 tok/s | +9.46% | all six measured requests 500 |
| E3 code | 158.92 | 166.05 tok/s | +4.49% | one control request ended at 489 |
| W2 code | 160.90 | 169.95 tok/s | +5.62% | one control request ended at 473; candidate spread high |

Narrative gains are matched, length-equal evidence; code gains are
provisional pending a fixed-length repeat. The approximate native
served 2x targets remain 170.30/170.84 tok/s, so **Gate B remains
open** on both classes and models. E3/W2 64-token A/B/A guards matched
the env-off texts exactly, including the A repeat; a 256-token
deterministic numbered-list continuation also matched exactly for both
models (SHA-256 `d48e237881bcddf4e2978fb8d7c3151935934272ffafbbd106d999f87f018bfc`).
These are targeted candidate guards, not full quality qualification.
Promote MMVF5 as the **opt-in experimental M=5 champion** on Q4 draft
cache, keeping the integrated DFL-09 library as its clean control.
Receipts: `gdn18/{e3,w2}-r258conv-rt2-dflash-n4-mmvf5{cand,control}1-*`,
`gdn18/{e3,w2}-r258conv-rt2-dflash-n4-mmvf5{state,long}*-*`.

An additive F16 draft-cache narrative screen on E3/MMVF5 measured
124.04 tok/s versus the nearby Q4/MMVF5 123.31, within ordinary
spread; do not promote that combination or spend a W2 additive screen
without new evidence. The accidental first run with
`ESCHA_DFLASH_ONLY=narrative` produced no benchmark because the harness
accepts `narr`; the corrected `mmvf5f16narr2` receipt is the valid one.
Qwen's follow-up review correctly warned that early code endings
invalidate a 500-token code gate claim, but its blanket rejection of
the narrative comparison overlooked the six full-length narrative
requests. DeepSeek 4.1 confirmed the MMVF five-row specialization and
its alignment guards; it flagged recurrent trajectory drift as the
main correctness risk, addressed provisionally by the exact 256-token
continuations above. Longer quality/KLD qualification waits for the
speed gate or a specific numerical concern.

Do not promote based on old v0.4.6 results. The prior SM120 W2/E3 direct-F32
preview measured 82.37/84.63 decode and 2902.77/2758.21 prefill on different
source/model shape; it is context for likely bottlenecks, not the v0.4.7 result.

## Historical experimental prerelease progression (r3 through r6)

This section records earlier decisions. Current state is r8 / PKG-07 above.
The requirement below to exclude GDN refers to the retired mismatched payload,
not the later source-matched GDN implementation.

The earlier relocatable experimental review asset is
`dist/escha-beellama-v047-sm120-experimental-r3.tar.zst`. It includes the portable
`$ORIGIN` relink, exact selected SM120 bridge and cubins, CUDA 13.0 user-space
libraries, a model/payload-hashing `doctor`, a one-command E3/W2 MTP launcher,
source rebase patch, licenses, hashes, and benchmark summaries. It excludes the
GGUFs. The r3 manifest records the binary ABI floor: glibc 2.38 and
GLIBCXX_3.4.32. Ubuntu 24.04 under WSL is the tested install target; older
distributions need another build and qualification. The package doctor passed
on RTX 5090 with both exact model SHA-256 hashes;
the packaged launcher returned `PARITY_OK` for both E3 and W2, mapped the
selected bridge, and released GPU memory to 16 MiB after each shutdown.
Receipt: `tuning/v047-sm120-package-launcher-20260924/result.json`.

The r4 asset packages the BM64 bridge and two explicit profiles:
`long-32k` (default, UB1024) and `fast-8k` (opt-in, UB2048). A fresh extraction
passed full hashes, E3/W2 doctor, launcher, corpus, and bounded profile smokes.
It is **withdrawn from use** because its enabled K=1 GDN bridge yields NaN
logits for full 2048-token batches. The MTP K3 server path did not load that
bridge, but the package exposes a broken non-MTP path. A repaired experimental
asset must disable and exclude GDN; old GDN-enabled bench receipts above are
historical and cannot justify a speed claim. Corrected GDN-disabled prefill is
2980 E3/2953 W2 F16 and 2869 E3/2791 W2 KVarN3/2 tok/s on the exact r4
binary with the selected BM64 code-GEMM bridge. A corrected same-card native
IQ3 benchmark measured 3343 F16 and 3105 KVarN prefill tok/s. The SM89 native
anchor and hardware-normalized ratios were corrected in NAT-04 and GDN-09.
Effective MTP server prefill is 2079 W2 and
2030 E3 tok/s versus native Escha SGLang at 3076 W2 and 3279 E3. The
SGLang reference uses original safetensors heads and a 2041-token prompt;
Bee uses merged GGUFs and 2048 tokens, so this is a runtime comparison, not a
matched-weight quality claim. The known FFN packed-GEMM mainloop accounted for
94.6% of an earlier same-runtime W2-vs-IQ3 prefill gap (GBrain “BASE-01 —
ESCHA W2 vs IQ3_XXS LowGPU same-runtime breakdown”). EXP-07 through EXP-10
were rejected; do not repeat those kernels without new evidence. The named
5090/model hashes were checked with the 8K fast and 32K long profiles only.

The repaired r5 archive is
`dist/escha-beellama-v047-sm120-experimental-r5.tar.zst`. It removes the broken
GDN wrapper and cubins, sets `ESCHA_GDN_CHUNK_BRIDGE=0`, and retains the BM64
code-GEMM bridge, CUDA 13.0 user-space libraries, two bounded profiles,
`doctor --report`, model hashes, and source rebase patch. Exact extraction and
E3/W2 install, logits, corpus, cache/retrieval and cleanup receipts are in
PKG-04. E3/W2 KVarN prefill is respectively 92.4%/89.9% of the corrected
same-card native IQ3 benchmark; served fast8k MTP prefill is 64.8%/68.4% of
the native Escha SGLang reference. With GDN off on both cards, KVarN prefill
scales 1.247× E3 and 1.221× W2 from SM89 to SM120, versus native IQ3's
1.236×. E3 KVarN decode scales only 1.211× versus native IQ3's 1.311×,
about 7.7% below that hardware-relative benchmark. Improving E3 decode and
served prefill remains open. The r5 binary, libllama, CUDA library and selected
code-GEMM bridge are byte-identical to the r4 GDN-off four-cell bench payload;
r5 removes the invalid optional GDN path.

The r6 local review archive is
`dist/escha-beellama-v047-sm120-experimental-r6.tar.zst`. It retains r5's
byte-identical runtime, launcher, profiles and bridge, and includes the
corrected GDN-off hardware-scaling receipt, final r5 bounded-qualification
summary, validated harness and current quickstart. A fresh r6 extraction
passed hashes and E3/W2 doctor/model/launcher smokes (PKG-05). r5 remains
frozen for provenance; r6 is the review candidate. Neither is published.

### DFL-13 — forced-length code bracket, Gate B code straddles the threshold

2026-09-24. The earlier code screens were ambiguous because controls
ended early. FORCE=500 (which sends min_tokens/ignore_eos) makes every
arm emit exactly 500 tokens, removing that bias; it is a new request
contract, so the native non-speculative control was re-measured under
it. Native IQ3 non-speculative code was **85.32 tok/s**
(gdn18/native-resident-dflash-n4-force500nonspec1-bench.log), giving a
2x threshold of **170.64 tok/s**. Same-binary env-off/on, five measured
requests plus one warmup per arm, fresh process each:

| Model | Env off | MMVF5 on | Delta | Candidate / native |
| --- | ---: | ---: | ---: | ---: |
| E3 code | 157.78 (CV 1.7%) | 168.49 (CV 6.2%) | +6.79% | 1.975x |
| W2 code | 155.35 (CV 3.7%) | 169.68 (CV 2.4%) | +9.23% | 1.989x |

Both code cells now sit within about 1.3% and 0.6% of the 2x gate, but
their uncertainty spans the threshold, so **Gate B code is not
declared**. One W2 control request returned 401 tokens despite
min_tokens; that is a harness caveat, not a runtime result. Treat
168-170 tok/s as an honest interim value rather than a pass. Receipts:
gdn18/{e3,w2}-r258conv-rt2-dflash-n4-mmvf5code5{a,b}-bench.log.

### DFL-14 — narrative gate budget after the verifier win

With the DFL-12 verifier, the same-library narrative cycle is now about
22.9 ms at roughly 2.82 useful tokens, giving 123.31 tok/s. The
depth-four 2x narrative gate of 170.30 tok/s permits a whole cycle of
**at most 16.6 ms**, which is now only about 2.2 ms above the target
verification phase alone at 14.43 ms. PRF-11's phase budget put draft
at 6.48 ms, process at 0.97 ms and verify at 0.65 ms; those were
trace-perturbed, but no trace-independent measurement contradicts them
by more than a few milliseconds. Native IQ3 with the identical Q4_K_M
drafter reached only 128.58 tok/s narrative (1.51x its own 85.15
non-speculative control) while exceeding 2x on code. **Conclusion: no
remaining verifier or launch-geometry change can close the narrative 2x
cell.** Narrative needs either a large draft-phase reduction (roughly
halving it) or a change in useful tokens per cycle; code is already at
the boundary and needs far less. This supersedes the PRF-16 priority
list for narrative and makes the draft phase the only high-EV narrative
target left under the fixed depth-four, fixed-drafter and
unchanged-weights constraints.
## Release gate redefinition and matched parity matrix (REL-01)

The release gate is now a ratio against native, not an absolute number. For
BOTH W2 and E3 the candidate must reach at least 0.95 of the matched native
GGUF throughput in the SAME operating mode: ordinary decode versus native
ordinary decode, and DFlash2 versus native GGUF running the identical
DFlash2 drafter and configuration. The earlier 150 tok/s, 170 tok/s and 2x
figures are descriptive only and are withdrawn as gates. Prefill and
ordinary-decode qualification are retained; DFlash2 cannot substitute for
them. Prose and code are separate cells for each model.

### Matched inputs

One CUDA library served every arm
(gdn18/r258conv-rt2-integrated/bin/libggml-cuda.so.0.23.0, SHA-256
1444dc528cc9bae9f5a2791b39a178430619169ee7d656b14377b6809f48fa93), so a
generally applicable optimization cannot be withheld from the native
control. Every arm used the same drafter
/home/sean/models/qwen3.8-27b-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf,
depth four, draft KV q4_0, target KVarN3/2, -c 8192 -np 1 -b 2048 -ub 512,
FlashAttention on, 8K one slot, FORCE=500 so every arm emits exactly 500
tokens, one warmup plus five measured requests. Native arms keep speculation
disabled for the ordinary cells and use ESCHA_DFLASH_NATIVE_SPEC=1 only for
the DFlash2 cells. E3/W2 GGUF hashes are unchanged.

### DFlash2 depth-four, five measured requests per arm, one fresh process each

| Model | Prose tok/s | Ratio | Code tok/s | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Native IQ3 GGUF | 123.67 | 1.000 | 170.23 | 1.000 |
| E3 | **127.08** | **1.028** | **179.84** | **1.057** |
| W2 | **124.72** | **1.009** | **178.84** | **1.051** |

All four DFlash2 cells pass, and Escha is faster than the matched native
control in every cell. Receipts: gdn18/*-dflash-n4-m2{narr,code}*-{bench,server}.log.

The E3 prose cell was extended before being accepted because its first
three-sample round measured 0.951. Three interleaved fresh starts of five
samples each gave native 126.61 / 127.78 / 123.98 and E3 122.43 / 123.50 /
124.22 tok/s, pooled 126.12 versus 123.38, ratio **0.978**. Receipts:
gdn18/*dflash-n4-narr5{a,b,c}-*-bench.log.

### Ordinary prefill and decode, five repetitions, KVarN3/2

| Cell | Native | E3 | E3 ratio | W2 | W2 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill p2048 UB2048 | 3024.81 | 3042.30 | **1.006** | 3145.94 | **1.040** |
| Decode p0 n256 UB512 | 90.53 | 86.42 | **0.955** | 86.11 | **0.951** |

Prefill passes with margin. Decode passes only narrowly: an interleaved
three-round bracket gave native 91.23 / 89.60 / 90.77 and per-round ratios
of 0.9485 / 0.9651 / 0.9501 for E3 and 0.9466 / 0.9602 / 0.9467 for W2.
The pooled ratios clear 0.95, but individual rounds fall below it, so the
ordinary-decode cell is **not yet a durable pass**; do not certify it on the
pooled mean alone. Receipts: gdn18/raw-rel1-*-{prefill,decode}/ and
gdn18/raw-dec{a,b,c}-*/.

### Runtime integration of the verification dispatch

The M=5 F16 dispatch that DFL-12 introduced behind ESCHA_SM120_M5_F16_MMVF is
now unconditional for cc >= 1200 with five input rows and at most 128 output
rows, so the shipped runtime needs no hidden switch. Patch
gdn18/m5-f16-mmvf-integrated.patch SHA-256
efc6d81f24fb03825972d2774dfc804ea891714da3580cb8ef322abd04d60dc3.
Rebuild is deterministic: rebuilding the same source twice produced the
identical library hash
2207dd98e31449a452781f274b94d82d4330233c6b429fb7e58d114558744148, and the
integrated library is that build plus the single mmvf.cu change. It
reproduces the env-gated logits exactly (five rows, same argmax, 20/20 top-20,
worst relative RMS 2.94e-4) and holds the verifier win with no environment
variable set: E3 M=5 14.5103 ms, W2 14.4780 ms. Receipts:
gdn18/intg-{e3,w2}-{guard,bench,route-ncu*}.

### REL-02 — ordinary decode extended to five interleaved rounds

Three rounds left the decode cell ambiguous, so two more interleaved rounds
were added rather than certifying on the earlier pooled mean. Each round runs
native, E3 and W2 back to back with five llama-bench repetitions, p0/n256,
UB512, KVarN3/2, t8, FA on, GPU offload 99, one shared CUDA library.

| Model | Round a | b | c | d | e | Pooled | Std | Pooled ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native IQ3 | 91.23 | 89.60 | 90.77 | 89.83 | 89.54 | 90.194 | 0.761 | 1.0000 |
| E3 | 86.53 | 86.48 | 86.24 | 87.08 | 86.32 | 86.530 | 0.328 | **0.9594** |
| W2 | 86.35 | 86.04 | 85.93 | 85.94 | 86.52 | 86.157 | 0.266 | **0.9552** |

Both models clear the 0.95 requirement, with a 4.06% penalty for E3 and
4.48% for W2. The Escha measurements are the stable ones (std 0.33 and 0.27);
the native control is the noisy side (std 0.76), so the per-round ratios range
0.9485-0.9694 for E3 and 0.9466-0.9663 for W2. Record this as a pass on the
pooled interleaved means with about half a point to a point of margin and
state the spread; do not describe any single round as the result. Receipts:
gdn18/raw-dec{a,b,c,d,e}-*/ and gdn18/dec5.log.

### REL-03 — where the remaining ordinary-decode penalty lives

An M=1 NVTX-bounded Nsight Compute range on the integrated runtime
(m1-nvtx-range.cpp/.so, one tagged single-token decode per model) attributes
the whole gap to the coded-projection finalization. Profiled sums are
instrumentation-only and exceed real wall time; they are used for attribution.

| Target | Launches in range | Profiled sum | Largest families |
| --- | ---: | ---: | --- |
| E3 | 1380 | 15.71 ms | escham_gemv_bw 400x 10.335 ms; finalize family 400x 2.428 ms; LM head 1x 0.728 ms |
| W2 | 1380 | 16.06 ms | escham_gemv_bw 400x 10.331 ms; finalize family 400x 2.512 ms; I8 head 1x 0.945 ms |
| Native IQ3 | 1998 | 13.62 ms | mul_mat_vec_q 463x 9.080 ms; quantize_q8_1 463x 1.070 ms |

Escha pays 400 extra launches per token because every coded projection
finalizes its split-K partials in a separate kernel, while native's fused
quantized matvec needs no reduction launch. That finalize family is 2.43-2.51
ms profiled and is essentially the entire remaining penalty. The structural
fix is to fold the ordered split sum, the Sylvester-Hadamard stage and the rout
scale into the last split CTA, for which an older lab prototype exists at
/home/sean/kernel-lab5090/lab/escha_last_split_finalize_r215. That is a
larger, higher-risk kernel change than the dispatch work so far, so it is
carried as the improvement track while the release proceeds on the measured
pass. Receipts: gdn18/m1-{e3,w2,native}-route-ncu.csv.

