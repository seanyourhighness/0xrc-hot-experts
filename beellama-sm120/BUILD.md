# Rebuild this SM120 runtime

Source identity: BeeLLaMA Preview commit
`1156b183630eb8bd76ff931348d0b1a258ae171b` plus the Escha patch in
`source/escha-v047-sm120-rebase.patch` (SHA-256
`886c10a96d027eab0126c43cfd0e5befcf4b8fb14a1950af389b78de0ee5cfa9`).
The patch applied cleanly to a detached checkout of that exact Preview commit;
the 56 changed runtime source/text files matched the working candidate by hash.
The converter's large metadata template and two `dep_k*.npy` tables are separate
inputs listed in `source/rebase-source-manifest.json`; they are not runtime
build inputs. No public source commit or immutable tag exists yet.

The tested build ran on z840 with Ubuntu GCC 13.3.0, CUDA nvcc 13.0.88,
CMake Release, SM120a, CUDA FlashAttention and KVarN enabled, and native CPU
code generation disabled. The tested RTX 5090 used driver 616.56. Paths below
are examples; choose paths available on the build host.

```bash
git -C /path/to/BeeLLaMA checkout 1156b183630eb8bd76ff931348d0b1a258ae171b
cd /path/to/BeeLLaMA
git apply --check /path/to/source/escha-v047-sm120-rebase.patch
git apply /path/to/source/escha-v047-sm120-rebase.patch

cmake -S . -B /path/to/build-sm120a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_KVARN=ON -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  '-DCMAKE_INSTALL_RPATH=$ORIGIN'
cmake --build /path/to/build-sm120a -j20 \
  --target llama-server llama-cli llama-bench
readelf -d /path/to/build-sm120a/bin/llama-server | grep RUNPATH
cuobjdump -lelf /path/to/build-sm120a/bin/libggml-cuda.so | grep sm_120a
```

For the r7 code-GEMM bridge, the bundle includes the exact control and selected CUDA
source under `source/bridge-control/` and `source/bridge-candidate/`, plus the
one-change patch `source/bridge-down-bm64.patch`. The patch selects the
vendor cubin's BM64/BK3 instantiation for the K3 `17408→5120` prefill down
projection; no GGUF transcode or model change is involved. Build the selected
wrapper with the matching header and CUDA 13.0 toolchain:

```bash
cd /path/to/extracted/source/bridge-candidate
/path/to/cuda-13.0/bin/nvcc -shared -Xcompiler=-fPIC -O3 -std=c++17 \
  -arch=sm_120a official_bridge_cuda.cu \
  -o libescha_official_bridge_cuda_sm120.so -lcuda
sha256sum libescha_official_bridge_cuda_sm120.so
```

The qualified selected wrapper has SHA-256
`59d63f96a9452e7a4bafebc67fbd6426fc65826808b00a062787e1a482ea7ce4`.
The `source/bridge-probe-manifest.json` records both source hashes, the
same-source control hash and build command. The precompiled cubin in `bridge/`
is required at runtime; its SHA-256 is pinned in `MANIFEST.json`.
The K=1 GDN chunk bridge from r4 produced NaN logits on a full 2K batch.
It is disabled in `profiles/escha-sm120-bridge.env`, and its wrapper and cubins
are excluded from r7. Rebuilding this runtime alone does not requalify that
optional GDN path.

The original runtime was built on z840 at
`/mnt/storage/ai-builds/escha-v047-sm120a` and relinked with `$ORIGIN` before
packaging. The bundle includes CUDA 13.0 `libcudart`, `libcublas`, and
`libcublasLt`; `libcuda.so.1` comes from the host driver. The bundle references
glibc 2.38 and GLIBCXX_3.4.32 at most; those are current host
ABI requirements. A broad Linux release needs an older glibc/GCC build host
and a fresh RTX 5090 qualification of its resulting binaries. The source build is
reproducible by command and hash-identified inputs; byte-identical output on a
different toolchain or host is not assumed. Use `MANIFEST.json` to compare the
actual binary and bridge hashes, and rerun the included performance/functional
receipts on the target RTX 5090 before claiming equivalence.
