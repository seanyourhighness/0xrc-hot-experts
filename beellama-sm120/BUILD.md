# Rebuild this SM120 runtime (r9)

Source identity: BeeLLaMA Preview commit
1156b183630eb8bd76ff931348d0b1a258ae171b plus the Escha patch in
source/escha-v047-sm120-r9-source.patch, SHA-256
9ab0b03242a99718edbbd6b715ca4c5b7e173937c6e702f3f1ab9e4b407b4d16. That patch
changes 12 files and applies cleanly to a fresh checkout of the pinned commit;
it is the authoritative source for this release, and no public fork commit or
tag exists yet.

Tested build host: Ubuntu 24.04, GCC 13.3.0, CUDA nvcc 13.0.88, CMake Release,
SM120a, CUDA FlashAttention and KVarN enabled, native CPU code generation
disabled, prebuilt Web UI disabled. Tested GPU: RTX 5090 on driver 616.56.
Paths below are examples; choose paths available on your build host.

    git -C /path/to/BeeLLaMA checkout 1156b183630eb8bd76ff931348d0b1a258ae171b
    cd /path/to/BeeLLaMA
    git apply --check /path/to/source/escha-v047-sm120-r9-source.patch
    git apply /path/to/source/escha-v047-sm120-r9-source.patch

    cmake -S . -B /path/to/build-sm120a -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/path/to/cuda-13.0/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CUDA=ON \
      -DGGML_CUDA_FA=ON -DGGML_CUDA_KVARN=ON -DGGML_NATIVE=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN'
    cmake --build /path/to/build-sm120a -j16 \
      --target llama-server llama-cli llama-bench
    readelf -d /path/to/build-sm120a/bin/llama-server | grep RUNPATH
    /path/to/cuda-13.0/bin/cuobjdump -lelf /path/to/build-sm120a/bin/libggml-cuda.so | grep sm_120a

The RUNPATH check must print $ORIGIN only, and the cuobjdump check must list
sm_120a cubins. The qualified runtime hashes are libllama.so.0.4.7
d3df7954c68202ab1062228b77e469faa6254e347df4d5c3560762b5214475e0,
libggml-cuda.so.0.23.0
db032fec0f0d3b29306401d368ca8f5a5ab2820c06293744c24e0a0c2a0cecbe and
llama-server 67992606909b6b13197265988680173256f0fb143cb6fe2baa46d7dbd5d489ab.
Builds of one source revision on this host were byte-reproducible; byte
identical output from a different toolchain or host is not assumed, so compare
against MANIFEST.json and rerun the parity and correctness checks on your own
RTX 5090 before claiming equivalence.

## Kernel payload

The release needs the bundled CUDA payload at runtime. These files ship in the
archive and are hashed in MANIFEST.json:

| File | SHA-256 |
| --- | --- |
| bridge/libescha_official_bridge_cuda_sm120.so | 59d63f96a9452e7a4bafebc67fbd6426fc65826808b00a062787e1a482ea7ce4 |
| bridge/libescha_gdn_chunk_bridge_sm120_sglangprefix2045.so | d09db62294d610dc6a7129e9ff3922f35270909c122e62c9cf76fa2bee0d0624 |
| bridge/libescha_gdn_chunk_k1_full2048.so | 10e0133861e469f27a55e31dea466689e4ac971efeda054b1fc9d98fdee7d69d |
| bridge/code-gemm-sm120.cubin | 6cdbf1ab440958d570ef0b3425df5fcea14c40f48c02d5774da983514d676d8b |
| bridge/f32_input.sm120.cubin | 04692f328b386967c02b68a2061e6ec846675f82e516978a7f1ab9555c99e39b |
| bridge-e3-down/libescha_official_bridge_cuda_sm120.so | 920e509d094552b2cd70fe8ba5a14b05aad2a6af198f1facfaf489489acae670 |

The official bridge wrappers are built from the CUDA sources under source/ with
the CUDA 13.0 toolchain:

    nvcc -shared -Xcompiler=-fPIC -O3 -std=c++17 -arch=sm_120a \
      official_bridge_cuda.cu -o libescha_official_bridge_cuda_sm120.so -lcuda

source/bridge-candidate holds the selected wrapper source and
source/bridge-control the same-source control; source/bridge-down-bm64.patch
selects the BM64/BK3 instantiation used by the E3 down projection;
source/gdn2045 holds the 2045-row chunk bridge patch and source;
source/gdn-k1/gdn_bridge_full2048.cu is the full-2048 K=1 GDN bridge source.

## Which GDN route applies to which target

ESCHA_GDN_CHUNK_BRIDGE and ESCHA_GDN_CHUNK_K1 are Escha-only prefill routes and
the launcher enables them for e3 and w2. They stay off for the stock GGUF
target, whose logits fail that bridge's numerical guard; every other SM120
route is enabled for all three targets. Do not force the K=1 bridge on for a
stock model.

## Host ABI

This payload requires glibc 2.38 and a libstdc++ exporting GLIBCXX_3.4.32.
CUDA 13.0 libcudart, libcublas and libcublasLt are bundled; libcuda.so.1 comes
from the host driver. Distributions older than Ubuntu 24.04 need a separate
build.

