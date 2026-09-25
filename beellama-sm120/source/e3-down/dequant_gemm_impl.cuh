// Isolated SM120 prefill screen. Only called for K3 17408->5120 and M>=1536.
// This decodes the exact packed codebook into transient F16 scratch, then uses
// FP32-accumulation cuBLAS and the same normalized 128-point output transform.

static __device__ __forceinline__ int deq_dep_pi(int r) {
    return (r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 3);
}

static __device__ __forceinline__ half deq_codebook_h(uint32_t idx) {
    uint32_t x = idx * 0xcbac1fedu;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    __half2 h;
    memcpy(&h, &x, sizeof(h));
    return __hadd(__low2half(h), __high2half(h));
}

static __global__ void deq_down_kernel(const uint16_t * __restrict__ code,
                                      half * __restrict__ weight) {
    constexpr int IC = 17408, OC = 5120, K = 3;
    constexpr int NW = 8 * K, NB = 32 * NW;
    constexpr int NTILES = (IC / 16) * (OC / 16);
    __shared__ uint32_t pay[NW];
    for (int tile = blockIdx.x; tile < NTILES; tile += gridDim.x) {
        const int ti = tile / (OC / 16);
        const int tj = tile % (OC / 16);
        if (threadIdx.x < NW) {
            pay[threadIdx.x] = reinterpret_cast<const uint32_t *>(code + size_t(tile) * 16 * K)[threadIdx.x];
        }
        __syncthreads();
        const int c = threadIdx.x / 16;
        const int r = threadIdx.x % 16;
        int sp = ((32 - K) - K * (deq_dep_pi(r) + 32 * c + 4 * (c >> 3))) % NB;
        if (sp < 0) sp += NB;
        const int g0 = sp >> 5;
        const int w0 = g0 ? NW - g0 : 0;
        const int w1 = w0 ? w0 - 1 : NW - 1;
        const uint32_t idx = __funnelshift_r(pay[w0], pay[w1], sp & 31) & 0xffffu;
        weight[size_t(tj * 16 + c) * IC + ti * 16 + r] = deq_codebook_h(idx);
        __syncthreads();
    }
}

static __global__ void deq_down_finalize(const float * __restrict__ tmp,
                                         const half * __restrict__ rout,
                                         half * __restrict__ out) {
    constexpr int OC = 5120;
    __shared__ float v[128];
    const int c = blockIdx.x * 128 + threadIdx.x;
    const int m = blockIdx.y;
    v[threadIdx.x] = tmp[size_t(m) * OC + c];
    __syncthreads();
    for (int len = 1; len < 128; len <<= 1) {
        if (threadIdx.x < 64) {
            const int j = threadIdx.x;
            const int i = (j / len) * (2 * len) + (j % len);
            const float a0 = v[i], a1 = v[i + len];
            v[i] = a0 + a1;
            v[i + len] = a0 - a1;
        }
        __syncthreads();
    }
    out[size_t(m) * OC + c] = __float2half_rn(v[threadIdx.x] * rsqrtf(128.0f) * __half2float(rout[c]));
}

struct deq_scratch {
    half * weight = nullptr;
    float * tmp = nullptr;
    cublasHandle_t handle = nullptr;
};

std::unordered_map<input_buffer_key, deq_scratch, input_buffer_key_hash> g_deq_scratch;

static deq_scratch * deq_scratch_for(int device, cudaStream_t stream) {
    const input_buffer_key key { device, reinterpret_cast<uintptr_t>(stream), 0 };
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_deq_scratch.find(key);
    if (it != g_deq_scratch.end()) return &it->second;
    if (cudaSetDevice(device) != cudaSuccess) {
        g_error = "dequant GEMM cudaSetDevice failed";
        return nullptr;
    }
    deq_scratch scratch;
    if (cudaMalloc(&scratch.weight, size_t(17408) * 5120 * sizeof(half)) != cudaSuccess ||
        cudaMalloc(&scratch.tmp, size_t(2048) * 5120 * sizeof(float)) != cudaSuccess ||
        cublasCreate(&scratch.handle) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(scratch.handle, stream) != CUBLAS_STATUS_SUCCESS) {
        if (scratch.handle) cublasDestroy(scratch.handle);
        if (scratch.tmp) cudaFree(scratch.tmp);
        if (scratch.weight) cudaFree(scratch.weight);
        g_error = "dequant GEMM scratch allocation failed";
        return nullptr;
    }
    return &g_deq_scratch.emplace(key, scratch).first->second;
}

static int deq_down_launch(cudaStream_t stream, int device, void * dst_f16,
                           const void * x_f16, const int16_t * code,
                           const void * rout_f16, int M) {
    deq_scratch * scratch = deq_scratch_for(device, stream);
    if (!scratch) return -1;
    deq_down_kernel<<<16384, 256, 0, stream>>>(
        reinterpret_cast<const uint16_t *>(code), scratch->weight);
    if (cudaGetLastError() != cudaSuccess) {
        g_error = "dequant GEMM decode kernel failed";
        return -1;
    }
    const float alpha = 1.0f, beta = 0.0f;
    if (cublasGemmEx(scratch->handle, CUBLAS_OP_T, CUBLAS_OP_N,
        5120, M, 17408, &alpha, scratch->weight, CUDA_R_16F, 17408,
        x_f16, CUDA_R_16F, 17408, &beta, scratch->tmp, CUDA_R_32F, 5120,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
        g_error = "dequant GEMM cuBLAS failed";
        return -1;
    }
    deq_down_finalize<<<dim3(5120 / 128, M), 128, 0, stream>>>(
        scratch->tmp, reinterpret_cast<const half *>(rout_f16),
        reinterpret_cast<half *>(dst_f16));
    if (cudaGetLastError() != cudaSuccess) {
        g_error = "dequant GEMM finalizer failed";
        return -1;
    }
    return 0;
}
