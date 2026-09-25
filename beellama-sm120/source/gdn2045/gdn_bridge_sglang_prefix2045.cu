#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int T = 2045;
constexpr int H = 48;
constexpr int HG = 16;
constexpr int D = 128;
constexpr int NT = (T + 63) / 64;
constexpr int N16 = (T + 15) / 16;

thread_local std::string last_error;

void cuda_ok(cudaError_t status, const char * what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

void cu_ok(CUresult status, const char * what) {
    if (status == CUDA_SUCCESS) return;
    const char * text = nullptr;
    cuGetErrorString(status, &text);
    throw std::runtime_error(std::string(what) + ": " + (text ? text : "unknown driver error"));
}

__global__ void cast_grouped_qk(
        const float * q, const float * k, __nv_bfloat16 * qg, __nv_bfloat16 * kg,
        int64_t token_stride) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) T * HG * D;
    if (i >= n) return;
    const int d = i % D;
    const size_t x = i / D;
    const int hg = x % HG;
    const int t = x / HG;
    const int source_head = token_stride == HG*D ? hg : hg*3;
    const size_t src = (size_t) t * token_stride + source_head * D + d;
    qg[i] = __float2bfloat16_rn(q[src]);
    kg[i] = __float2bfloat16_rn(k[src]);
}

__global__ void cast_compact_v(const float * v, __nv_bfloat16 * compact, int64_t token_stride) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) T * H * D;
    if (i >= n) return;
    const int d = i % D;
    const size_t x = i / D;
    const int h = x % H;
    const int t = x / H;
    compact[i] = __float2bfloat16_rn(v[(size_t) t * token_stride + h * D + d]);
}

__global__ void state_to_fla(const float * physical, float * fla) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) H * D * D;
    if (i >= n) return;
    const int col = i % D;
    const size_t x = i / D;
    const int row = x % D;
    const int head = x / D;
    fla[i] = physical[((size_t) head * D + col) * D + row];
}

__global__ void state_from_fla(const float * fla, float * physical) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) H * D * D;
    if (i >= n) return;
    const int col = i % D;
    const size_t x = i / D;
    const int row = x % D;
    const int head = x / D;
    physical[((size_t) head * D + col) * D + row] = fla[i];
}

__global__ void output_to_f32(const __nv_bfloat16 * src, float * dst) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) T * H * D;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

// Independent KKT contract probe. This writes A[t,h,j] directly from the
// rotated BF16 key, float beta and local cumulative gate. The cached KKT cubin
// remains the default; set ESCHA_GDN_CUSTOM_KKT=1 only in this isolated probe.
__global__ void kkt_reference_cuda(const __nv_bfloat16 * key,
                                   const float * beta, const float * gsum,
                                   float * A) {
    __shared__ __nv_bfloat16 sk[64 * D];
    __shared__ float sbeta[64], sg[64];
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int grouped_head = head / (H / HG);
    const int token0 = chunk * 64;
    for (int i = threadIdx.x; i < 64 * D; i += blockDim.x) {
        const int row = i / D;
        const int col = i % D;
        sk[i] = key[((size_t) (token0 + row) * HG + grouped_head) * D + col];
    }
    if (threadIdx.x < 64) {
        sbeta[threadIdx.x] = beta[(size_t) (token0 + threadIdx.x) * H + head];
        sg[threadIdx.x] = gsum[(size_t) (token0 + threadIdx.x) * H + head];
    }
    __syncthreads();
    for (int index = threadIdx.x; index < 64 * 64; index += blockDim.x) {
        const int row = index / 64;
        const int col = index % 64;
        float value = 0.0f;
        if (row > col) {
            float dot = 0.0f;
#pragma unroll 4
            for (int d = 0; d < D; ++d) {
                dot = fmaf(__bfloat162float(sk[row * D + d]),
                           __bfloat162float(sk[col * D + d]), dot);
            }
            value = dot * sbeta[row] * expf(sg[row] - sg[col]);
        }
        A[((size_t) (token0 + row) * H + head) * 64 + col] = value;
    }
}

// Direct lower-triangular solve for (I + A)^-1. Each lane owns one output
// column, so the recurrence for that column is sequential and independent.
__global__ void inverse64_reference_cuda(const float * A,
                                          __nv_bfloat16 * inverse) {
    __shared__ float block_A[64 * 64];
    __shared__ float block_inv[64 * 64];
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int col = threadIdx.x;
    const int token0 = chunk * 64;
    for (int row = 0; row < 64; ++row) {
        block_A[row * 64 + col] = A[((size_t) (token0 + row) * H + head) * 64 + col];
    }
    __syncthreads();
    for (int row = 0; row < 64; ++row) {
        float value = row == col ? 1.0f : 0.0f;
        for (int j = 0; j < row; ++j) {
            value -= block_A[row * 64 + j] * block_inv[j * 64 + col];
        }
        block_inv[row * 64 + col] = value;
    }
    for (int row = 0; row < 64; ++row) {
        inverse[((size_t) (token0 + row) * H + head) * 64 + col] =
            __float2bfloat16_rn(block_inv[row * 64 + col]);
    }
}

struct Kernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
};

struct Workspace {
    int device = -1;
    bool matched_sglang = false;
    Kernel cumsum, kkt, solve, merge, recompute, state, output;
    __nv_bfloat16 * q = nullptr;
    __nv_bfloat16 * k = nullptr;
    __nv_bfloat16 * v = nullptr;
    __nv_bfloat16 * Ai = nullptr;
    __nv_bfloat16 * w = nullptr;
    __nv_bfloat16 * u = nullptr;
    __nv_bfloat16 * h = nullptr;
    __nv_bfloat16 * vnew = nullptr;
    __nv_bfloat16 * out = nullptr;
    float * gsum = nullptr;
    float * A = nullptr;
    float * Ad = nullptr;
    float * state_fla = nullptr;
    float * state_final = nullptr;
    int * cu_seqlens = nullptr;
    int * chunk_indices64 = nullptr;
    int * chunk_indices16 = nullptr;
    int * initial_state_indices = nullptr;
    int64_t * chunk_offsets = nullptr;
};

Workspace workspace;
std::once_flag init_once;

Kernel load_kernel(const std::string & path, const char * name, int shared) {
    Kernel k;
    cu_ok(cuModuleLoad(&k.module, path.c_str()), "cuModuleLoad");
    cu_ok(cuModuleGetFunction(&k.function, k.module, name), "cuModuleGetFunction");
    if (shared > 48 * 1024) {
        cu_ok(cuFuncSetAttribute(k.function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared),
              "cuFuncSetAttribute");
    }
    return k;
}

void allocate(void ** ptr, size_t bytes) { cuda_ok(cudaMalloc(ptr, bytes), "cudaMalloc GDN workspace"); }

void initialize(int device) {
    cuda_ok(cudaSetDevice(device), "cudaSetDevice");
    cu_ok(cuInit(0), "cuInit");
    const char * matched_root = std::getenv("ESCHA_GDN_SGLANG_CUBIN_ROOT");
    if (matched_root == nullptr || matched_root[0] == '\0') {
        throw std::runtime_error("ESCHA_GDN_SGLANG_CUBIN_ROOT is required");
    }
    workspace.matched_sglang = true;
    const std::string root = matched_root;
    workspace.device = device;
    if (workspace.matched_sglang) {
        workspace.cumsum = load_kernel(root + "/cumsum.cubin", "chunk_local_cumsum_scalar_kernel", 8);
        workspace.kkt = load_kernel(root + "/kkt.cubin", "chunk_scaled_dot_kkt_fwd_kernel", 16384);
        workspace.solve = load_kernel(root + "/solve.cubin", "solve_tril_16x16_kernel", 0);
        workspace.merge = load_kernel(root + "/merge.cubin", "merge_16x16_to_64x64_inverse_kernel", 10240);
        workspace.recompute = load_kernel(root + "/recompute.cubin", "recompute_w_u_fwd_kernel", 28672);
        workspace.state = load_kernel(root + "/state.cubin", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", 49412);
        workspace.output = load_kernel(root + "/output.cubin", "chunk_fwd_kernel_o", 32768);
    } else {
        workspace.cumsum = load_kernel(root + "/6QYCAHTXXKFFSHAC3YVR42Y3GCI2DG2BTQWNRZEWAUAY6QRWS7IQ/chunk_local_cumsum_scalar_kernel.cubin", "chunk_local_cumsum_scalar_kernel", 8);
        workspace.kkt = load_kernel(root + "/KLQYP7TG6PVPLHS3OXWENWOZAJBOOJLQE6ZJJHLJMWW6TTPFID6Q/chunk_scaled_dot_kkt_fwd_kernel.cubin", "chunk_scaled_dot_kkt_fwd_kernel", 16384);
        workspace.solve = load_kernel(root + "/GG6MUT3BZU4WCQJRMIEBXKRTQ4YCHOVKMGNZLRUMSLV5JCH5FKEA/solve_tril_16x16_kernel.cubin", "solve_tril_16x16_kernel", 0);
        workspace.merge = load_kernel(root + "/BVFQ2K6N6FY67PGQ774H3GBHCXWNBHDX6O3Z7VCZPKCMWRLTH5OA/merge_16x16_to_64x64_inverse_kernel.cubin", "merge_16x16_to_64x64_inverse_kernel", 10240);
        workspace.recompute = load_kernel(root + "/5A2PDUPQAMJVG4FCWPV5LDJ4TSHV5KRKVBE64KSLU4DNPFU2AAFQ/recompute_w_u_fwd_kernel.cubin", "recompute_w_u_fwd_kernel", 32768);
        workspace.state = load_kernel(root + "/B3JF6EJLT6K7RTADX6SUTV6E4QVUVKO5LLQKWEH7OUG3FEHDBNBA/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.cubin", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", 90632);
        workspace.output = load_kernel(root + "/GRSIIZIDJF7Q7GJFOMIOVY6DS6IDRQEQJDEGXO47VJXO3ALVSRKQ/chunk_fwd_kernel_o.cubin", "chunk_fwd_kernel_o", 32768);
    }
    allocate((void **) &workspace.q, (size_t) T*HG*D*2);
    allocate((void **) &workspace.k, (size_t) T*HG*D*2);
    allocate((void **) &workspace.v, (size_t) T*H*D*2);
    allocate((void **) &workspace.gsum, (size_t) T*H*4);
    allocate((void **) &workspace.A, (size_t) T*H*64*4);
    allocate((void **) &workspace.Ad, (size_t) T*H*16*4);
    allocate((void **) &workspace.Ai, (size_t) T*H*64*2);
    allocate((void **) &workspace.w, (size_t) T*H*D*2);
    allocate((void **) &workspace.u, (size_t) T*H*D*2);
    allocate((void **) &workspace.h, (size_t) NT*H*D*D*2);
    allocate((void **) &workspace.vnew, (size_t) T*H*D*2);
    allocate((void **) &workspace.out, (size_t) T*H*D*2);
    allocate((void **) &workspace.state_fla, (size_t) H*D*D*4);
    allocate((void **) &workspace.state_final, (size_t) H*D*D*4);
    allocate((void **) &workspace.cu_seqlens, 2*sizeof(int));
    if (workspace.matched_sglang) {
        allocate((void **) &workspace.chunk_indices64, 2*NT*sizeof(int));
        allocate((void **) &workspace.chunk_indices16, 2*N16*sizeof(int));
        allocate((void **) &workspace.initial_state_indices, sizeof(int));
        int indices64[2*NT], indices16[2*N16];
        for (int i=0;i<NT;i++) { indices64[2*i]=0; indices64[2*i+1]=i; }
        for (int i=0;i<N16;i++) { indices16[2*i]=0; indices16[2*i+1]=i; }
        const int initial_index=0;
        cuda_ok(cudaMemcpy(workspace.chunk_indices64,indices64,sizeof(indices64),cudaMemcpyHostToDevice),"chunk indices64");
        cuda_ok(cudaMemcpy(workspace.chunk_indices16,indices16,sizeof(indices16),cudaMemcpyHostToDevice),"chunk indices16");
        cuda_ok(cudaMemcpy(workspace.initial_state_indices,&initial_index,sizeof(initial_index),cudaMemcpyHostToDevice),"state index");
    }
    allocate((void **) &workspace.chunk_offsets, 2*sizeof(int64_t));
    const int seqlens[2] = {0, T};
    const int64_t offsets[2] = {0, NT};
    cuda_ok(cudaMemcpy(workspace.cu_seqlens,seqlens,sizeof(seqlens),cudaMemcpyHostToDevice),"state seqlens");
    cuda_ok(cudaMemcpy(workspace.chunk_offsets,offsets,sizeof(offsets),cudaMemcpyHostToDevice),"state chunk offsets");
}

void launch(CUfunction fn, dim3 grid, int warps, int shared, void ** args, cudaStream_t stream) {
    cu_ok(cuLaunchKernel(fn, grid.x, grid.y, grid.z, warps*32, 1, 1, shared,
                        reinterpret_cast<CUstream>(stream), args, nullptr), "cuLaunchKernel");
}

void inspect_f32(const char * name, const float * ptr, size_t n, cudaStream_t stream) {
    cuda_ok(cudaStreamSynchronize(stream), "GDN debug sync");
    std::vector<float> values(n);
    cuda_ok(cudaMemcpy(values.data(),ptr,n*sizeof(float),cudaMemcpyDeviceToHost),"GDN debug copy");
    size_t finite=0,nan=0,zero=0;
    float maxabs=0,minval=INFINITY,maxval=-INFINITY;
    size_t max_index=0;
    for (size_t i=0;i<n;i++) {
        float x=values[i];
        finite+=std::isfinite(x); nan+=std::isnan(x); zero+=(x==0);
        if (std::isfinite(x)) {
            if (std::abs(x)>maxabs) { maxabs=std::abs(x); max_index=i; }
            minval=std::min(minval,x); maxval=std::max(maxval,x);
        }
    }
    std::fprintf(stderr,"GDN_INSPECT %s n=%zu finite=%zu nan=%zu zero=%zu min=%g max=%g maxabs=%g max_index=%zu first=%g\n",
                 name,n,finite,nan,zero,minval,maxval,maxabs,max_index,values.empty()?0:values[0]);
    if (std::strcmp(name,"gsum")==0) {
        for (int t=0;t<16;t++) std::fprintf(stderr,"GDN_INSPECT gsum_head11 token=%d value=%g\n",t,values[(size_t)t*H+11]);
        for (int chunk=0;chunk<NT;chunk++) {
            float largest_drop=0; int worst_h=0;
            for (int h=0;h<H;h++) {
                float start=values[(size_t)(chunk*64)*H+h];
                float end=values[(size_t)(chunk*64+63)*H+h];
                if (start-end>largest_drop) { largest_drop=start-end; worst_h=h; }
            }
            std::fprintf(stderr,"GDN_INSPECT gsum_chunk chunk=%d max_drop=%g head=%d\n",chunk,largest_drop,worst_h);
        }
    }
    if (std::strcmp(name,"A")==0) {
        for (int col=0;col<16;col++) {
            size_t idx=((size_t)13*H+11)*64+col;
            std::fprintf(stderr,"GDN_INSPECT A_token13_head11 col=%d value=%g\n",col,values[idx]);
        }
    }
}

void inspect_bf16(const char * name, const __nv_bfloat16 * ptr, size_t n, cudaStream_t stream) {
    cuda_ok(cudaStreamSynchronize(stream), "GDN debug sync");
    std::vector<uint16_t> values(n);
    cuda_ok(cudaMemcpy(values.data(),ptr,n*sizeof(uint16_t),cudaMemcpyDeviceToHost),"GDN debug copy");
    size_t finite=0,nan=0,zero=0;
    float maxabs=0,minval=INFINITY,maxval=-INFINITY;
    for (uint16_t x:values) {
        const bool is_finite=(x&0x7f80)!=0x7f80;
        finite+=is_finite; nan+=(!is_finite && (x&0x7f)); zero+=((x&0x7fff)==0);
        if (is_finite) {
            uint32_t bits=(uint32_t)x<<16;
            float f;
            std::memcpy(&f,&bits,sizeof(f));
            maxabs=std::max(maxabs,std::abs(f));
            minval=std::min(minval,f); maxval=std::max(maxval,f);
        }
    }
    std::fprintf(stderr,"GDN_INSPECT %s n=%zu finite=%zu nan=%zu zero=%zu min=%g max=%g maxabs=%g first=0x%x\n",
                 name,n,finite,nan,zero,minval,maxval,maxabs,values.empty()?0:values[0]);
}

} // namespace

extern "C" int escha_gdn_chunk_prefill(
        cudaStream_t stream, int device,
        const float * q32, const float * k32, const float * v32,
        const float * g32, const float * beta32, const float * state_in,
        float * dst32, float * state_out,
        int64_t S_v, int64_t heads, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sv1, int64_t sv2,
        int64_t sb1, int64_t sb2, float scale) {
    try {
        last_error.clear();
        if (q32 == nullptr || k32 == nullptr || v32 == nullptr || g32 == nullptr ||
            beta32 == nullptr || state_in == nullptr || dst32 == nullptr || state_out == nullptr ||
            S_v != D || heads != H || n_tokens != T || n_seqs != 1 ||
            // BeeLlama's ggml tensors are laid out as [D,H,T,B]: the
            // element stride is one and the token stride is H*D.  The
            // earlier standalone probe accidentally required D for the
            // element stride, which could never match the runtime tensor.
            // sq1/sv1 are head strides in float elements, not scalar strides.
            sq1 != D || (sq2 != HG*D && sq2 != H*D) || sv1 != D || sv2 < H*D ||
            sb1 != 1 || sb2 != H || std::abs(scale - 1.0f/std::sqrt((float)D)) > 1e-6f) {
            throw std::runtime_error("unsupported GDN chunk shape or stride");
        }
        std::call_once(init_once, [device] { initialize(device); });
        static std::atomic<int> debug_calls{0};
        const bool inspect = std::getenv("ESCHA_GDN_INSPECT") != nullptr &&
                             debug_calls.fetch_add(1) == 0;
        if (workspace.device != device) throw std::runtime_error("GDN bridge initialized on another device");
        cuda_ok(cudaSetDevice(device), "cudaSetDevice dispatch");
        constexpr int threads = 256;
        cast_grouped_qk<<<((size_t)T*HG*D+threads-1)/threads,threads,0,stream>>>(
            q32,k32,workspace.q,workspace.k,sq2);
        cast_compact_v<<<((size_t)T*H*D+threads-1)/threads,threads,0,stream>>>(
            v32,workspace.v,sv2);
        if (inspect) {
            inspect_f32("g_input",g32,(size_t)T*H,stream);
            inspect_bf16("k_cast",workspace.k,(size_t)T*HG*D,stream);
            inspect_bf16("v_cast",workspace.v,(size_t)T*H*D,stream);
            inspect_f32("beta",beta32,(size_t)T*H,stream);
        }
        state_to_fla<<<((size_t)H*D*D+threads-1)/threads,threads,0,stream>>>(
            state_in,workspace.state_fla);

        const int t = T;
        auto trace = [](const char * stage) {
            if (std::getenv("ESCHA_GDN_DEBUG") != nullptr) std::fprintf(stderr, "GDN_STAGE %s\n", stage);
        };
        CUdeviceptr null_scratch=0;
        CUdeviceptr pg=(CUdeviceptr)g32, pgs=(CUdeviceptr)workspace.gsum;
        CUdeviceptr pseq=(CUdeviceptr)workspace.cu_seqlens;
        CUdeviceptr pci64=(CUdeviceptr)workspace.chunk_indices64;
        CUdeviceptr pci16=(CUdeviceptr)workspace.chunk_indices16;
        CUdeviceptr pindex=(CUdeviceptr)workspace.initial_state_indices;
        // Non-varlen Triton ABI: slots 2/3 and 5/6 are retained but unused;
        // T is slot 4 (the cached bridge cubins are not the varlen ABI).
        void * ac[]={&pg,&pgs,workspace.matched_sglang ? &pseq : &null_scratch,
                     workspace.matched_sglang ? &pci64 : &null_scratch,
                     const_cast<int *>(&t),&null_scratch,&null_scratch};
        trace("cumsum");
        launch(workspace.cumsum.function,dim3(NT,H,1),8,8,ac,stream);
        if (inspect) inspect_f32("gsum",workspace.gsum,(size_t)T*H,stream);
        CUdeviceptr pk=(CUdeviceptr)workspace.k,pb=(CUdeviceptr)beta32,pA=(CUdeviceptr)workspace.A;
        // KKT: k,beta,g,A,cu_seqlens,chunk_indices,T plus two unused slots.
        void * ak[]={&pk,&pb,&pgs,&pA,
                     workspace.matched_sglang ? &pseq : &null_scratch,
                     workspace.matched_sglang ? &pci64 : &null_scratch,
                     const_cast<int *>(&t),&null_scratch,&null_scratch};
        cuda_ok(cudaMemsetAsync(workspace.A,0,(size_t)T*H*64*4,stream),"clear KKT matrix");
        trace("kkt");
        if (!workspace.matched_sglang && std::getenv("ESCHA_GDN_CUSTOM_KKT") != nullptr) {
            kkt_reference_cuda<<<dim3(NT,H),256,0,stream>>>(
                workspace.k,beta32,workspace.gsum,workspace.A);
            cuda_ok(cudaGetLastError(),"custom KKT launch");
        } else {
            launch(workspace.kkt.function,dim3(NT,H,1),8,16384,ak,stream);
        }
        if (inspect) inspect_f32("A",workspace.A,(size_t)T*H*64,stream);
        CUdeviceptr pAi=(CUdeviceptr)workspace.Ai;
        CUdeviceptr pAd=(CUdeviceptr)workspace.Ad;
        if (workspace.matched_sglang) {
            void * as[]={&pA,&pAd,&pseq,&pci16,const_cast<int *>(&t),
                         &null_scratch,&null_scratch};
            trace("solve16");
            launch(workspace.solve.function,dim3(N16,H,1),1,0,as,stream);
            if (inspect) inspect_f32("Ad",workspace.Ad,(size_t)T*H*16,stream);
        }
        // This cached merge computes the block inverse directly from A.
        // Its upper triangle is not written, so initialize the whole output.
        cuda_ok(cudaMemsetAsync(workspace.Ai,0,(size_t)T*H*64*2,stream),"clear inverse");
        // merge: A,Ai,cu_seqlens,chunk_indices,T plus two unused slots.
        void * am[]={&pA,&pAi,&null_scratch,&null_scratch,const_cast<int *>(&t),&null_scratch,&null_scratch};
        trace("merge");
        if (workspace.matched_sglang) {
            void * am_sglang[]={&pA,&pAd,&pAi,&pseq,&pci64,
                                const_cast<int *>(&t),&null_scratch,&null_scratch};
            launch(workspace.merge.function,dim3(NT,H,1),4,10240,am_sglang,stream);
        } else if (std::getenv("ESCHA_GDN_CUSTOM_INVERSE") != nullptr) {
            inverse64_reference_cuda<<<dim3(NT,H),64,0,stream>>>(workspace.A,workspace.Ai);
            cuda_ok(cudaGetLastError(),"custom inverse launch");
        } else {
            launch(workspace.merge.function,dim3(NT,H,1),4,10240,am,stream);
        }
        if (inspect) inspect_bf16("Ai_merge",workspace.Ai,(size_t)T*H*64,stream);
        CUdeviceptr pv=(CUdeviceptr)workspace.v,pw=(CUdeviceptr)workspace.w,pu=(CUdeviceptr)workspace.u;
        // recompute: k,v,beta,w,u,A,g,cu_seqlens,chunk_indices,T.
        void * ar[]={&pk,&pv,&pb,&pw,&pu,&pAi,&pgs,
                     workspace.matched_sglang ? &pseq : &null_scratch,
                     workspace.matched_sglang ? &pci64 : &null_scratch,
                     const_cast<int *>(&t),&null_scratch,&null_scratch};
        trace("recompute");
        launch(workspace.recompute.function,dim3(NT,H,1),4,
               workspace.matched_sglang ? 28672 : 32768,ar,stream);
        if (inspect) {
            inspect_bf16("w",workspace.w,(size_t)T*H*D,stream);
            inspect_bf16("u",workspace.u,(size_t)T*H*D,stream);
        }
        CUdeviceptr pvnew=(CUdeviceptr)workspace.vnew,ph=(CUdeviceptr)workspace.h;
        CUdeviceptr ps=(CUdeviceptr)workspace.state_fla;
        CUdeviceptr pht=(CUdeviceptr)workspace.state_final;
        CUdeviceptr poff=(CUdeviceptr)workspace.chunk_offsets;
        // Cached vLLM state ABI: k,v,w,v_new,g,h,h0,ht,cu_seqlens,
        // chunk_offsets,T, followed by two Triton scratch pointers.
        void * ah[]={&pk,&pu,&pw,&pvnew,&pgs,&ph,&ps,
                     workspace.matched_sglang ? &pindex : &pht,
                     &pseq,&poff,const_cast<int *>(&t),
                     &null_scratch,&null_scratch};
        cuda_ok(cudaMemsetAsync(workspace.h,0xff,(size_t)NT*H*D*D*2,stream),"mark chunk states");
        cuda_ok(cudaMemsetAsync(workspace.state_final,0xff,(size_t)H*D*D*4,stream),"mark final state");
        trace("state");
        launch(workspace.state.function,dim3(workspace.matched_sglang ? 4 : 2,H,1),4,
               workspace.matched_sglang ? 49412 : 90632,ah,stream);
        if (inspect) {
            inspect_bf16("h",workspace.h,(size_t)NT*H*D*D,stream);
            inspect_f32("state_final",workspace.matched_sglang ? workspace.state_fla : workspace.state_final,(size_t)H*D*D,stream);
        }
        cuda_ok(cudaMemsetAsync(workspace.out,0,(size_t)T*H*D*2,stream),"cudaMemset output");
        CUdeviceptr pq=(CUdeviceptr)workspace.q,po=(CUdeviceptr)workspace.out;
        // output: q,k,v,h,g,o,cu_seqlens,chunk_indices,scale,T plus two unused slots.
        void * ao[]={&pq,&pk,&pvnew,&ph,&pgs,&po,
                     workspace.matched_sglang ? &pseq : &null_scratch,
                     workspace.matched_sglang ? &pci64 : &null_scratch,
                     &scale,const_cast<int *>(&t),&null_scratch,&null_scratch};
        trace("output");
        launch(workspace.output.function,dim3(2,NT,H),4,32768,ao,stream);
        if (inspect) inspect_bf16("out",workspace.out,(size_t)T*H*D,stream);
        output_to_f32<<<((size_t)T*H*D+threads-1)/threads,threads,0,stream>>>(workspace.out,dst32);
        state_from_fla<<<((size_t)H*D*D+threads-1)/threads,threads,0,stream>>>(
            workspace.matched_sglang ? workspace.state_fla : workspace.state_final,state_out);
        cuda_ok(cudaPeekAtLastError(), "GDN bridge launch");
        return 0;
    } catch (const std::exception & e) {
        last_error=e.what();
        return -1;
    } catch (...) {
        last_error="unknown GDN bridge error";
        return -2;
    }
}

extern "C" const char * escha_gdn_chunk_error() { return last_error.c_str(); }
