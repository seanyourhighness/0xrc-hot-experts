#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "escha_official_bridge_v1.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// CUDA-only replacement for the Sprint/Bridge C ABI.  The code-GEMM and
// F32-input decode cubins are extracted from the retained Escha wheel; this
// adapter deliberately does not call torch, SGLang, or the native BeeLlama
// prefill implementation.

namespace {

thread_local std::string g_error;
std::mutex g_mutex;

CUmodule g_code_module = nullptr;
std::unordered_map<std::string, CUfunction> g_code_functions;
std::unordered_map<uint64_t, float *> g_ones;

struct decode_module_key {
    int device;
    std::string path;

    bool operator == (const decode_module_key & other) const {
        return device == other.device && path == other.path;
    }
};

struct decode_module_key_hash {
    size_t operator()(const decode_module_key & key) const {
        size_t h = std::hash<int>{}(key.device);
        h ^= std::hash<std::string>{}(key.path) + size_t(0x9e3779b9) + (h << 6) + (h >> 2);
        return h;
    }
};

struct decode_module_functions {
    CUmodule module = nullptr;
    CUfunction k2 = nullptr;
    CUfunction k3 = nullptr;
};

std::unordered_map<decode_module_key, decode_module_functions, decode_module_key_hash>
    g_decode_modules;

struct input_buffer_key {
    int device;
    uintptr_t stream;
    size_t length;

    bool operator == (const input_buffer_key & other) const {
        return device == other.device && stream == other.stream && length == other.length;
    }
};

struct input_buffer_key_hash {
    size_t operator()(const input_buffer_key & key) const {
        size_t h = std::hash<int>{}(key.device);
        h ^= std::hash<uintptr_t>{}(key.stream) + size_t(0x9e3779b9) + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(key.length) + size_t(0x9e3779b9) + (h << 6) + (h >> 2);
        return h;
    }
};

// The conversion is asynchronous.  A buffer keyed only by (device, length)
// can be overwritten by a second stream while the first decode still reads
// it.  Stream-keying preserves the no-hot-allocation behavior while making
// concurrent gate/up or DFlash-style streams independent.
std::unordered_map<input_buffer_key, uint16_t *, input_buffer_key_hash> g_input_f16;

void set_driver_error(const char * what, CUresult status) {
    const char * text = nullptr;
    cuGetErrorString(status, &text);
    g_error = std::string(what) + ": " + (text ? text : "unknown driver error");
}

bool ensure_context(int device) {
    if (cudaSetDevice(device) != cudaSuccess || cuInit(0) != CUDA_SUCCESS) {
        g_error = "failed to initialize CUDA context";
        return false;
    }
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS) {
        g_error = "cuCtxGetCurrent failed";
        return false;
    }
    if (current == nullptr) {
        CUdevice cu_device = 0;
        if (cuDeviceGet(&cu_device, device) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(&current, cu_device) != CUDA_SUCCESS ||
            cuCtxSetCurrent(current) != CUDA_SUCCESS) {
            g_error = "failed to establish CUDA primary context";
            return false;
        }
    }
    return true;
}

bool load_code_module(int device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_code_module != nullptr) {
        return true;
    }
    const char * path = std::getenv("ESCHA_OFFICIAL_CODE_GEMM_CUBIN");
    if (path == nullptr || path[0] == '\0') {
        g_error = "ESCHA_OFFICIAL_CODE_GEMM_CUBIN is required";
        return false;
    }
    if (!ensure_context(device)) {
        return false;
    }
    const CUresult status = cuModuleLoad(&g_code_module, path);
    if (status != CUDA_SUCCESS) {
        set_driver_error("cuModuleLoad code-GEMM cubin failed", status);
        g_code_module = nullptr;
        return false;
    }
    return true;
}

bool load_decode_module(int device, const char * requested_path,
                        CUfunction * k2_out, CUfunction * k3_out) {
    const char * path = requested_path;
    if (path == nullptr || path[0] == '\0') {
        path = std::getenv("ESCHA_OFFICIAL_F32_DECODE_CUBIN");
    }
    if (path == nullptr || path[0] == '\0') {
        g_error = "ESCHA_OFFICIAL_F32_DECODE_CUBIN is required";
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    const decode_module_key key { device, path };
    auto cached = g_decode_modules.find(key);
    if (cached != g_decode_modules.end()) {
        if (k2_out != nullptr) {
            *k2_out = cached->second.k2;
        }
        if (k3_out != nullptr) {
            *k3_out = cached->second.k3;
        }
        return true;
    }
    if (!ensure_context(device)) {
        return false;
    }
    decode_module_functions functions;
    CUresult status = cuModuleLoad(&functions.module, path);
    if (status != CUDA_SUCCESS) {
        set_driver_error("cuModuleLoad F32 decode cubin failed", status);
        return false;
    }
    constexpr const char * k2_symbol =
        "_ZN12escha_escham21escham_gemv_bw_kernelILi1ELi2ELb0ELb0ELb0EEEvPfPK6__halfPKtS4_PKfiiiiiPS2_S4_S8_Pi";
    constexpr const char * k3_symbol =
        "_ZN12escha_escham21escham_gemv_bw_kernelILi1ELi3ELb0ELb0ELb0EEEvPfPK6__halfPKtS4_PKfiiiiiPS2_S4_S8_Pi";
    status = cuModuleGetFunction(&functions.k2, functions.module, k2_symbol);
    if (status == CUDA_SUCCESS) {
        status = cuModuleGetFunction(&functions.k3, functions.module, k3_symbol);
    }
    if (status != CUDA_SUCCESS) {
        set_driver_error("cuModuleGetFunction F32 decode failed", status);
        cuModuleUnload(functions.module);
        return false;
    }

    auto inserted = g_decode_modules.emplace(key, functions);
    if (k2_out != nullptr) {
        *k2_out = inserted.first->second.k2;
    }
    if (k3_out != nullptr) {
        *k3_out = inserted.first->second.k3;
    }
    return true;
}

const char * code_symbol(int K, bool small, int acc_mode, bool fast_down) {
    // Existing vendor instantiation: only the qualified long-IC K3 down shape.
    if (fast_down) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi3ELi64ELi64ELi3ELb0ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 2 && !small && acc_mode == 0) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi2ELi128ELi64ELi2ELb0ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 2 && !small && acc_mode == 1) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi2ELi128ELi64ELi2ELb1ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 3 && !small && acc_mode == 0) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi3ELi128ELi64ELi2ELb0ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 3 && !small && acc_mode == 1) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi3ELi128ELi64ELi2ELb1ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 2 && small && acc_mode == 0) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi2ELi64ELi32ELi3ELb0ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 2 && small && acc_mode == 1) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi2ELi64ELi32ELi3ELb1ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 3 && small && acc_mode == 0) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi3ELi64ELi32ELi3ELb0ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    if (K == 3 && small && acc_mode == 1) {
        return "_ZN12escha_escham23escham_code_gemm_kernelILi1ELi3ELi64ELi32ELi3ELb1ELb1EEEvPfP6__halfPKS2_PKtS5_PKfiiii";
    }
    return nullptr;
}

CUfunction resolve_code(int device, int K, bool small, int acc_mode, bool fast_down) {
    if (!load_code_module(device)) {
        return nullptr;
    }
    const char * symbol = code_symbol(K, small, acc_mode, fast_down);
    if (symbol == nullptr) {
        g_error = "unsupported raw Escha code-GEMM selection";
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_code_functions.find(symbol);
    if (it != g_code_functions.end()) {
        return it->second;
    }
    CUfunction function = nullptr;
    const CUresult status = cuModuleGetFunction(&function, g_code_module, symbol);
    if (status != CUDA_SUCCESS) {
        set_driver_error("cuModuleGetFunction code-GEMM failed", status);
        return nullptr;
    }
    g_code_functions.emplace(symbol, function);
    return function;
}

float * ones_for(int device, int length) {
    const uint64_t key = (uint64_t(uint32_t(device)) << 32) | uint32_t(length);
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_ones.find(key);
    if (it != g_ones.end()) {
        return it->second;
    }
    std::vector<float> host((size_t) length, 1.0f);
    float * device_ptr = nullptr;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMalloc(&device_ptr, host.size() * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(device_ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        if (device_ptr != nullptr) {
            cudaFree(device_ptr);
        }
        g_error = "failed to allocate Escha scale vector";
        return nullptr;
    }
    g_ones.emplace(key, device_ptr);
    return device_ptr;
}

uint16_t * input_f16_for(int device, cudaStream_t stream, size_t length) {
    const input_buffer_key key {
        device,
        reinterpret_cast<uintptr_t>(stream),
        length,
    };
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_input_f16.find(key);
    if (it != g_input_f16.end()) {
        return it->second;
    }
    uint16_t * device_ptr = nullptr;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMalloc(&device_ptr, length * sizeof(uint16_t)) != cudaSuccess) {
        g_error = "failed to allocate Escha FP16 decode input";
        if (device_ptr != nullptr) {
            cudaFree(device_ptr);
        }
        return nullptr;
    }
    g_input_f16.emplace(key, device_ptr);
    return device_ptr;
}

// The retained code-GEMM decode symbols consume the original F16 activation
// ABI.  The retargeted f32_input cubin is different: the PTX changes the four
// activation loads to ld.global.nc.f32 while retaining the historical mangled
// name.  Make that ABI choice explicit instead of silently feeding one type to
// the other (which produces plausible-looking but incorrect tokens).
bool decode_input_is_f32_abi() {
    const char * value = std::getenv("ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "f16") == 0) {
        return false;
    }
    if (std::strcmp(value, "f32") == 0) {
        return true;
    }
    g_error = "ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI must be f16 or f32";
    return false;
}

bool require_direct_f32_for_batched_decode() {
    const char * value = std::getenv("ESCHA_OFFICIAL_REQUIRE_DIRECT_F32");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

__global__ void convert_f32_to_f16(const float * src, uint16_t * dst, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        reinterpret_cast<__half *>(dst)[i] = __float2half(src[i]);
    }
}

int decode_main_f32_launch(
        cudaStream_t stream,
        int device,
        float * partial_f32,
        const float * x_f32,
        const int16_t * code,
        const void * rin_f16,
        int M,
        int IC,
        int OC,
        int K,
        int splits,
        uint32_t input_abi,
        uint32_t flags,
        const char * cubin_path) {
    if (partial_f32 == nullptr || x_f32 == nullptr || code == nullptr || rin_f16 == nullptr ||
        M <= 0 || M > 16 || IC <= 0 || OC <= 0 || (K != 2 && K != 3) ||
        IC % 128 != 0 || OC % 128 != 0 || splits <= 0 || splits > IC / 128) {
        g_error = "invalid F32-input decode-main arguments";
        return -1;
    }
    if (input_abi != ESCHA_OFFICIAL_BRIDGE_INPUT_F16 &&
        input_abi != ESCHA_OFFICIAL_BRIDGE_INPUT_F32) {
        g_error = "unsupported Escha decode input ABI";
        return -1;
    }
    if ((flags & ESCHA_OFFICIAL_BRIDGE_FLAG_ROTATED_F16_INPUT) != 0) {
        g_error = "rotated-F16 input is not supported by the decode-main descriptor";
        return -1;
    }

    CUfunction decode_k2 = nullptr;
    CUfunction decode_k3 = nullptr;
    if (!load_decode_module(device, cubin_path, &decode_k2, &decode_k3)) {
        return -1;
    }
    float * s_in = ones_for(device, IC);
    if (s_in == nullptr) {
        return -1;
    }
    const bool input_f32_abi = input_abi == ESCHA_OFFICIAL_BRIDGE_INPUT_F32;
    if (M > 1 && require_direct_f32_for_batched_decode() && !input_f32_abi) {
        g_error = "batched decode requires ESCHA_OFFICIAL_F32_DECODE_INPUT_ABI=f32";
        return -1;
    }

    void * input_arg = nullptr;
    const size_t input_count = size_t(M) * size_t(IC);
    if (input_f32_abi) {
        input_arg = const_cast<float *>(x_f32);
    } else {
        uint16_t * input_f16 = input_f16_for(device, stream, input_count);
        if (input_f16 == nullptr) {
            return -1;
        }
        convert_f32_to_f16<<<(input_count + 255) / 256, 256, 0, stream>>>(
            x_f32, input_f16, input_count);
        if (cudaGetLastError() != cudaSuccess) {
            g_error = "failed to launch Escha FP32-to-FP16 decode conversion";
            return -1;
        }
        input_arg = input_f16;
    }

    const uint16_t * packed = reinterpret_cast<const uint16_t *>(code);
    const void * rin = rin_f16;
    int output_blocks = OC / 16;
    void * bias = nullptr;
    void * bias_scale = nullptr;
    float * correction = nullptr;
    int * route = nullptr;
    void * arguments[] = {
        &partial_f32, &input_arg, const_cast<uint16_t **>(&packed),
        const_cast<void **>(&rin), &s_in, &M, &IC, &OC, &output_blocks, &splits,
        &bias, &bias_scale, &correction, &route,
    };
    const CUresult status = cuLaunchKernel(
        K == 2 ? decode_k2 : decode_k3,
        (unsigned) (OC / 128), (unsigned) splits, 1,
        256, 1, 1, 0, reinterpret_cast<CUstream>(stream), arguments, nullptr);
    if (status != CUDA_SUCCESS) {
        set_driver_error("raw Escha F32 decode launch failed", status);
        return -1;
    }
    return 0;
}

int unsupported(const char * name) {
    g_error = std::string(name) + " is unavailable in the CUDA-only raw bridge";
    return -1;
}

} // namespace

extern "C" int escha_official_code_gemm(
        cudaStream_t, int, float *, const float *, const int16_t *, const void *, const void *,
        int, int, int, int, int, int, int) {
    return unsupported("escha_official_code_gemm");
}

extern "C" int escha_official_code_gemm_pretransformed(
        cudaStream_t stream,
        int device,
        void * dst_f16,
        const void * x_rotated_f16,
        const int16_t * code,
        const void * rout_f16,
        int M,
        int IC,
        int OC,
        int K,
        int acc_mode) {
    try {
        g_error.clear();
        if (dst_f16 == nullptr || x_rotated_f16 == nullptr || code == nullptr ||
            rout_f16 == nullptr || M <= 0 || IC <= 0 || OC <= 0 ||
            (K != 2 && K != 3) || IC % 128 != 0 || OC % 128 != 0 ||
            (acc_mode != 0 && acc_mode != 1)) {
            g_error = "invalid pretransformed Escha GEMM argument";
            return -1;
        }
        const bool small = OC <= 1024;
        const bool fast_down = K == 3 && IC == 17408 && OC == 5120 && acc_mode == 0;
        CUfunction function = resolve_code(device, K, small, acc_mode, fast_down);
        if (function == nullptr) {
            return -1;
        }
        float * accumulation = nullptr;
        auto * output = static_cast<uint16_t *>(dst_f16);
        const auto * input = static_cast<const uint16_t *>(x_rotated_f16);
        const auto * packed = reinterpret_cast<const uint16_t *>(code);
        const auto * rout = static_cast<const uint16_t *>(rout_f16);
        float * s_out = ones_for(device, OC);
        if (s_out == nullptr) {
            return -1;
        }
        const int output_blocks = OC / 16;
        void * arguments[] = {
            &accumulation, &output, &input, &packed, &rout, &s_out,
            &M, &IC, &OC, const_cast<int *>(&output_blocks),
        };
        const int BM = fast_down || small ? 64 : 128;
        const CUresult status = cuLaunchKernel(
            function, (unsigned) (OC / 128), (unsigned) ((M + BM - 1) / BM), 1,
            256, 1, 1, 0, reinterpret_cast<CUstream>(stream), arguments, nullptr);
        if (status != CUDA_SUCCESS) {
            set_driver_error("raw Escha code-GEMM launch failed", status);
            return -1;
        }
        return 0;
    } catch (...) {
        g_error = "unexpected exception in raw Escha code-GEMM";
        return -2;
    }
}

extern "C" int escha_official_decode_gemv_raw(
        cudaStream_t, int, float *, void *, const void *, const int16_t *, const void *, const void *,
        int, int, int, int, int) {
    return unsupported("escha_official_decode_gemv_raw");
}

extern "C" int escha_official_decode_gemv_main_raw(
        cudaStream_t, int, float *, const void *, const int16_t *, const void *,
        int, int, int, int, int) {
    return unsupported("escha_official_decode_gemv_main_raw");
}

extern "C" int escha_official_decode_gemv_main_f32_raw(
        cudaStream_t stream,
        int device,
        float * partial_f32,
        const float * x_f32,
        const int16_t * code,
        const void * rin_f16,
        int M,
        int IC,
        int OC,
        int K,
    int splits) {
    try {
        g_error.clear();
        const bool input_f32_abi = decode_input_is_f32_abi();
        if (!g_error.empty()) {
            return -1;
        }
        return decode_main_f32_launch(
            stream, device, partial_f32, x_f32, code, rin_f16,
            M, IC, OC, K, splits,
            input_f32_abi ? ESCHA_OFFICIAL_BRIDGE_INPUT_F32
                          : ESCHA_OFFICIAL_BRIDGE_INPUT_F16,
            0, nullptr);
    } catch (...) {
        g_error = "unexpected exception in raw Escha F32 decode";
        return -2;
    }
}

extern "C" int escha_official_bridge_abi_version() {
    return ESCHA_OFFICIAL_BRIDGE_ABI_V1;
}

extern "C" int escha_official_decode_gemv_main_desc_v1(
        const escha_official_bridge_decode_desc_v1 * desc) {
    try {
        g_error.clear();
        if (desc == nullptr || desc->abi_version != ESCHA_OFFICIAL_BRIDGE_ABI_V1 ||
            desc->struct_bytes < sizeof(*desc)) {
            g_error = "invalid Escha bridge decode descriptor version or size";
            return -1;
        }
        return decode_main_f32_launch(
            desc->stream, desc->device, desc->partial_f32, desc->x_f32,
            desc->code, desc->rin_f16, desc->M, desc->IC, desc->OC,
            desc->K, desc->splits, desc->input_abi, desc->flags,
            desc->cubin_path);
    } catch (...) {
        g_error = "unexpected exception in Escha bridge decode descriptor";
        return -2;
    }
}

extern "C" int escha_official_decode_gemv_main_batch_v1(
        const escha_official_bridge_batch_v1 * batch) {
    try {
        g_error.clear();
        if (batch == nullptr || batch->abi_version != ESCHA_OFFICIAL_BRIDGE_ABI_V1 ||
            batch->struct_bytes < sizeof(*batch) || batch->projection_count <= 0 ||
            batch->projection_count > 64 || batch->projections == nullptr ||
            (batch->input_abi != ESCHA_OFFICIAL_BRIDGE_INPUT_F16 &&
             batch->input_abi != ESCHA_OFFICIAL_BRIDGE_INPUT_F32)) {
            g_error = "invalid Escha bridge batch descriptor version, size, or count";
            return -1;
        }
        for (int i = 0; i < batch->projection_count; ++i) {
            const escha_official_bridge_decode_desc_v1 & desc = batch->projections[i];
            if (desc.abi_version != ESCHA_OFFICIAL_BRIDGE_ABI_V1 ||
                desc.struct_bytes < sizeof(desc) || desc.device != batch->device ||
                desc.stream != batch->stream || desc.input_abi != batch->input_abi) {
                g_error = "Escha bridge batch contains an incompatible projection descriptor";
                return -1;
            }
            const char * path = desc.cubin_path != nullptr ? desc.cubin_path : batch->cubin_path;
            const int rc = decode_main_f32_launch(
                batch->stream, batch->device, desc.partial_f32, desc.x_f32,
                desc.code, desc.rin_f16, desc.M, desc.IC, desc.OC,
                desc.K, desc.splits, batch->input_abi, batch->flags | desc.flags, path);
            if (rc != 0) {
                return rc;
            }
        }
        return 0;
    } catch (...) {
        g_error = "unexpected exception in Escha bridge batch descriptor";
        return -2;
    }
}

extern "C" const char * escha_official_bridge_error() {
    return g_error.c_str();
}

extern "C" int escha_official_bridge_probe() {
    return load_code_module(0) ? 0 : -1;
}
