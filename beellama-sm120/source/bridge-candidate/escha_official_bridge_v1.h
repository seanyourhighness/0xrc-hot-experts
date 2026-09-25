#pragma once

// Additive, CUDA-only bridge ABI.  The legacy entry points remain unchanged;
// this header is for callers that need per-call input ABI selection and a
// stable descriptor contract for future batched speculative runtimes.

#include <cuda_runtime_api.h>

#include <cstdint>

enum : uint32_t {
    ESCHA_OFFICIAL_BRIDGE_ABI_V1 = 1,
    ESCHA_OFFICIAL_BRIDGE_INPUT_F16 = 0,
    ESCHA_OFFICIAL_BRIDGE_INPUT_F32 = 1,
};

enum : uint32_t {
    ESCHA_OFFICIAL_BRIDGE_FLAG_NATIVE_EPILOGUE = 1u << 0,
    ESCHA_OFFICIAL_BRIDGE_FLAG_SWIGLU = 1u << 1,
    ESCHA_OFFICIAL_BRIDGE_FLAG_ROTATED_F16_INPUT = 1u << 2,
    ESCHA_OFFICIAL_BRIDGE_FLAG_DFLASH2 = 1u << 3,
};

// One existing Escha decode-main projection.  The bridge does not infer any
// tensor layout from a model object: all dimensions and device pointers are
// explicit, and the caller owns their lifetime until the stream reaches the
// queued work.
struct escha_official_bridge_decode_desc_v1 {
    uint32_t abi_version;
    uint32_t struct_bytes;
    int32_t device;
    int32_t M;
    int32_t IC;
    int32_t OC;
    int32_t K;
    int32_t splits;
    uint32_t input_abi;
    uint32_t flags;
    const char * cubin_path;       // nullptr: use the process default
    cudaStream_t stream;
    float * partial_f32;
    const float * x_f32;
    const int16_t * code;
    const void * rin_f16;
    const void * rout_f16;         // reserved for a future fused epilogue
    void * output_f16;             // reserved for a future legacy epilogue
};

// A batch is an enqueue envelope, not a claim that the retained cubin has a
// multi-projection kernel.  The v1 implementation may enqueue one supported
// projection per descriptor on the supplied stream.  Future DFlash2 cubins can
// add a fused implementation without changing the envelope or the legacy ABI.
struct escha_official_bridge_batch_v1 {
    uint32_t abi_version;
    uint32_t struct_bytes;
    int32_t device;
    int32_t projection_count;
    uint32_t input_abi;
    uint32_t flags;
    const char * cubin_path;
    cudaStream_t stream;
    const escha_official_bridge_decode_desc_v1 * projections;
};

extern "C" int escha_official_bridge_abi_version();

extern "C" int escha_official_decode_gemv_main_desc_v1(
        const escha_official_bridge_decode_desc_v1 * desc);

extern "C" int escha_official_decode_gemv_main_batch_v1(
        const escha_official_bridge_batch_v1 * batch);
