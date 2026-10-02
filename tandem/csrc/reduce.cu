// M3: dst[i] += src[i] on the GPU, for the reduce step of a collective.
//
// src may live on another GPU (a peer's slot, read over peer-to-peer access) or
// in this GPU's memory. The arithmetic matches ATen's add_ exactly: float and
// double add in their own precision; half and bfloat16 add in float and round
// once to nearest-even, as ATen's "opmath" does. So results stay bit-identical
// to the Python engine and to PyTorch.
//
// No ATen here: plain pointers, so this file needs only the CUDA toolkit.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "reduce.h"

namespace tandem {

namespace {

template <class T>
struct Op {
  __device__ static T add(T a, T b) { return a + b; }
};
template <>
struct Op<__half> {
  __device__ static __half add(__half a, __half b) { return __float2half_rn(__half2float(a) + __half2float(b)); }
};
template <>
struct Op<__nv_bfloat16> {
  __device__ static __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16_rn(__bfloat162float(a) + __bfloat162float(b));
  }
};

template <class T>
__global__ void add_kernel(T* __restrict__ dst, const T* __restrict__ src, std::int64_t n) {
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    dst[i] = Op<T>::add(dst[i], src[i]);
  }
}

// float with 16-byte loads and stores: four elements per thread per step.
__global__ void add_kernel_f4(float4* __restrict__ dst, const float4* __restrict__ src, std::int64_t n4) {
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 a = dst[i];
    const float4 b = src[i];
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    dst[i] = a;
  }
}

int blocks_for(std::int64_t n, int threads) {
  const std::int64_t b = (n + threads - 1) / threads;
  return static_cast<int>(b < 1024 ? (b < 1 ? 1 : b) : 1024);  // grid-stride covers the rest
}

template <class T>
void launch(void* dst, const void* src, std::int64_t n, cudaStream_t stream) {
  constexpr int kThreads = 256;
  add_kernel<T><<<blocks_for(n, kThreads), kThreads, 0, stream>>>(static_cast<T*>(dst), static_cast<const T*>(src), n);
}

}  // namespace

bool add_supported(DType dtype) { return dtype != DType::kOther; }

cudaError_t add_into(void* dst, const void* src, std::int64_t n, DType dtype, cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  constexpr int kThreads = 256;
  switch (dtype) {
    case DType::kFloat: {
      const bool aligned = reinterpret_cast<std::uintptr_t>(dst) % 16 == 0 && reinterpret_cast<std::uintptr_t>(src) % 16 == 0;
      const std::int64_t n4 = aligned ? n / 4 : 0;
      if (n4 > 0) {
        add_kernel_f4<<<blocks_for(n4, kThreads), kThreads, 0, stream>>>(static_cast<float4*>(dst),
                                                                          static_cast<const float4*>(src), n4);
      }
      if (n4 * 4 < n) launch<float>(static_cast<float*>(dst) + n4 * 4, static_cast<const float*>(src) + n4 * 4, n - n4 * 4, stream);
      break;
    }
    case DType::kDouble:
      launch<double>(dst, src, n, stream);
      break;
    case DType::kHalf:
      launch<__half>(dst, src, n, stream);
      break;
    case DType::kBFloat16:
      launch<__nv_bfloat16>(dst, src, n, stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace tandem
