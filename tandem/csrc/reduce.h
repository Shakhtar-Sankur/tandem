// M3: the reduce kernel's interface (see reduce.cu).
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace tandem {

enum class DType { kFloat, kDouble, kHalf, kBFloat16, kOther };

bool add_supported(DType dtype);

// Enqueues dst[i] += src[i] for i < n on `stream`. src may be on another GPU
// when peer access is enabled. Returns the launch error, if any.
cudaError_t add_into(void* dst, const void* src, std::int64_t n, DType dtype, cudaStream_t stream);

}  // namespace tandem
