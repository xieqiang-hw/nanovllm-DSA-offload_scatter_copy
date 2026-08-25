#pragma once

#include <initializer_list>
#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace nanovllm_dsa_a5_impl {

constexpr int64_t kBlockSize = 128;
constexpr int64_t kKpeDim = 64;
constexpr int64_t kCkvDim = 512;
constexpr int64_t kMaxSourceCapacity = 1 << 18;

inline void CheckOneDeviceAndContiguous(
    const at::Tensor& reference,
    std::initializer_list<const at::Tensor*> tensors,
    const char* op_name) {
  TORCH_CHECK(reference.device().is_privateuseone(), op_name, " inputs must be on NPU.");
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(
        tensor->device() == reference.device() && tensor->is_contiguous(),
        op_name, " inputs must be contiguous tensors on one NPU.");
  }
}

}  // namespace nanovllm_dsa_a5_impl
