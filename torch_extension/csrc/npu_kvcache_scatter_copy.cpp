#include <limits>
#include <tuple>
#include <torch/extension.h>
#include <torch/library.h>
#include "op_api_common.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace {
using OptionalTensor = c10::optional<at::Tensor>;

void CheckCache(const at::Tensor& tensor, int64_t width, at::ScalarType dtype)
{
    TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == 4 &&
                tensor.size(0) > 0 && tensor.size(0) <= std::numeric_limits<uint32_t>::max() &&
                tensor.size(1) == 128 && tensor.size(2) == 1 && tensor.size(3) == width,
                "Cache must have matching dtype and shape [blocks,128,1,", width, "].");
}

void CheckInputs(
    const at::Tensor& hbm, const at::Tensor& dram,
    const OptionalTensor& hbmKpe, const OptionalTensor& dramKpe,
    const at::Tensor& hbmTable, const at::Tensor& dramTable,
    const at::Tensor& src, const at::Tensor& dst, const at::Tensor& counts)
{
    const bool bf16 = hbm.scalar_type() == at::kBFloat16;
    TORCH_CHECK(bf16 || hbm.scalar_type() == at::kChar, "Cache dtype must be bfloat16 or int8.");
    CheckCache(hbm, bf16 ? 512 : 656, hbm.scalar_type());
    CheckCache(dram, bf16 ? 512 : 656, hbm.scalar_type());
    TORCH_CHECK(hbmKpe.has_value() == bf16 && dramKpe.has_value() == bf16,
                "BF16 requires both KPE tensors; C8 requires both to be None.");
    if (bf16) {
        CheckCache(*hbmKpe, 64, at::kBFloat16);
        CheckCache(*dramKpe, 64, at::kBFloat16);
        TORCH_CHECK(hbmKpe->size(0) == hbm.size(0) && dramKpe->size(0) == dram.size(0),
                    "KV and KPE physical block counts must agree.");
    }
    TORCH_CHECK(counts.dim() == 1 && counts.size(0) > 0 &&
                counts.size(0) <= std::numeric_limits<uint32_t>::max(), "copy_counts must be [B], B > 0.");
    const auto batch = counts.size(0);
    TORCH_CHECK(src.dim() == 3 && src.size(0) == batch && src.size(1) == 1 &&
                src.size(2) >= 1 && src.size(2) <= 65536 && dst.sizes() == src.sizes(),
                "source_token_ids/destination_slots must be [B,1,C], 1 <= C <= 65536.");
    for (const auto* table : {&hbmTable, &dramTable})
        TORCH_CHECK(table->dim() == 2 && table->size(0) == batch && table->size(1) > 0 &&
                    table->size(1) <= (int64_t(1) << 24), "Block tables must be [B,blocks] with int32-addressable slots.");
    for (const auto* tensor : {&hbmTable, &dramTable, &src, &dst, &counts})
        TORCH_CHECK(tensor->scalar_type() == at::kInt, "Copy metadata must be int32.");
    for (const auto* tensor : {&hbm, &dram, bf16 ? &*hbmKpe : &hbm, bf16 ? &*dramKpe : &dram,
                              &hbmTable, &dramTable, &src, &dst, &counts})
        TORCH_CHECK(tensor->device() == hbm.device() && tensor->is_contiguous(),
                    "Inputs must be contiguous tensors on the same device.");
}

void Scatter(
    at::Tensor hbm, const at::Tensor& dram, const OptionalTensor& hbmKpe,
    const OptionalTensor& dramKpe, const at::Tensor& hbmTable, const at::Tensor& dramTable,
    const at::Tensor& src, const at::Tensor& dst, const at::Tensor& counts)
{
    CheckInputs(hbm, dram, hbmKpe, dramKpe, hbmTable, dramTable, src, dst, counts);
    TORCH_CHECK(hbm.device().is_privateuseone(), "kvcache_scatter_copy requires NPU tensors.");
    const c10_npu::NPUGuard deviceGuard(hbm.device());
    const auto keepalive = std::make_tuple(hbm, dram, hbmKpe, dramKpe, hbmTable, dramTable, src, dst, counts);
    // CANN cannot generate optional inout references reliably. C8 reuses existing
    // KV tensors for the required KPE slots; its kernel never accesses these slots.
    const auto& apiHbmKpe = hbmKpe.has_value() ? *hbmKpe : hbm;
    const auto& apiDramKpe = dramKpe.has_value() ? *dramKpe : dram;
    EXEC_NPU_CMD_ORDERED(aclnnKvcacheScatterCopy, keepalive,
                        hbm, dram, apiHbmKpe, apiDramKpe, hbmTable, dramTable, src, dst, counts);
}
} // namespace

TORCH_LIBRARY_IMPL(kvcache_ops, PrivateUse1, m) { m.impl("kvcache_scatter_copy", &Scatter); }
TORCH_LIBRARY_IMPL(kvcache_ops, Meta, m) { m.impl("kvcache_scatter_copy", &CheckInputs); }
