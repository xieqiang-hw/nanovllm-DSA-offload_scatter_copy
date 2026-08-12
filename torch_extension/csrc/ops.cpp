#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "torch_npu/csrc/framework/OpCommand.h"

namespace {
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t K_ROPE_DIM = 64;
constexpr int64_t KV_CACHE_DIM = 512;

void CheckInputs(
    const at::Tensor& hbmKRoPE,
    const at::Tensor& hbmKvCache,
    const at::Tensor& dramKRoPE,
    const at::Tensor& dramKvCache,
    const at::Tensor& hbmBlockTable,
    const at::Tensor& dramBlockTable,
    const at::Tensor& srcTokenIds,
    const at::Tensor& dstSlots,
    const at::Tensor& copyCounts)
{
    TORCH_CHECK(hbmKRoPE.dim() == 3 && hbmKRoPE.size(1) == BLOCK_SIZE &&
                    hbmKRoPE.size(2) == K_ROPE_DIM,
                "HBM KPE must be [blocks,128,64].");
    TORCH_CHECK(hbmKvCache.dim() == 3 && hbmKvCache.size(1) == BLOCK_SIZE &&
                    hbmKvCache.size(2) == KV_CACHE_DIM,
                "HBM CKV must be [blocks,128,512].");
    TORCH_CHECK(dramKRoPE.dim() == 3 && dramKRoPE.size(1) == BLOCK_SIZE &&
                    dramKRoPE.size(2) == K_ROPE_DIM,
                "DRAM KPE must be [blocks,128,64].");
    TORCH_CHECK(dramKvCache.dim() == 3 && dramKvCache.size(1) == BLOCK_SIZE &&
                    dramKvCache.size(2) == KV_CACHE_DIM,
                "DRAM CKV must be [blocks,128,512].");
    TORCH_CHECK(hbmKRoPE.size(0) == hbmKvCache.size(0) &&
                    dramKRoPE.size(0) == dramKvCache.size(0),
                "CKV/KPE block counts must match in each memory tier.");
    TORCH_CHECK(hbmBlockTable.dim() == 2 && dramBlockTable.dim() == 2 &&
                    srcTokenIds.dim() == 2 && dstSlots.dim() == 2 &&
                    copyCounts.dim() == 1,
                "Block tables, IDs, slots and counts have invalid ranks.");
    TORCH_CHECK(srcTokenIds.sizes() == dstSlots.sizes() &&
                    srcTokenIds.size(0) == copyCounts.size(0) &&
                    hbmBlockTable.size(0) == copyCounts.size(0) &&
                    dramBlockTable.size(0) == copyCounts.size(0),
                "All batch dimensions and ID capacities must agree.");

    const auto dtype = hbmKRoPE.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf || dtype == at::kChar,
                "A5KvcacheScatterCopy supports bf16/fp16/int8.");
    for (const at::Tensor* tensor : {&hbmKvCache, &dramKRoPE, &dramKvCache}) {
        TORCH_CHECK(tensor->scalar_type() == dtype,
                    "All CKV/KPE tensors must use the same dtype.");
    }
    for (const at::Tensor* tensor :
         {&hbmBlockTable, &dramBlockTable, &srcTokenIds, &dstSlots, &copyCounts}) {
        TORCH_CHECK(tensor->scalar_type() == at::kInt,
                    "All metadata tensors must be int32.");
    }
    for (const at::Tensor* tensor : {&hbmKRoPE, &hbmKvCache, &dramKRoPE,
                                     &dramKvCache, &hbmBlockTable, &dramBlockTable,
                                     &srcTokenIds, &dstSlots, &copyCounts}) {
        TORCH_CHECK(tensor->device() == hbmKRoPE.device() && tensor->is_contiguous(),
                    "All inputs must be contiguous tensors on one NPU.");
    }
}

std::tuple<at::Tensor, at::Tensor> KvcacheScatterCopyNpu(
    at::Tensor hbmKRoPE, at::Tensor hbmKvCache,
    const at::Tensor& dramKRoPE, const at::Tensor& dramKvCache,
    const at::Tensor& hbmBlockTable, const at::Tensor& dramBlockTable,
    const at::Tensor& srcTokenIds, const at::Tensor& dstSlots,
    const at::Tensor& copyCounts)
{
    CheckInputs(hbmKRoPE, hbmKvCache, dramKRoPE, dramKvCache, hbmBlockTable,
                dramBlockTable, srcTokenIds, dstSlots, copyCounts);
    at_npu::native::OpCommand cmd;
    cmd.Name("A5KvcacheScatterCopy")
        .Input(hbmKRoPE).Input(hbmKvCache).Input(dramKRoPE).Input(dramKvCache)
        .Input(hbmBlockTable).Input(dramBlockTable).Input(srcTokenIds)
        .Input(dstSlots).Input(copyCounts).Output(hbmKRoPE).Output(hbmKvCache).Run();
    return std::make_tuple(hbmKRoPE, hbmKvCache);
}

std::tuple<at::Tensor, at::Tensor> KvcacheScatterCopyMeta(
    at::Tensor hbmKRoPE, at::Tensor hbmKvCache, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&)
{
    return std::make_tuple(hbmKRoPE, hbmKvCache);
}
} // namespace

TORCH_LIBRARY(ops_dsa_offload_a5, m)
{
    m.def("kvcache_scatter_copy("
          "Tensor(a!) hbm_k_rope, Tensor(b!) hbm_kv_cache, "
          "Tensor dram_k_rope, Tensor dram_kv_cache, "
          "Tensor hbm_block_table, Tensor dram_block_table, "
          "Tensor source_token_ids, Tensor destination_slots, Tensor copy_counts"
          ") -> (Tensor(a!), Tensor(b!))");
}

TORCH_LIBRARY_IMPL(ops_dsa_offload_a5, PrivateUse1, m)
{
    m.impl("kvcache_scatter_copy", &KvcacheScatterCopyNpu);
}

TORCH_LIBRARY_IMPL(ops_dsa_offload_a5, Meta, m)
{
    m.impl("kvcache_scatter_copy", &KvcacheScatterCopyMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
