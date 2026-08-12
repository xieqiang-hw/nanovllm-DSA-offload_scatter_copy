#include <tuple>
#include <torch/library.h>

#include "ops_common.h"

namespace ops_overlap_impl {

void check_scatter_inputs(
    const at::Tensor& hbm_k_rope,
    const at::Tensor& hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts)
{
    TORCH_CHECK(
        hbm_k_rope.dim() == 3 && hbm_k_rope.size(1) == 128 &&
            hbm_k_rope.size(2) == 64,
        "HBM KPE must be [blocks,128,64].");
    TORCH_CHECK(
        hbm_kv_cache.dim() == 3 && hbm_kv_cache.size(1) == 128 &&
            hbm_kv_cache.size(2) == 512,
        "HBM CKV must be [blocks,128,512].");
    TORCH_CHECK(
        dram_k_rope.dim() == 3 && dram_k_rope.size(1) == 128 &&
            dram_k_rope.size(2) == 64 &&
            dram_kv_cache.dim() == 3 && dram_kv_cache.size(1) == 128 &&
            dram_kv_cache.size(2) == 512,
        "DRAM CKV/KPE shapes are invalid.");
    TORCH_CHECK(
        source_token_ids.dim() == 2 &&
            destination_slots.sizes() == source_token_ids.sizes() &&
            copy_counts.dim() == 1 &&
            copy_counts.size(0) == source_token_ids.size(0),
        "SCATTER IDs/slots/counts have inconsistent shapes.");
    TORCH_CHECK(
        hbm_block_table.dim() == 2 && dram_block_table.dim() == 2 &&
            hbm_block_table.size(0) == copy_counts.size(0) &&
            dram_block_table.size(0) == copy_counts.size(0),
        "SCATTER block tables must be [B,max_blocks].");
    const auto dtype = hbm_k_rope.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
                "SCATTER supports fp16/bf16.");
    for (const auto* tensor :
         {&hbm_kv_cache, &dram_k_rope, &dram_kv_cache}) {
        TORCH_CHECK(tensor->scalar_type() == dtype,
                    "SCATTER floating inputs must share one dtype.");
    }
    for (const auto* tensor :
         {&hbm_block_table, &dram_block_table, &source_token_ids,
          &destination_slots, &copy_counts}) {
        TORCH_CHECK(tensor->scalar_type() == at::kInt,
                    "SCATTER metadata must be int32.");
    }
    for (const auto* tensor :
         {&hbm_k_rope, &hbm_kv_cache, &dram_k_rope, &dram_kv_cache,
          &hbm_block_table, &dram_block_table, &source_token_ids,
          &destination_slots, &copy_counts}) {
        TORCH_CHECK(tensor->device() == hbm_k_rope.device() &&
                        tensor->is_contiguous(),
                    "SCATTER inputs must be contiguous on one NPU.");
    }
}

std::tuple<at::Tensor, at::Tensor> kvcache_scatter_copy_npu(
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts)
{
    check_scatter_inputs(
        hbm_k_rope, hbm_kv_cache, dram_k_rope, dram_kv_cache,
        hbm_block_table, dram_block_table, source_token_ids,
        destination_slots, copy_counts);
    EXEC_NPU_CMD_V1(
        aclnnNanovllmKvcacheScatterCopy,
        hbm_k_rope, hbm_kv_cache, dram_k_rope, dram_kv_cache,
        hbm_block_table, dram_block_table, source_token_ids,
        destination_slots, copy_counts);
    return std::make_tuple(hbm_k_rope, hbm_kv_cache);
}

std::tuple<at::Tensor, at::Tensor> kvcache_scatter_copy_meta(
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&)
{
    return std::make_tuple(hbm_k_rope, hbm_kv_cache);
}

}  // namespace ops_overlap_impl

TORCH_LIBRARY_IMPL(ops_overlap, PrivateUse1, m) {
    m.impl("kvcache_scatter_copy",
           &ops_overlap_impl::kvcache_scatter_copy_npu);
}

TORCH_LIBRARY_IMPL(ops_overlap, Meta, m) {
    m.impl("kvcache_scatter_copy",
           &ops_overlap_impl::kvcache_scatter_copy_meta);
}
