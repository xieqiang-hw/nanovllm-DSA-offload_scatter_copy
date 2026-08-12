#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(ops_overlap, m) {
    m.def(
        "kvcache_scatter_copy("
        "Tensor(a!) hbm_k_rope, Tensor(b!) hbm_kv_cache, "
        "Tensor dram_k_rope, Tensor dram_kv_cache, "
        "Tensor hbm_block_table, Tensor dram_block_table, "
        "Tensor source_token_ids, Tensor destination_slots, "
        "Tensor copy_counts"
        ") -> (Tensor(a!), Tensor(b!))");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
