#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(kvcache_ops, m) {
  m.def(
      "kvcache_scatter_copy(Tensor(a!) hbm_kv, Tensor dram_kv, "
      "Tensor(b!)? hbm_kpe, Tensor? dram_kpe, "
      "Tensor hbm_block_table, Tensor dram_block_table, "
      "Tensor source_token_ids, Tensor destination_slots, "
      "Tensor copy_counts) -> ()");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
