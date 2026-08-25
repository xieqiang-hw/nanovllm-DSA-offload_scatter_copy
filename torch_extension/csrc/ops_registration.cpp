#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(nanovllm_dsa, m) {
  m.def(
      "kvcache_scatter_copy(Tensor(a!) hbm_kpe, Tensor(b!) hbm_ckv, "
      "Tensor dram_kpe, Tensor dram_ckv, Tensor hbm_block_table, "
      "Tensor dram_block_table, Tensor source_token_ids, "
      "Tensor destination_slots, Tensor copy_counts) "
      "-> (Tensor(a!), Tensor(b!))");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
