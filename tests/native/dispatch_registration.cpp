// Real PyTorch dispatcher/Meta checks; the test replaces only NPU launch headers.
#include <cassert>
#include <iostream>
#include <ATen/core/dispatch/Dispatcher.h>
#include "npu_kvcache_scatter_copy.cpp"
#include "ops_registration.cpp"

int main()
{
    const auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("kvcache_ops::kvcache_scatter_copy", "").typed<decltype(Scatter)>();
    assert(op.hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1));
    assert(op.hasKernelForDispatchKey(c10::DispatchKey::Meta));
    assert(op.schema().arguments().size() == 9 && op.schema().returns().empty());
    assert(op.schema().arguments()[0].alias_info()->isWrite());
    assert(op.schema().arguments()[2].alias_info()->isWrite());
    const auto ints = at::TensorOptions().dtype(at::kInt).device(c10::kMeta);
    const auto table = at::empty({2, 1}, ints);
    const auto ids = at::empty({2, 1, 128}, ints);
    const auto counts = at::empty({2}, ints);
    const auto reject = [](auto&& call) {
        try { call(); } catch (const c10::Error&) { return; }
        throw std::runtime_error("Invalid scatter inputs were accepted by Meta.");
    };
    for (bool bf16 : {true, false}) {
        const auto options = ints.dtype(bf16 ? at::kBFloat16 : at::kChar);
        const auto hbm = at::empty({2, 128, 1, bf16 ? 512 : 656}, options);
        const auto dram = at::empty_like(hbm);
        const OptionalTensor hbmKpe = bf16 ? OptionalTensor(at::empty({2, 128, 1, 64}, options)) : c10::nullopt;
        const OptionalTensor dramKpe = bf16 ? OptionalTensor(at::empty({2, 128, 1, 64}, options)) : c10::nullopt;
        const auto* owned = hbm.unsafeGetTensorImpl();
        op.call(hbm, dram, hbmKpe, dramKpe, table, table, ids, ids, counts);
        assert(hbm.unsafeGetTensorImpl() == owned);
        reject([&] { op.call(hbm, dram, bf16 ? c10::nullopt : OptionalTensor(hbm),
                            dramKpe, table, table, ids, ids, counts); });
        reject([&] { op.call(hbm, dram, hbmKpe, dramKpe, table, table,
                            at::empty({2, 128}, ints), ids, counts); });
        reject([&] { op.call(hbm, dram, hbmKpe, dramKpe, table, table,
                            ids, ids, at::empty({2}, ints.dtype(at::kLong))); });
    }
    std::cout << "dispatcher OK: PrivateUse1 + Meta, BF16/C8, aliases and input checks\n";
}
