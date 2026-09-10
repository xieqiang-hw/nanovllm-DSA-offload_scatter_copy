#pragma once
#include "kernel_operator.h"
#include "scatter_target.h"

// The build selects the overload supported by the target CANN/SoC.
__aicore__ inline void CopyFromGm(
    const AscendC::LocalTensor<uint8_t>& dst,
    const AscendC::GlobalTensor<uint8_t>& src, uint32_t bytes)
{
    AscendC::DataCopyExtParams params{1, bytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
#if SCATTER_A5
    AscendC::DataCopyPad<uint8_t, AscendC::PaddingMode::Normal>(dst, src, params, pad);
#else
    AscendC::DataCopyPad(dst, src, params, pad);
#endif
}

__aicore__ inline void CopyToGm(
    const AscendC::GlobalTensor<uint8_t>& dst,
    const AscendC::LocalTensor<uint8_t>& src, uint32_t bytes)
{
    AscendC::DataCopyExtParams params{1, bytes, 0, 0, 0};
#if SCATTER_A5
    AscendC::DataCopyPad<uint8_t, AscendC::PaddingMode::Normal>(dst, src, params);
#else
    AscendC::DataCopyPad(dst, src, params);
#endif
}
