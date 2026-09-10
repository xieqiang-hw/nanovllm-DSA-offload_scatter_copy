#pragma once
#include <cstdint>

struct KvcacheScatterCopyTilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t copyCap;
    uint32_t hbmMaxBlockNum;
    uint32_t dramMaxBlockNum;
    uint32_t hbmPhysicalBlockCount;
    uint32_t dramPhysicalBlockCount;
    uint32_t reserved;
    uint64_t totalPairSlots;
};
