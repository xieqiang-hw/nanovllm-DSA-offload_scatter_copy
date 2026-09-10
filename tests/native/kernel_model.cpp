#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include "../../csrc/src/kvcache_scatter_copy/op_kernel/kvcache_scatter_copy.cpp"

template <typename T> uint8_t* Register(std::vector<T>& values) {
    AscendC::regions.emplace_back(reinterpret_cast<uintptr_t>(values.data()), values.size() * sizeof(T));
    return reinterpret_cast<uint8_t*>(values.data());
}

void Check(bool bf16, uint32_t cap, uint32_t cores, std::vector<int32_t> counts, bool full = false) {
    AscendC::regions.clear();
    const uint32_t batch = counts.size();
    const uint32_t slots = full ? cap : 259;
    const uint32_t blocks = (slots + 127) / 128;
    const uint32_t sourceBlocks = batch * blocks, targetBlocks = batch * blocks + 1;
    const uint32_t width = bf16 ? 1024 : 656;
    std::mt19937 rng(37);
    std::vector<int32_t> hbmTable(batch * blocks), dramTable(batch * blocks);
    std::iota(hbmTable.begin(), hbmTable.end(), 0); std::iota(dramTable.begin(), dramTable.end(), 0);
    std::shuffle(hbmTable.begin(), hbmTable.end(), rng); std::shuffle(dramTable.begin(), dramTable.end(), rng);
    std::vector<int32_t> src(batch * cap, -1), dst(batch * cap, -1);
    std::vector<uint8_t> source(sourceBlocks * 128ULL * width), target(targetBlocks * 128ULL * width, 65);
    std::vector<uint8_t> sourceRope(bf16 ? sourceBlocks * 128ULL * 128 : 0), targetRope(bf16 ? targetBlocks * 128ULL * 128 : 0, 65);
    for (auto& byte : source) byte = rng();
    for (auto& byte : sourceRope) byte = rng();
    auto expected = target, expectedRope = targetRope;
    for (uint32_t b = 0; b < batch; ++b) {
        std::vector<int32_t> destinations(slots); std::iota(destinations.begin(), destinations.end(), 0);
        std::shuffle(destinations.begin(), destinations.end(), rng);
        for (int32_t i = 0; i < counts[b]; ++i) {
            // Include repeated sources, adjacent destinations and both sides of block boundaries.
            src[b * cap + i] = i < 4 ? (i % 2 ? 128 : 127) % slots : rng() % slots;
            dst[b * cap + i] = full ? i : destinations[i];
            const uint32_t s = src[b * cap + i], d = dst[b * cap + i];
            const uint64_t srow = dramTable[b * blocks + s / 128] * 128ULL + s % 128;
            const uint64_t drow = hbmTable[b * blocks + d / 128] * 128ULL + d % 128;
            std::memcpy(expected.data() + drow * width, source.data() + srow * width, width);
            if (bf16) std::memcpy(expectedRope.data() + drow * 128, sourceRope.data() + srow * 128, 128);
        }
    }
    const auto sourceBefore = source, sourceRopeBefore = sourceRope;
    const auto countsBefore = counts, srcBefore = src, dstBefore = dst;
    const auto hbmBefore = hbmTable, dramBefore = dramTable;
    auto* hbm = Register(target); auto* dram = Register(source);
    // Match the ACLNN adapter: C8 supplies KV aliases in the unused KPE slots.
    auto* hbmKpe = bf16 ? Register(targetRope) : hbm;
    auto* dramKpe = bf16 ? Register(sourceRope) : dram;
    auto* hb = Register(hbmTable); auto* db = Register(dramTable);
    auto* s = Register(src); auto* d = Register(dst); auto* c = Register(counts);
    KvcacheScatterCopyTilingData t{std::min(cores, batch * cap), batch, cap, blocks, blocks, targetBlocks, sourceBlocks, 0, uint64_t(batch) * cap};
    AscendC::tilingKey = bf16 ? 1 : 2;
    for (AscendC::coreIndex = 0; AscendC::coreIndex < t.usedCoreNum; ++AscendC::coreIndex)
        kvcache_scatter_copy(hbm, dram, hbmKpe, dramKpe, hb, db, s, d, c,
                            hbm, hbmKpe, nullptr, reinterpret_cast<uint8_t*>(&t));
    assert(target == expected && targetRope == expectedRope);
    assert(source == sourceBefore && sourceRope == sourceRopeBefore);
    assert(src == srcBefore && dst == dstBefore && counts == countsBefore);
    assert(hbmTable == hbmBefore && dramTable == dramBefore);
}

int main() {
    for (bool bf16 : {false, true}) {
        for (uint32_t cores : {1, 2, 3, 48, 64}) {
            Check(bf16, 1, cores, {0, 1, 0});
            Check(bf16, 129, cores, {0, 1, 129});
            Check(bf16, 2048, cores, {0, 200, 0, 3});
            Check(bf16, 16384, cores, {1, 0, 259});
            Check(bf16, 65536, cores, {0, 0, 0});
        }
        Check(bf16, 16384, 48, {16384}, true);
        Check(bf16, 65536, 64, {65536}, true);
    }
    std::cout << "kernel model OK: 54 cases, byte exact and no out-of-range accesses\n";
}
