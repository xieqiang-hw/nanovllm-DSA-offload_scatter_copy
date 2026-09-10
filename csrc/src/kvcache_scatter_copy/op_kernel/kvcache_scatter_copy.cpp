/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * Derived from the BF16 and packed-C8 scatter implementations in this repository.
 */
#include "copy_compat.h"
#include "kvcache_scatter_copy_tiling.h"

namespace {
using namespace AscendC;
constexpr uint32_t BLOCK_SHIFT = 7;
constexpr uint32_t BLOCK_MASK = 127;
constexpr uint32_t QUEUE_DEPTH = 2;

template <bool BF16>
class ScatterKernel {
    static constexpr uint32_t KV_BYTES = BF16 ? 1024 : 656;
    static constexpr uint32_t BUFFER_BYTES = BF16 ? 1152 : 672;
    struct Address {
        uint64_t srcKv = 0, dstKv = 0, srcRope = 0, dstRope = 0;
    };

public:
    __aicore__ inline ScatterKernel(TPipe* pipe, const KvcacheScatterCopyTilingData* t)
        : pipe_(pipe), t_(t) {}

    __aicore__ inline void Init(
        GM_ADDR hbmKv, GM_ADDR dramKv, GM_ADDR hbmKpe, GM_ADDR dramKpe,
        GM_ADDR hbmTable, GM_ADDR dramTable, GM_ADDR srcIds,
        GM_ADDR dstSlots, GM_ADDR counts)
    {
        core_ = GetBlockIdx();
        pipe_->InitBuffer(queue_, QUEUE_DEPTH, BUFFER_BYTES);
        hbmKv_.SetGlobalBuffer((__gm__ uint8_t*)hbmKv);
        dramKv_.SetGlobalBuffer((__gm__ uint8_t*)dramKv);
        if constexpr (BF16) {
            hbmKpe_.SetGlobalBuffer((__gm__ uint8_t*)hbmKpe);
            dramKpe_.SetGlobalBuffer((__gm__ uint8_t*)dramKpe);
        } else {
            pipe_->InitBuffer(srcRingBuffer_, QUEUE_DEPTH * sizeof(uint64_t));
            pipe_->InitBuffer(dstRingBuffer_, QUEUE_DEPTH * sizeof(uint64_t));
            srcRing_ = srcRingBuffer_.Get<uint64_t>();
            dstRing_ = dstRingBuffer_.Get<uint64_t>();
        }
        hbmTable_.SetGlobalBuffer((__gm__ int32_t*)hbmTable);
        dramTable_.SetGlobalBuffer((__gm__ int32_t*)dramTable);
        srcIds_.SetGlobalBuffer((__gm__ int32_t*)srcIds);
        dstSlots_.SetGlobalBuffer((__gm__ int32_t*)dstSlots);
        counts_.SetGlobalBuffer((__gm__ int32_t*)counts);
    }

    __aicore__ inline void Process()
    {
        if (core_ >= t_->usedCoreNum) return;
        if constexpr (BF16) ProcessBf16();
        else ProcessC8();
    }

private:
    __aicore__ inline uint64_t FirstOwned(uint64_t start) const
    {
        if (start <= core_) return core_;
        return core_ + ((start - core_ + t_->usedCoreNum - 1) /
                        t_->usedCoreNum) * t_->usedCoreNum;
    }

    __aicore__ inline uint64_t FindNext(uint64_t pair)
    {
        while (pair < t_->totalPairSlots) {
            const uint32_t batch = static_cast<uint32_t>(pair / t_->copyCap);
            const uint32_t index = static_cast<uint32_t>(pair - uint64_t(batch) * t_->copyCap);
            if (batch != cachedBatch_) {
                cachedCount_ = counts_.GetValue(batch);
                ASSERT_MSG(cachedCount_ >= 0 && cachedCount_ <= int32_t(t_->copyCap),
                           "copy_counts must be in [0, C]");
                if (cachedCount_ < 0) cachedCount_ = 0;
                if (cachedCount_ > int32_t(t_->copyCap)) cachedCount_ = t_->copyCap;
                // Preserve the C8 row-base cache; BF16 shares the same lookup.
                dramBase_ = uint64_t(batch) * t_->dramMaxBlockNum;
                hbmBase_ = uint64_t(batch) * t_->hbmMaxBlockNum;
                cachedBatch_ = batch;
            }
            if (index < uint32_t(cachedCount_)) return pair;
            pair = FirstOwned((uint64_t(batch) + 1) * t_->copyCap);
        }
        return t_->totalPairSlots;
    }

    __aicore__ inline bool Resolve(uint64_t pair, Address& address)
    {
        const int32_t src = srcIds_.GetValue(pair);
        const int32_t dst = dstSlots_.GetValue(pair);
        if (src < 0 || dst < 0) return false;
        const uint32_t srcCol = uint32_t(src) >> BLOCK_SHIFT;
        const uint32_t dstCol = uint32_t(dst) >> BLOCK_SHIFT;
        if (srcCol >= t_->dramMaxBlockNum || dstCol >= t_->hbmMaxBlockNum) return false;
        const int32_t srcBlock = dramTable_.GetValue(dramBase_ + srcCol);
        const int32_t dstBlock = hbmTable_.GetValue(hbmBase_ + dstCol);
        if (srcBlock < 0 || dstBlock < 0 ||
            uint32_t(srcBlock) >= t_->dramPhysicalBlockCount ||
            uint32_t(dstBlock) >= t_->hbmPhysicalBlockCount) return false;
        const uint64_t srcRow = (uint64_t(srcBlock) << BLOCK_SHIFT) + (uint32_t(src) & BLOCK_MASK);
        const uint64_t dstRow = (uint64_t(dstBlock) << BLOCK_SHIFT) + (uint32_t(dst) & BLOCK_MASK);
        address.srcKv = srcRow * KV_BYTES;
        address.dstKv = dstRow * KV_BYTES;
        if constexpr (BF16) {
            address.srcRope = srcRow * 128;
            address.dstRope = dstRow * 128;
        }
        return true;
    }

    __aicore__ inline void CopyIn(const Address& address)
    {
        LocalTensor<uint8_t> local = queue_.AllocTensor<uint8_t>();
        CopyFromGm(local, dramKv_[address.srcKv], KV_BYTES);
        if constexpr (BF16) CopyFromGm(local[1024], dramKpe_[address.srcRope], 128);
        queue_.EnQue<uint8_t>(local);
    }

    __aicore__ inline void CopyOut(const Address& address)
    {
        LocalTensor<uint8_t> local = queue_.DeQue<uint8_t>();
        CopyToGm(hbmKv_[address.dstKv], local, KV_BYTES);
        if constexpr (BF16) CopyToGm(hbmKpe_[address.dstRope], local[1024], 128);
        queue_.FreeTensor(local);
    }

    __aicore__ inline uint64_t NextResolved(uint64_t pair, Address& address)
    {
        pair = FindNext(pair);
        while (pair < t_->totalPairSlots && !Resolve(pair, address))
            pair = FindNext(pair + t_->usedCoreNum);
        return pair;
    }

    __aicore__ inline void ProcessBf16()
    {
        Address current;
        uint64_t pair = NextResolved(core_, current);
        if (pair >= t_->totalPairSlots) return;
        CopyIn(current);
        while (true) {
            Address next;
            const uint64_t nextPair = NextResolved(pair + t_->usedCoreNum, next);
            const bool hasNext = nextPair < t_->totalPairSlots;
            // Keep the original next-read-before-current-write ordering.
            if (hasNext) CopyIn(next);
            CopyOut(current);
            if (!hasNext) break;
            pair = nextPair;
            current = next;
        }
    }

    __aicore__ inline void ProcessC8()
    {
        uint64_t next = FindNext(core_);
        uint32_t write = 0, read = 0, inflight = 0;
        while (inflight > 0 || next < t_->totalPairSlots) {
            if (inflight == QUEUE_DEPTH || next >= t_->totalPairSlots) {
                if (inflight == 0) break;
                Address out;
                out.srcKv = srcRing_.GetValue(read);
                out.dstKv = dstRing_.GetValue(read);
                CopyOut(out);
                read = (read + 1) % QUEUE_DEPTH;
                --inflight;
                continue;
            }
            Address in;
            if (Resolve(next, in)) {
                srcRing_.SetValue(write, in.srcKv);
                dstRing_.SetValue(write, in.dstKv);
                CopyIn(in);
                ++inflight;
                write = (write + 1) % QUEUE_DEPTH;
            }
            next = FindNext(next + t_->usedCoreNum);
        }
    }

    TPipe* pipe_;
    const KvcacheScatterCopyTilingData* t_;
    uint32_t core_ = 0, cachedBatch_ = uint32_t(-1);
    int32_t cachedCount_ = 0;
    uint64_t dramBase_ = 0, hbmBase_ = 0;
    GlobalTensor<uint8_t> hbmKv_, dramKv_, hbmKpe_, dramKpe_;
    GlobalTensor<int32_t> hbmTable_, dramTable_, srcIds_, dstSlots_, counts_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, QUEUE_DEPTH> queue_;
    TBuf<QuePosition::VECCALC> srcRingBuffer_, dstRingBuffer_;
    LocalTensor<uint64_t> srcRing_, dstRing_;
};
} // namespace

extern "C" __global__ __aicore__ void kvcache_scatter_copy(
    GM_ADDR hbmKv, GM_ADDR dramKv, GM_ADDR hbmKpe, GM_ADDR dramKpe,
    GM_ADDR hbmTable, GM_ADDR dramTable, GM_ADDR srcIds, GM_ADDR dstSlots,
    GM_ADDR counts, GM_ADDR hbmKvOut, GM_ADDR hbmKpeOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(KvcacheScatterCopyTilingData);
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        ScatterKernel<true> op(&pipe, &data);
        op.Init(hbmKv, dramKv, hbmKpe, dramKpe, hbmTable, dramTable, srcIds, dstSlots, counts);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ScatterKernel<false> op(&pipe, &data);
        op.Init(hbmKv, dramKv, hbmKpe, dramKpe, hbmTable, dramTable, srcIds, dstSlots, counts);
        op.Process();
    }
}
