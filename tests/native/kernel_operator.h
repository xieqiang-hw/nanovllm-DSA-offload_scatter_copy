// CPU memory/queue model for testing actual kernel address logic. Not a CANN emulator.
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <vector>
#include "scatter_target.h"

#define __aicore__
#define __global__
#define __gm__
#define GM_ADDR uint8_t*
#define ASSERT_MSG(test, message) do { if (!(test)) throw std::runtime_error(message); } while (0)
#define KERNEL_TASK_TYPE_DEFAULT(...)
#define REGISTER_TILING_DEFAULT(...)
#define GET_TILING_DATA(name, pointer) auto& name = *reinterpret_cast<KvcacheScatterCopyTilingData*>(pointer)
#define TILING_KEY_IS(key) (AscendC::tilingKey == (key))

namespace AscendC {
inline uint32_t coreIndex, tilingKey;
inline uint32_t GetBlockIdx() { return coreIndex; }
enum class QuePosition { VECIN, VECOUT, VECCALC };
enum class PaddingMode { Normal };
inline std::vector<std::pair<uintptr_t, size_t>> regions;
inline void CheckRange(const void* pointer, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    for (const auto& region : regions)
        if (address >= region.first && address - region.first <= region.second &&
            bytes <= region.second - (address - region.first)) return;
    throw std::runtime_error("GM access outside an allocated tensor");
}
template <typename T> struct GlobalTensor {
    T* pointer{};
    void SetGlobalBuffer(T* value) { pointer = value; }
    GlobalTensor operator[](uint64_t offset) const { return {pointer + offset}; }
    T GetValue(uint64_t offset) const { CheckRange(pointer + offset, sizeof(T)); return pointer[offset]; }
};
template <typename T> struct LocalTensor {
    std::shared_ptr<std::vector<uint8_t>> bytes;
    size_t offset = 0;
    T* data() const { return reinterpret_cast<T*>(bytes->data() + offset); }
    LocalTensor operator[](uint64_t index) const { return {bytes, offset + index * sizeof(T)}; }
    void Check(size_t count) const { assert(offset <= bytes->size() && count <= bytes->size() - offset); }
    T GetValue(uint64_t index) const { (*this)[index].Check(sizeof(T)); return data()[index]; }
    void SetValue(uint64_t index, T value) { (*this)[index].Check(sizeof(T)); data()[index] = value; }
};
template <QuePosition, QuePosition, unsigned Depth> struct TQueBind {
    std::vector<std::shared_ptr<std::vector<uint8_t>>> buffers;
    std::vector<bool> used;
    std::deque<LocalTensor<uint8_t>> pending;
    void Init(unsigned depth, size_t bytes) {
        assert(depth == Depth);
        for (unsigned i = 0; i < depth; ++i) buffers.push_back(std::make_shared<std::vector<uint8_t>>(bytes, 0xCD));
        used.resize(depth);
    }
    template <typename T> LocalTensor<T> AllocTensor() {
        for (unsigned i = 0; i < Depth; ++i) if (!used[i]) { used[i] = true; return {buffers[i], 0}; }
        throw std::runtime_error("Queue buffer reused before copy-out completed");
    }
    template <typename T> void EnQue(LocalTensor<T> tensor) { pending.push_back({tensor.bytes, tensor.offset}); }
    template <typename T> LocalTensor<T> DeQue() {
        if (pending.empty()) throw std::runtime_error("Empty queue");
        auto front = pending.front(); pending.pop_front(); return {front.bytes, front.offset};
    }
    template <typename T> void FreeTensor(LocalTensor<T> tensor) {
        for (unsigned i = 0; i < Depth; ++i) if (buffers[i] == tensor.bytes) { assert(used[i]); used[i] = false; return; }
        throw std::runtime_error("Unknown queue allocation");
    }
    ~TQueBind() { assert(pending.empty()); for (bool value : used) assert(!value); }
};
template <QuePosition> struct TBuf {
    std::shared_ptr<std::vector<uint8_t>> bytes;
    void Init(size_t size) { bytes = std::make_shared<std::vector<uint8_t>>(size); }
    template <typename T> LocalTensor<T> Get() { return {bytes, 0}; }
};
struct TPipe {
    template <typename Q> void InitBuffer(Q& queue, unsigned count, size_t bytes) { queue.Init(count, bytes); }
    template <typename B> void InitBuffer(B& buffer, size_t bytes) { buffer.Init(bytes); }
};
struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen, srcStride, dstStride, reserved; };
template <typename T> struct DataCopyPadExtParams { bool isPad; uint8_t left, right; T value; };
template <typename T> void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                      const DataCopyExtParams& params, const DataCopyPadExtParams<T>&) {
    assert(params.blockCount == 1); dst.Check(params.blockLen); CheckRange(src.pointer, params.blockLen);
    std::memcpy(dst.data(), src.pointer, params.blockLen);
}
template <typename T> void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                      const DataCopyExtParams& params) {
    assert(params.blockCount == 1); src.Check(params.blockLen); CheckRange(dst.pointer, params.blockLen);
    std::memcpy(dst.pointer, src.data(), params.blockLen);
}
#if SCATTER_A5
template <typename T, PaddingMode> void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                                   const DataCopyExtParams& p, const DataCopyPadExtParams<T>& pad) {
    DataCopyPad(dst, src, p, pad);
}
template <typename T, PaddingMode> void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                                   const DataCopyExtParams& p) { DataCopyPad(dst, src, p); }
#endif
} // namespace AscendC
