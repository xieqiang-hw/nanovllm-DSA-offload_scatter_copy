#include <cstdint>
#include <initializer_list>
#include <limits>
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/kvcache_scatter_copy_tiling.h"
#include "scatter_target.h"

namespace optiling {
static bool CacheShape(const gert::Shape& s, int64_t width)
{
    return s.GetDimNum() == 4 && s.GetDim(0) > 0 &&
        s.GetDim(0) <= std::numeric_limits<uint32_t>::max() &&
        s.GetDim(1) == 128 && s.GetDim(2) == 1 && s.GetDim(3) == width;
}

static ge::graphStatus TilingScatter(gert::TilingContext* context)
{
    if (!context || !context->GetPlatformInfo()) return ge::GRAPH_FAILED;
    for (size_t i = 0; i <= 8; ++i) {
        if (!context->GetRequiredInputShape(i) || !context->GetRequiredInputDesc(i))
            return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetRequiredInputDesc(0)->GetDataType();
    const bool bf16 = dtype == ge::DT_BF16;
    if ((!bf16 && dtype != ge::DT_INT8) ||
        context->GetRequiredInputDesc(1)->GetDataType() != dtype) return ge::GRAPH_FAILED;
    const auto& hbm = context->GetRequiredInputShape(0)->GetStorageShape();
    const auto& dram = context->GetRequiredInputShape(1)->GetStorageShape();
    if (!CacheShape(hbm, bf16 ? 512 : 656) || !CacheShape(dram, bf16 ? 512 : 656))
        return ge::GRAPH_FAILED;
    for (size_t i : {2U, 3U}) {
        const auto& shape = context->GetRequiredInputShape(i)->GetStorageShape();
        // C8 uses KV aliases as unused KPE placeholders; BF16 uses real KPE caches.
        if (context->GetRequiredInputDesc(i)->GetDataType() != dtype ||
            !CacheShape(shape, bf16 ? 64 : 656) ||
            shape.GetDim(0) != (i == 2 ? hbm : dram).GetDim(0)) return ge::GRAPH_FAILED;
    }
    for (size_t i = 4; i <= 8; ++i)
        if (context->GetRequiredInputDesc(i)->GetDataType() != ge::DT_INT32)
            return ge::GRAPH_FAILED;
    const auto& counts = context->GetRequiredInputShape(8)->GetStorageShape();
    if (counts.GetDimNum() != 1 || counts.GetDim(0) <= 0 ||
        counts.GetDim(0) > std::numeric_limits<uint32_t>::max()) return ge::GRAPH_FAILED;
    const int64_t batch = counts.GetDim(0);
    const auto& src = context->GetRequiredInputShape(6)->GetStorageShape();
    const auto& dst = context->GetRequiredInputShape(7)->GetStorageShape();
    for (const auto* s : {&src, &dst}) {
        if (s->GetDimNum() != 3 || s->GetDim(0) != batch || s->GetDim(1) != 1 ||
            s->GetDim(2) < 1 || s->GetDim(2) > 65536) return ge::GRAPH_FAILED;
    }
    if (src.GetDim(2) != dst.GetDim(2)) return ge::GRAPH_FAILED;
    const auto& hbmTable = context->GetRequiredInputShape(4)->GetStorageShape();
    const auto& dramTable = context->GetRequiredInputShape(5)->GetStorageShape();
    for (const auto* s : {&hbmTable, &dramTable}) {
        if (s->GetDimNum() != 2 || s->GetDim(0) != batch || s->GetDim(1) <= 0 ||
            s->GetDim(1) > (int64_t(1) << 24)) return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aiv = platform.GetCoreNumAiv();
    if (aiv == 0) return ge::GRAPH_FAILED;
    const uint64_t slots = uint64_t(batch) * src.GetDim(2);
    auto* t = context->GetTilingData<KvcacheScatterCopyTilingData>();
    if (!t) return ge::GRAPH_FAILED;
    *t = {uint32_t(slots < aiv ? slots : aiv), uint32_t(batch), uint32_t(src.GetDim(2)),
          uint32_t(hbmTable.GetDim(1)), uint32_t(dramTable.GetDim(1)),
          uint32_t(hbm.GetDim(0)), uint32_t(dram.GetDim(0)), 0, slots};
    context->SetBlockDim(t->usedCoreNum);
    context->SetTilingKey(bf16 ? 1 : 2);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

struct ScatterCompileInfo {};
static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(KvcacheScatterCopy).Tiling(TilingScatter).TilingParse<ScatterCompileInfo>(TilingParse);
} // namespace optiling

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (!context) return ge::GRAPH_FAILED;
    for (size_t output = 0; output < 2; ++output) {
        const auto* input = context->GetRequiredInputShape(output * 2);
        auto* shape = context->GetOutputShape(output);
        if (!input || !shape) return ge::GRAPH_FAILED;
        *shape = *input;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    if (!context) return ge::GRAPH_FAILED;
    const auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvcacheScatterCopy).InferShape(InferShape).InferDataType(InferDtype);

class KvcacheScatterCopy : public OpDef {
public:
    explicit KvcacheScatterCopy(const char* name) : OpDef(name)
    {
        // Required references avoid CANN's duplicate optional-inout ACLNN parameters.
        // The public Torch API still accepts None for C8 KPE; its adapter supplies KV aliases.
        for (const char* input : {"hbm_kv", "dram_kv", "hbm_kpe", "dram_kpe"})
            this->Input(input).ParamType(REQUIRED)
                .DataType({ge::DT_BF16, ge::DT_INT8}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const char* input : {"hbm_block_table", "dram_block_table", "source_token_ids",
                                 "destination_slots", "copy_counts"})
            this->Input(input).ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // Matching input/output names generate a caller-owned ACLNN reference ABI.
        for (const char* output : {"hbm_kv", "hbm_kpe"})
            this->Output(output).ParamType(REQUIRED)
                .DataType({ge::DT_BF16, ge::DT_INT8}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig(SCATTER_SOC);
    }
};
OP_ADD(KvcacheScatterCopy);
} // namespace ops
