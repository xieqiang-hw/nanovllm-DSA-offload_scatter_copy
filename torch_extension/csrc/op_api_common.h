/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
// Queue-safe ACLNN launch adapter; only dense tensor arguments are needed.
#include <cstdlib>
#include <dlfcn.h>
#include <tuple>
#include <vector>
#include <acl/acl_rt.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

struct aclTensor;
struct aclOpExecutor;
using OpApiFunc = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
using _aclCreateTensor = aclTensor* (*)(const int64_t*, uint64_t, aclDataType,
    const int64_t*, int64_t, aclFormat, const int64_t*, uint64_t, void*);
using _aclDestroyTensor = int (*)(const aclTensor*);
void* GetOpApiFuncAddr(const char* name);
inline const char* GetOpApiLibName() { return "repository-local libcust_opapi.so"; }
#define GET_OP_API_FUNC(name) reinterpret_cast<_##name>(GetOpApiFuncAddr(#name))

inline aclTensor* ConvertType(const at::Tensor& tensor)
{
    if (!tensor.defined()) return nullptr;
    static const auto create = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(create, "aclCreateTensor is unavailable.");
    aclDataType dtype;
    switch (tensor.scalar_type()) {
        case at::kBFloat16: dtype = ACL_BF16; break;
        case at::kChar: dtype = ACL_INT8; break;
        case at::kInt: dtype = ACL_INT32; break;
        default: TORCH_CHECK(false, "Unsupported scatter tensor dtype.");
    }
    // Preserve the reference adapter's base-format check without any casts/copies.
    const auto format = static_cast<torch_npu::NPUStorageImpl*>(tensor.storage().unsafeGetStorageImpl())->npu_desc_.npu_format_;
    TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW ||
                format == ACL_FORMAT_NHWC || format == ACL_FORMAT_NCDHW,
                "Scatter requires dense base-format tensors.");
    const int64_t storageElements = tensor.storage().nbytes() / tensor.element_size();
    auto* result = create(tensor.sizes().data(), tensor.dim(), dtype,
        tensor.strides().data(), tensor.storage_offset(), ACL_FORMAT_ND,
        &storageElements, 1, const_cast<void*>(tensor.storage().data()));
    TORCH_CHECK(result, "aclCreateTensor failed.");
    return result;
}
inline aclTensor* ConvertType(const c10::optional<at::Tensor>& tensor)
{
    return tensor.has_value() ? ConvertType(*tensor) : nullptr;
}
template <typename T> T ConvertType(T value) { return value; }
template <typename... T> auto ConvertTypes(T&... args) { return std::make_tuple(ConvertType(args)...); }
template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& args, void* address, std::index_sequence<I...>)
{
    using Function = int (*)(typename std::decay<decltype(std::get<I>(args))>::type...);
    return reinterpret_cast<Function>(address);
}
template <typename Tuple> auto ConvertToOpApiFunc(const Tuple& args, void* address)
{
    return ConvertToOpApiFunc(args, address, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}
template <typename Function, typename Tuple> auto call(Function function, const Tuple& args)
{
    return std::apply(function, args);
}
inline void Release(aclTensor* tensor)
{
    static const auto destroy = GET_OP_API_FUNC(aclDestroyTensor);
    if (tensor && destroy) destroy(tensor);
}
template <typename T> void Release(T) {}
template <typename Tuple> void ReleaseConvertTypes(const Tuple& args)
{
    std::apply([](auto... value) { (Release(value), ...); }, args);
}
using InitHugeMemThreadLocal = int (*)(void*, bool);
using UnInitHugeMemThreadLocal = void (*)(void*, bool);
using ReleaseHugeMem = void (*)(void*, bool);
using InitPTACacheThreadLocal = void (*)();
using SetPTAHashKey = void (*)(uint64_t);
inline void UnInitCacheThreadLocal()
{
    static const auto function = reinterpret_cast<void (*)()>(GetOpApiFuncAddr("UnInitPTACacheThreadLocal"));
    if (function) function();
}
using AclUseStreamResFunc = void (*)(void*);
AclUseStreamResFunc GetUseStreamResFuncCoreNum();

#define EXEC_NPU_CMD_ORDERED(aclnn_api, tensor_keepalive, ...)                                                         \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    "not found.");                                                                                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        if (c10_npu::check_enqueue_need_use(acl_stream)) {                                                             \
            auto useStreamResFunc = GetUseStreamResFuncCoreNum();                                                      \
            TORCH_CHECK(useStreamResFunc != nullptr, "aclrtUseStreamResInCurrentThread is unavailable.");            \
            useStreamResFunc(acl_stream);                                                                              \
        }                                                                                                              \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                          \
            reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                    \
        SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);                          \
        if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                        \
            initPTACacheThreadLocalFunc();                                                                             \
            setPTAHashKeyFunc(0);                                                                                      \
        }                                                                                                              \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        TORCH_CHECK(workspace_status == 0, "call " #aclnn_api "GetWorkspaceSize failed, detail:",                   \
                    aclGetRecentErrMsg());                                                                             \
        void *workspace_addr = nullptr;                                                                                \
        at::Tensor workspace_tensor;                                                                                   \
        if (workspace_size != 0) {                                                                                     \
            at::TensorOptions options =                                                                                \
                at::TensorOptions(torch_npu::utils::get_npu_device_type());                                            \
            workspace_tensor =                                                                                         \
                at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(at::kByte));                           \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto ordered_tensor_keepalive = tensor_keepalive;                                                              \
        auto acl_call = [converted_params, workspace_tensor, workspace_addr, workspace_size, acl_stream, executor,     \
                         ordered_tensor_keepalive]()->int {                                                            \
            (void)workspace_tensor;                                                                                    \
            (void)ordered_tensor_keepalive;                                                                            \
            if (c10_npu::check_dequeue_need_use(acl_stream)) {                                                         \
                auto useStreamResFunc = GetUseStreamResFuncCoreNum();                                                  \
                TORCH_CHECK(useStreamResFunc != nullptr, "aclrtUseStreamResInCurrentThread is unavailable.");        \
                useStreamResFunc(acl_stream);                                                                          \
            }                                                                                                          \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());                 \
            ReleaseConvertTypes(converted_params);                                                                     \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                          \
            if (releaseMemFunc) {                                                                                      \
                releaseMemFunc(nullptr, false);                                                                        \
            }                                                                                                          \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);                                                   \
        if (unInitMemFunc) {                                                                                           \
            unInitMemFunc(nullptr, false);                                                                             \
        }                                                                                                              \
        UnInitCacheThreadLocal();                                                                                      \
    } while (false)
