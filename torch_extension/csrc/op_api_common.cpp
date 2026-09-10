/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "op_api_common.h"
#include <cstring>

void* GetOpApiFuncAddr(const char* name)
{
    static void* custom = [] {
        const char* path = std::getenv("SCATTER_OPAPI_LIB");
        TORCH_CHECK(path, "Import kvcache_ops before calling scatter.");
        void* handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
        TORCH_CHECK(handle, "Cannot load ", path, ": ", dlerror());
        return handle;
    }();
    if (void* function = dlsym(custom, name)) return function;
    // Never pick a same-named scatter operator out of another installed OPP.
    TORCH_CHECK(std::strncmp(name, "aclnnKvcacheScatterCopy", 23) != 0,
                "The local OPP is missing ", name, ". Rebuild with bash build.sh.");
    static const std::vector<void*> libraries = [] {
        std::vector<void*> result;
        for (const char* library : {"libopapi.so", "libaclnn_ops_infer.so", "libaclnn_ops_train.so",
             "libaclnn_math.so", "libopapi_math.so", "libopapi_nn.so", "libopapi_cv.so",
             "libopapi_transformer.so", "libopapi_legacy.so"})
            if (void* handle = dlopen(library, RTLD_LAZY | RTLD_LOCAL)) result.push_back(handle);
        return result;
    }();
    for (void* library : libraries)
        if (void* function = dlsym(library, name)) return function;
    return dlsym(RTLD_DEFAULT, name);
}

AclUseStreamResFunc GetUseStreamResFuncCoreNum()
{
    static void* library = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
    return library ? reinterpret_cast<AclUseStreamResFunc>(
        dlsym(library, "aclrtUseStreamResInCurrentThread")) : nullptr;
}
