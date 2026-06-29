// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/ort/external_weights_manager.h"

#include "base/notreached.h"
#include "services/webnn/ort/ort_status.h"
#include "services/webnn/ort/platform_functions_ort.h"

namespace webnn::ort {

ExternalWeightsManager::ExternalWeightsManager() {
  const OrtApi* ort_api = PlatformFunctions::GetInstance()->ort_api();
  CHECK_STATUS(ort_api->CreateCpuMemoryInfo(
      OrtDeviceAllocator, OrtMemTypeDefault,
      ScopedOrtMemoryInfo::Receiver(cpu_memory_info_).get()));

  OrtAllocator::version = ORT_API_VERSION;
  // `Info`, `Free`, `Alloc` and `Reserve` are function pointers. `Alloc` and
  // `Reserve` should never be called in the context of
  // `ExternalWeightsManager`. ORT will only call `Info` and `Free` to get the
  // memory info and free the external weights, respectively.
  OrtAllocator::Info =
      [](const OrtAllocator* this_allocator) -> const OrtMemoryInfo* {
    return static_cast<const ExternalWeightsManager*>(this_allocator)
        ->cpu_memory_info_.get();
  };
  OrtAllocator::Free = [](OrtAllocator* this_allocator, void* p) {
    static_cast<ExternalWeightsManager*>(this_allocator)->FreeImpl(p);
  };
  OrtAllocator::Alloc = [](OrtAllocator* /*this_allocator*/,
                           size_t /*size*/) -> void* {
    NOTREACHED() << "[WebNN] OrtAllocator::Alloc() should never be called.";
  };
  OrtAllocator::Reserve = [](OrtAllocator* /*this_allocator*/,
                             size_t /*size*/) -> void* {
    NOTREACHED() << "[WebNN] OrtAllocator::Reserve() should never be called.";
  };
}

// `external_weights_` may still hold entries at destruction when graph
// construction fails before a session is created (e.g.
// `GraphBuilderOrt::BuildModel()` returns an error after some initializers were
// registered): in that case ORT never took ownership of `this` as an allocator,
// so it will never call `FreeImpl()`, and the remaining `base::HeapArray`s
// simply free themselves as the set is destroyed. The dangerous case this used
// to guard against -- the session outliving the manager and ORT calling
// `FreeImpl()` on a destroyed object -- is instead prevented structurally: in
// `ComputeResources` the session is declared after (and therefore destroyed
// before) the manager.
ExternalWeightsManager::~ExternalWeightsManager() = default;

ScopedOrtValue ExternalWeightsManager::CreateInitializer(
    base::HeapArray<uint8_t> data,
    base::span<const int64_t> shape,
    ONNXTensorElementDataType data_type) {
  const OrtApi* ort_api = PlatformFunctions::GetInstance()->ort_api();
  ScopedOrtValue initializer;
  CHECK_STATUS(ort_api->CreateTensorWithDataAndDeleterAsOrtValue(
      this, data.data(), data.size(), shape.data(), shape.size(), data_type,
      ScopedOrtValue::Receiver(initializer).get()));
  external_weights_.insert(std::move(data));
  return initializer;
}

void ExternalWeightsManager::FreeImpl(void* p) {
  CHECK_EQ(external_weights_.erase(p), 1u);
}

size_t ExternalWeightsManager::ExternalWeightsHash::operator()(
    const base::HeapArray<uint8_t>& weights) const {
  return std::hash<const void*>{}(weights.data());
}

size_t ExternalWeightsManager::ExternalWeightsHash::operator()(
    const void* p) const {
  return std::hash<const void*>{}(p);
}

bool ExternalWeightsManager::ExternalWeightsEqual::operator()(
    const base::HeapArray<uint8_t>& lhs,
    const base::HeapArray<uint8_t>& rhs) const {
  return lhs.data() == rhs.data();
}

bool ExternalWeightsManager::ExternalWeightsEqual::operator()(
    const base::HeapArray<uint8_t>& lhs,
    const void* rhs) const {
  return lhs.data() == rhs;
}

}  // namespace webnn::ort
