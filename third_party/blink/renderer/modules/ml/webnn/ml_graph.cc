// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"

#include "base/task/single_thread_task_runner.h"
#include "base/types/expected_macros.h"
#include "services/webnn/public/mojom/webnn_graph.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_device_type.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_dynamic_dimension.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_input_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_mldynamicdimension_unsignedlongenforcerange.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_tensor.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

namespace {

#define THROW_AND_RETURN_IF_ERROR(func, msg)                      \
  RETURN_IF_ERROR(func, [&exception_state](const String& error) { \
    exception_state.ThrowTypeError(StrCat({msg, error}));         \
    return;                                                       \
  });

// NOTE: Tensor validation (shape + dynamic-dimension consistency) lives in
// `MLContext::ValidateNamedMLTensors` (blink layer) and
// `webnn::WebNNContextImpl::Dispatch` (service layer). The legacy duplicate
// helpers that previously lived in this anonymous namespace have been
// removed; do not re-introduce them here without a clear reason.

}  // namespace

MLGraph::MLGraph(
    ExecutionContext* execution_context,
    MLContext* context,
    mojo::PendingRemote<webnn::mojom::blink::WebNNGraph> pending_graph_remote,
    blink::WebNNGraphToken graph_token,
    NamedOperandDescriptors input_constraints,
    NamedOperandDescriptors output_constraints,
    Vector<V8MLDeviceType> devices,
    base::PassKey<MLGraphBuilder> /*pass_key*/)
    : input_constraints_(std::move(input_constraints)),
      output_constraints_(std::move(output_constraints)),
      ml_context_(context),
      graph_token_(graph_token),
      remote_graph_(execution_context),
      devices_(std::move(devices)) {
  // Bind the end point of `WebNNGraph` mojo interface in the blink side.
  remote_graph_.Bind(
      std::move(pending_graph_remote),
      execution_context->GetTaskRunner(TaskType::kMachineLearning));
  remote_graph_.set_disconnect_handler(
      BindOnce(&MLGraph::OnConnectionError, WrapWeakPersistent(this)));
}

MLGraph::~MLGraph() = default;

void MLGraph::Dispose() {
  // When GC collects the graph without an explicit destroy() call, send
  // DestroyGraph through the context pipe to ensure ordering with
  // Dispatch/ReadTensor/WriteTensor. If destroy() was already called,
  // the remote is unbound and this is a no-op.
  if (!IsDestroyed()) {
    ml_context_->DestroyGraph(graph_token_);
    OnConnectionError();
  }
}

void MLGraph::Trace(Visitor* visitor) const {
  visitor->Trace(ml_context_);
  visitor->Trace(remote_graph_);
  ScriptWrappable::Trace(visitor);
}

void MLGraph::destroy() {
  // Delegate to Dispose(), which sends DestroyGraph through the context pipe
  // to ensure ordering with Dispatch/ReadTensor/WriteTensor.
  Dispose();
}

bool MLGraph::IsDestroyed() const {
  return !remote_graph_.is_bound();
}

Vector<V8MLDeviceType> MLGraph::devices() const {
  return devices_;
}

namespace {

MLInputOperandDescriptor* MakeInputOperandDescriptor(
    const webnn::OperandDescriptor& descriptor) {
  auto* desc = MLInputOperandDescriptor::Create();
  desc->setDataType(ToBlinkDataType(descriptor.data_type()));
  HeapVector<Member<V8UnionMLDynamicDimensionOrUnsignedLongEnforceRange>> shape;
  for (const auto& dim : descriptor.shape()) {
    if (std::holds_alternative<uint32_t>(dim)) {
      shape.push_back(MakeGarbageCollected<
                      V8UnionMLDynamicDimensionOrUnsignedLongEnforceRange>(
          std::get<uint32_t>(dim)));
    } else {
      const auto& dynamic_dim = std::get<webnn::DynamicDimension>(dim);
      auto* dyn = MLDynamicDimension::Create();
      dyn->setName(String::FromUtf8(dynamic_dim.name));
      if (dynamic_dim.max_size.has_value()) {
        dyn->setMaxSize(*dynamic_dim.max_size);
      }
      dyn->setMinSize(dynamic_dim.min_size);
      shape.push_back(
          MakeGarbageCollected<
              V8UnionMLDynamicDimensionOrUnsignedLongEnforceRange>(dyn));
    }
  }
  desc->setShape(std::move(shape));
  return desc;
}

}  // namespace

HeapVector<std::pair<String, Member<MLInputOperandDescriptor>>>
MLGraph::inputs() const {
  HeapVector<std::pair<String, Member<MLInputOperandDescriptor>>> result;
  for (const auto& [name, descriptor] : input_constraints_) {
    CHECK(descriptor.has_value());
    result.emplace_back(name, MakeInputOperandDescriptor(*descriptor));
  }
  return result;
}

HeapVector<std::pair<String, Member<MLInputOperandDescriptor>>>
MLGraph::outputs() const {
  HeapVector<std::pair<String, Member<MLInputOperandDescriptor>>> result;
  for (const auto& [name, descriptor] : output_constraints_) {
    CHECK(descriptor.has_value());
    result.emplace_back(name, MakeInputOperandDescriptor(*descriptor));
  }
  return result;
}

const MLGraph::NamedOperandDescriptors& MLGraph::GetInputConstraints() const {
  return input_constraints_;
}

const MLGraph::NamedOperandDescriptors& MLGraph::GetOutputConstraints() const {
  return output_constraints_;
}

const MLContext* MLGraph::Context() const {
  return ml_context_.Get();
}

void MLGraph::OnConnectionError() {
  remote_graph_.reset();
}

}  // namespace blink
