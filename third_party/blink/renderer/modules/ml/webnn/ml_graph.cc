// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"

#include "base/numerics/safe_conversions.h"
#include "base/task/single_thread_task_runner.h"
#include "base/types/expected.h"
#include "base/types/expected_macros.h"
#include "services/webnn/public/mojom/webnn_graph.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_union_string_unsignedlongenforcerange.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_device_type.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_input_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
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
  HeapVector<Member<V8UnionStringOrUnsignedLongEnforceRange>> shape;
  for (const auto& dim : descriptor.shape()) {
    if (std::holds_alternative<uint32_t>(dim)) {
      shape.push_back(
          MakeGarbageCollected<V8UnionStringOrUnsignedLongEnforceRange>(
              std::get<uint32_t>(dim)));
    } else {
      const auto& dynamic_dim = std::get<webnn::DynamicDimension>(dim);
      if (dynamic_dim.name.has_value()) {
        // Named dynamic dim -> string.
        shape.push_back(
            MakeGarbageCollected<V8UnionStringOrUnsignedLongEnforceRange>(
                String::FromUtf8(*dynamic_dim.name)));
      } else {
        // Unnamed dynamic dim -> null element.
        shape.push_back(nullptr);
      }
    }
  }
  desc->setShape(std::move(shape));
  return desc;
}

// Builds an `MLOperandDescriptor` (concrete shape) from a `webnn` descriptor
// whose dimensions have all been resolved to concrete values.
MLOperandDescriptor* MakeConcreteOperandDescriptor(
    const webnn::OperandDescriptor& descriptor) {
  auto* desc = MLOperandDescriptor::Create();
  desc->setDataType(ToBlinkDataType(descriptor.data_type()));
  Vector<uint32_t> shape;
  shape.reserve(base::checked_cast<wtf_size_t>(descriptor.shape().size()));
  for (const auto& dim : descriptor.shape()) {
    // ComputeShapes() resolves every output dimension to a concrete value
    // before the service returns it.
    CHECK(std::holds_alternative<uint32_t>(dim));
    shape.push_back(std::get<uint32_t>(dim));
  }
  desc->setShape(std::move(shape));
  return desc;
}

// Validates the caller-provided `input_shapes` against the graph's
// `input_constraints` and, on success, converts them into the mojo map sent to
// the service. Mirrors the structural checks the service performs (which it
// would otherwise treat as a misbehaving renderer), so that well-formed but
// incorrect requests surface as a normal exception instead.
bool ValidateAndConvertInputShapes(
    const Vector<std::pair<String, Vector<uint32_t>>>& input_shapes,
    const MLGraph::NamedOperandDescriptors& input_constraints,
    HashMap<String, Vector<uint32_t>>& named_input_shapes,
    ExceptionState& exception_state) {
  if (input_shapes.size() != input_constraints.size()) {
    exception_state.ThrowTypeError(
        "The number of input shapes does not match the number of graph "
        "inputs.");
    return false;
  }
  HashMap<String, uint32_t> dim_name_to_value;
  for (const auto& [name, shape] : input_shapes) {
    auto constraint_it = input_constraints.find(name);
    if (constraint_it == input_constraints.end()) {
      exception_state.ThrowTypeError(
          StrCat({"Unknown graph input \"", name, "\"."}));
      return false;
    }
    if (!named_input_shapes.insert(name, shape).is_new_entry) {
      exception_state.ThrowTypeError(
          StrCat({"Duplicate graph input \"", name, "\"."}));
      return false;
    }
    CHECK(constraint_it->value.has_value());
    const std::vector<webnn::Dimension>& spec_shape =
        constraint_it->value->shape();
    if (shape.size() != spec_shape.size()) {
      exception_state.ThrowTypeError(
          StrCat({"The shape provided for input \"", name,
                  "\" has an unexpected "
                  "number of dimensions."}));
      return false;
    }
    for (wtf_size_t i = 0; i < shape.size(); ++i) {
      if (std::holds_alternative<uint32_t>(spec_shape[i])) {
        if (shape[i] != std::get<uint32_t>(spec_shape[i])) {
          exception_state.ThrowTypeError(
              StrCat({"The shape provided for input \"", name,
                      "\" does not match a static dimension."}));
          return false;
        }
        continue;
      }
      // A dynamic dimension accepts any positive concrete size.
      if (shape[i] == 0) {
        exception_state.ThrowTypeError(
            StrCat({"A dynamic dimension of the shape provided for input \"",
                    name, "\" must be greater than 0."}));
        return false;
      }
      const auto& dynamic_dim =
          std::get<webnn::DynamicDimension>(spec_shape[i]);
      // Only named dynamic dims must resolve consistently. An unnamed dim is an
      // independent unknown, so it is not tracked.
      if (!dynamic_dim.name.has_value()) {
        continue;
      }
      String dynamic_dim_name = String::FromUtf8(*dynamic_dim.name);
      auto add_result = dim_name_to_value.insert(dynamic_dim_name, shape[i]);
      if (!add_result.is_new_entry &&
          add_result.stored_value->value != shape[i]) {
        exception_state.ThrowTypeError(
            StrCat({"Inconsistent values provided for dynamic dimension \"",
                    dynamic_dim_name, "\"."}));
        return false;
      }
    }
  }
  return true;
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

ScriptPromise<IDLRecord<IDLString, MLOperandDescriptor>> MLGraph::computeShapes(
    ScriptState* script_state,
    const Vector<std::pair<String, Vector<uint32_t>>>& input_shapes,
    ExceptionState& exception_state) {
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid script state.");
    return ScriptPromise<IDLRecord<IDLString, MLOperandDescriptor>>();
  }
  if (IsDestroyed()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kInvalidStateError,
        "Graph has been destroyed or the context is lost.");
    return ScriptPromise<IDLRecord<IDLString, MLOperandDescriptor>>();
  }

  HashMap<String, Vector<uint32_t>> named_input_shapes;
  if (!ValidateAndConvertInputShapes(input_shapes, input_constraints_,
                                     named_input_shapes, exception_state)) {
    return ScriptPromise<IDLRecord<IDLString, MLOperandDescriptor>>();
  }

  auto* resolver = MakeGarbageCollected<
      ScriptPromiseResolver<IDLRecord<IDLString, MLOperandDescriptor>>>(
      script_state, exception_state.GetContext());
  auto promise = resolver->Promise();
  remote_graph_->ComputeShapes(
      std::move(named_input_shapes),
      blink::BindOnce(&MLGraph::DidComputeShapes, WrapPersistent(this),
                      WrapPersistent(resolver)));
  return promise;
}

void MLGraph::DidComputeShapes(
    ScriptPromiseResolver<IDLRecord<IDLString, MLOperandDescriptor>>* resolver,
    base::expected<webnn::mojom::blink::ComputeShapesSuccessPtr,
                   webnn::mojom::blink::ErrorPtr> result) {
  ScriptState* script_state = resolver->GetScriptState();
  if (!script_state->ContextIsValid()) {
    return;
  }

  if (!result.has_value()) {
    const webnn::mojom::blink::Error& error = *result.error();
    DOMExceptionCode code =
        error.code == webnn::mojom::blink::Error::Code::kNotSupportedError
            ? DOMExceptionCode::kNotSupportedError
            : DOMExceptionCode::kUnknownError;
    resolver->RejectWithDOMException(code, error.message);
    return;
  }

  HeapVector<std::pair<String, Member<MLOperandDescriptor>>> output_descriptors;
  output_descriptors.reserve(base::checked_cast<wtf_size_t>(
      result.value()->output_descriptors.size()));
  for (const auto& named : result.value()->output_descriptors) {
    output_descriptors.emplace_back(
        named->name, MakeConcreteOperandDescriptor(named->descriptor));
  }
  resolver->Resolve(std::move(output_descriptors));
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
