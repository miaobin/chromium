// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_graph_impl.h"

#include <math.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "base/containers/flat_set.h"
#include "base/dcheck_is_on.h"
#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "base/task/bind_post_task.h"
#include "base/types/expected.h"
#include "base/types/optional_ref.h"
#include "base/types/pass_key.h"
#include "services/webnn/error.h"
#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/public/cpp/webnn_trace.h"
#include "services/webnn/public/cpp/webnn_types.h"
#include "services/webnn/public/mojom/webnn_error.mojom.h"
#include "services/webnn/scoped_gpu_sequence.h"
#include "services/webnn/webnn_context_impl.h"
#include "services/webnn/webnn_graph_builder_impl.h"
#include "services/webnn/webnn_tensor_impl.h"

namespace webnn {

namespace {

// Returns true if any input descriptor contains a DynamicDimension.
bool InputsHaveDynamicDimensions(
    const base::flat_map<std::string, OperandDescriptor>&
        input_names_to_descriptors) {
  for (const auto& [name, descriptor] : input_names_to_descriptors) {
    if (!descriptor.StaticShape().has_value()) {
      return true;
    }
  }
  return false;
}

// Validates the renderer-provided `named_input_shapes` against the graph's
// `input_names_to_descriptors`: every declared input must be present exactly
// once, with a matching rank, static dimensions equal to the spec, dynamic
// dimensions resolving to a positive size, and same-named dynamic
// dimensions resolving to a single consistent value. This mirrors the checks
// `WebNNContextImpl::Dispatch()` performs on input tensors, but operates on
// bare shapes (no data type is supplied to ComputeShapes).
bool ValidateNamedInputShapes(
    const base::flat_map<std::string, std::vector<uint32_t>>&
        named_input_shapes,
    const base::flat_map<std::string, OperandDescriptor>&
        input_names_to_descriptors) {
  if (named_input_shapes.size() != input_names_to_descriptors.size()) {
    return false;
  }
  // Both containers are `base::flat_map`s sorted by key, so a pairwise walk
  // verifies the name sets match exactly.
  base::flat_map<std::string, uint32_t> dim_name_to_value;
  auto spec_it = input_names_to_descriptors.begin();
  for (const auto& [name, actual_shape] : named_input_shapes) {
    if (spec_it->first != name) {
      return false;
    }
    const std::vector<Dimension>& spec_shape = spec_it->second.shape();
    ++spec_it;
    if (actual_shape.size() != spec_shape.size()) {
      return false;
    }
    for (size_t i = 0; i < spec_shape.size(); ++i) {
      if (std::holds_alternative<uint32_t>(spec_shape[i])) {
        if (actual_shape[i] != std::get<uint32_t>(spec_shape[i])) {
          return false;
        }
        continue;
      }
      // A dynamic dimension accepts any positive concrete size.
      if (actual_shape[i] == 0) {
        return false;
      }
      const auto& dynamic_dim = std::get<DynamicDimension>(spec_shape[i]);
      // Only named dynamic dims must resolve consistently across inputs. An
      // unnamed dim is an independent unknown, so it is not tracked.
      if (dynamic_dim.name.has_value()) {
        auto [map_it, inserted] =
            dim_name_to_value.emplace(*dynamic_dim.name, actual_shape[i]);
        if (!inserted && map_it->second != actual_shape[i]) {
          return false;
        }
      }
    }
  }
  return true;
}

// Converts a name -> descriptor map into the mojo array form returned by
// WebNNGraph::ComputeShapes().
std::vector<mojom::NamedOutputDescriptorPtr> ToNamedOutputDescriptors(
    const base::flat_map<std::string, OperandDescriptor>& descriptors) {
  std::vector<mojom::NamedOutputDescriptorPtr> result;
  result.reserve(descriptors.size());
  for (const auto& [name, descriptor] : descriptors) {
    result.push_back(mojom::NamedOutputDescriptor::New(name, descriptor));
  }
  return result;
}

}  // namespace

WebNNGraphImpl::ComputeResourceInfo::ComputeResourceInfo(
    base::flat_map<std::string, OperandDescriptor> input_names_to_descriptors,
    base::flat_map<std::string, OperandDescriptor> output_names_to_descriptors,
    base::flat_map<OperandId, base::flat_set<OperationId>>
        operand_to_dependent_operations,
    base::flat_map<OperandId, OperationId> operand_to_producing_operation,
    std::vector<mojom::OperandPtr> graph_operands,
    std::vector<mojom::OperationPtr> graph_operations,
    std::vector<OperandId> graph_input_operand_ids,
    base::flat_map<OperandId, std::vector<uint8_t>> integer_constant_data,
    base::PassKey<WebNNGraphBuilderImpl> pass_key)
    : input_names_to_descriptors(std::move(input_names_to_descriptors)),
      output_names_to_descriptors(std::move(output_names_to_descriptors)),
      operand_to_dependent_operations(
          std::move(operand_to_dependent_operations)),
      operand_to_producing_operation(std::move(operand_to_producing_operation)),
      has_dynamic_inputs(
          InputsHaveDynamicDimensions(this->input_names_to_descriptors)),
      graph_operands(std::move(graph_operands)),
      graph_operations(std::move(graph_operations)),
      graph_input_operand_ids(std::move(graph_input_operand_ids)),
      integer_constant_data(std::move(integer_constant_data)) {}

WebNNGraphImpl::ComputeResourceInfo::ComputeResourceInfo(
    ComputeResourceInfo&&) = default;
WebNNGraphImpl::ComputeResourceInfo&
WebNNGraphImpl::ComputeResourceInfo::operator=(ComputeResourceInfo&&) = default;

WebNNGraphImpl::ComputeResourceInfo::~ComputeResourceInfo() = default;

WebNNGraphImpl::WebNNGraphImpl(
    mojo::PendingReceiver<mojom::WebNNGraph> receiver,
    WebNNContextImpl& context,
    ComputeResourceInfo compute_resource_info,
    std::vector<mojom::Device> devices)
    : WebNNObjectImpl<mojom::WebNNGraph,
                      blink::WebNNGraphToken,
                      mojo::Receiver<mojom::WebNNGraph>>(
          std::move(receiver),
          context.mojo_task_runner(),
          context.owning_task_runner()),
      context_(context),
      compute_resource_info_(std::move(compute_resource_info)),
      devices_(std::move(devices)) {}

WebNNGraphImpl::~WebNNGraphImpl() = default;

// TODO(crbug.com/514413524): Remove the WebNNGraph interface
void WebNNGraphImpl::OnDisconnect() {
  // Graph pipe disconnect does not remove the graph from graph_impls_. Removal
  // is handled by DestroyGraph on the context pipe, which preserves ordering
  // with Dispatch/ReadTensor/WriteTensor.
  //
  // If the renderer crashes without sending DestroyGraph, the graph remains in
  // graph_impls_ until the context pipe also disconnects, which destroys the
  // entire context and all its graphs.
  ResetMojoReceiver();
}

void WebNNGraphImpl::ComputeShapes(
    const base::flat_map<std::string, std::vector<uint32_t>>&
        named_input_shapes,
    ComputeShapesCallback callback) {
  // Structural validation of the renderer-provided input shapes against the
  // graph's input constraints. A correct renderer always satisfies these (the
  // Blink bindings validate them before sending), so a violation indicates a
  // misbehaving renderer and is fatal, mirroring the input checks in
  // `WebNNContextImpl::Dispatch()`.
  if (!ValidateNamedInputShapes(
          named_input_shapes,
          compute_resource_info_.input_names_to_descriptors)) {
    // Report through the receiver so the binding is closed; this releases the
    // unrun response callback (dropping it while the pipe is open would
    // DCHECK).
    GetMojoReceiver().ReportBadMessage(kBadMessageInvalidTensor);
    return;
  }

  // Static graphs: the output shapes do not depend on the inputs, so return the
  // build-time (already concrete) output descriptors directly.
  if (!compute_resource_info_.has_dynamic_inputs) {
    std::move(callback).Run(
        mojom::ComputeShapesSuccess::New(ToNamedOutputDescriptors(
            compute_resource_info_.output_names_to_descriptors)));
    return;
  }

  // Dynamic graphs: run forward shape inference. Unlike the structural checks
  // above, an inference failure (e.g. an output shape that would require
  // reading input tensor data) is a legitimate "cannot determine shape" answer
  // for the caller rather than a protocol violation, so it is reported as an
  // `Error` instead of terminating the renderer.
  auto inferred = InferConcreteOutputShapes(named_input_shapes);
  if (!inferred.has_value()) {
    std::move(callback).Run(base::unexpected(mojom::Error::New(
        mojom::Error::Code::kNotSupportedError, std::move(inferred.error()))));
    return;
  }

  // Inference is expected to fully resolve every output shape. Guard against a
  // residual dynamic dimension so the renderer always receives concrete
  // descriptors (it cannot represent a dynamic dimension in the result).
  for (const auto& [name, descriptor] : inferred.value()) {
    if (!descriptor.StaticShape().has_value()) {
      std::move(callback).Run(base::unexpected(mojom::Error::New(
          mojom::Error::Code::kNotSupportedError,
          "Could not resolve a concrete shape for output \"" + name + "\".")));
      return;
    }
  }
  std::move(callback).Run(mojom::ComputeShapesSuccess::New(
      ToNamedOutputDescriptors(inferred.value())));
}

base::expected<base::flat_map<std::string, OperandDescriptor>, std::string>
WebNNGraphImpl::InferConcreteOutputShapes(
    const base::flat_map<std::string, std::vector<uint32_t>>&
        named_input_shapes) const {
  const ComputeResourceInfo& resource_info = compute_resource_info_;
  CHECK(resource_info.has_dynamic_inputs);

  // Build a mapping from dynamic dimension names to their concrete values,
  // gathered from the input shapes. Callers must have already validated that
  // the input shapes are consistent with the graph's input constraints, so the
  // descriptor lookups and shape indexing below are guaranteed in bounds.
  base::flat_map<std::string, uint32_t> dim_name_to_value;
  for (const auto& [input_name, actual_shape] : named_input_shapes) {
    auto it = resource_info.input_names_to_descriptors.find(input_name);
    CHECK(it != resource_info.input_names_to_descriptors.end());
    const std::vector<Dimension>& spec_shape = it->second.shape();
    for (size_t i = 0; i < spec_shape.size(); ++i) {
      if (const auto* dynamic_dim =
              std::get_if<DynamicDimension>(&spec_shape[i])) {
        // Only named dynamic dims can be resolved by name; an unnamed dim is an
        // independent unknown.
        if (dynamic_dim->name.has_value()) {
          dim_name_to_value[*dynamic_dim->name] = actual_shape[i];
        }
      }
    }
  }

  // Clone all operands. For operands whose shape contains a DynamicDimension,
  // substitute concrete values resolved from `dim_name_to_value` when all of
  // its symbolic dimensions are known.
  std::vector<mojom::OperandPtr> concrete_operands;
  concrete_operands.reserve(resource_info.graph_operands.size());
  for (const auto& operand : resource_info.graph_operands) {
    auto cloned = operand.Clone();
    // Unranked operands (intermediates whose rank is unknown until dispatch)
    // carry no dimensions to substitute; the forward propagation below infers
    // their concrete shapes from the resolved inputs.
    bool has_dynamic = false;
    if (cloned->descriptor.HasRank()) {
      for (const auto& dim : cloned->descriptor.shape()) {
        if (std::holds_alternative<DynamicDimension>(dim)) {
          has_dynamic = true;
          break;
        }
      }
    }
    if (has_dynamic) {
      const auto& orig_shape = cloned->descriptor.shape();
      std::vector<uint32_t> concrete_shape;
      concrete_shape.reserve(orig_shape.size());
      bool all_resolved = true;
      for (const auto& dim : orig_shape) {
        if (std::holds_alternative<uint32_t>(dim)) {
          concrete_shape.push_back(std::get<uint32_t>(dim));
        } else {
          // Only a named dynamic dim can be resolved from the input shapes; an
          // unnamed dim stays unresolved.
          const auto& dyn = std::get<DynamicDimension>(dim);
          auto val_it = dyn.name ? dim_name_to_value.find(*dyn.name)
                                 : dim_name_to_value.end();
          if (val_it != dim_name_to_value.end()) {
            concrete_shape.push_back(val_it->second);
          } else {
            all_resolved = false;
            break;
          }
        }
      }
      if (all_resolved) {
        auto new_desc = OperandDescriptor::CreateForDeserialization(
            cloned->descriptor.data_type(), concrete_shape,
            cloned->descriptor.pending_permutation());
        if (!new_desc.has_value()) {
          return base::unexpected(
              "Failed to resolve a concrete operand descriptor from the "
              "given input shapes.");
        }
        cloned->descriptor = std::move(new_desc.value());
      }
    }
    concrete_operands.push_back(std::move(cloned));
  }

  // Identify pre-processed operands (inputs + constants) for re-validation.
  base::flat_set<OperandId> processed_operands;
  for (size_t i = 0; i < concrete_operands.size(); ++i) {
    if (concrete_operands[i]->kind == mojom::Operand::Kind::kInput ||
        concrete_operands[i]->kind == mojom::Operand::Kind::kConstant) {
      processed_operands.insert(OperandId(base::checked_cast<uint32_t>(i)));
    }
  }

  // Forward-propagate concrete shapes through the graph, writing inferred
  // output descriptors back into `concrete_operands`.
  if (!WebNNGraphBuilderImpl::InferAndValidateConcreteShapes(
          context_->properties(), concrete_operands,
          resource_info.graph_operations, processed_operands,
          resource_info.integer_constant_data, dim_name_to_value)) {
    return base::unexpected(
        "Failed to infer concrete output shapes from the given input shapes.");
  }

  // Collect the inferred concrete output descriptors.
  base::flat_map<std::string, OperandDescriptor> concrete_output_descriptors;
  for (const auto& operand : concrete_operands) {
    if (operand->kind == mojom::Operand::Kind::kOutput && operand->name) {
      concrete_output_descriptors.emplace(*operand->name, operand->descriptor);
    }
  }
  return concrete_output_descriptors;
}

void WebNNGraphImpl::RunDispatch(
    base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>>
        name_to_input_tensor_map,
    base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>>
        name_to_output_tensor_map,
    ScopedTrace scoped_trace,
    mojo::ReportBadMessageCallback bad_message_cb) {
  // Call DispatchImpl() implemented by an `mojom::WebNNGraph` backend.
  context_->RunOrScheduleTask(base::BindOnce(
      [](WebNNGraphImpl* self,
         base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>>
             name_to_input_tensor_map,
         base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>>
             name_to_output_tensor_map,
         ScopedTrace scoped_trace,
         mojo::ReportBadMessageCallback bad_message_cb) {
        for (auto& [name, tensor] : name_to_input_tensor_map) {
          if (tensor->is_exported()) {
            LOG(ERROR)
                << "[WebNN] Invalid to dispatch graph when input tensor (" +
                       name + ") is exported.";
            std::move(bad_message_cb).Run(kBadMessageInvalidTensor);
            return;
          }
        }

        for (auto& [name, tensor] : name_to_output_tensor_map) {
          if (tensor->is_exported()) {
            LOG(ERROR) << "[WebNN] Invalid to dispatch graph when output "
                          "tensor (" +
                              name + ") is exported.";
            std::move(bad_message_cb).Run(kBadMessageInvalidTensor);
            return;
          }
        }

        self->DispatchImpl(std::move(name_to_input_tensor_map),
                           std::move(name_to_output_tensor_map));
      },
      base::RetainedRef(this), std::move(name_to_input_tensor_map),
      std::move(name_to_output_tensor_map), std::move(scoped_trace),
      std::move(bad_message_cb)));
}

}  // namespace webnn
