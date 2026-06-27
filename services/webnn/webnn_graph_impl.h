// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_WEBNN_GRAPH_IMPL_H_
#define SERVICES_WEBNN_WEBNN_GRAPH_IMPL_H_

#include <string>
#include <vector>

#include "base/component_export.h"
#include "base/containers/flat_map.h"
#include "base/memory/raw_ref.h"
#include "base/types/expected.h"
#include "base/types/pass_key.h"
#include "mojo/public/cpp/bindings/message.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/bindings/receiver.h"
#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/public/cpp/webnn_trace.h"
#include "services/webnn/public/cpp/webnn_types.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"
#include "services/webnn/webnn_object_impl.h"

namespace webnn {

class WebNNContextImpl;
class WebNNGraphBuilderImpl;
class WebNNTensorImpl;

// GPU process implementation of the `MLGraph` interface. While this class is
// reference-counted a `WebNNGraphImpl` is guaranteed not to outlive the
// `WebNNContextImpl` that created it because references are only held by the
// context itself or by tasks scheduled to its `gpu::Scheduler` sequence which
// is shut down when the context is destroyed.
//
// This invariant is checked by the `raw_ref<WebNNContextImpl>` member, which
// will trigger dangling pointer warnings in debug builds and safe crashes in
// release builds.
class COMPONENT_EXPORT(WEBNN_SERVICE) WebNNGraphImpl
    : public WebNNObjectImpl<mojom::WebNNGraph,
                             blink::WebNNGraphToken,
                             mojo::Receiver<mojom::WebNNGraph>> {
 public:
  // Describes the constraints of a graph's inputs and outputs.
  struct COMPONENT_EXPORT(WEBNN_SERVICE) ComputeResourceInfo {
    ComputeResourceInfo(
        base::flat_map<std::string, OperandDescriptor>
            input_names_to_descriptors,
        base::flat_map<std::string, OperandDescriptor>
            output_names_to_descriptors,
        base::flat_map<OperandId, base::flat_set<OperationId>>
            operand_to_dependent_operations,
        base::flat_map<OperandId, OperationId> operand_to_producing_operation,
        std::vector<mojom::OperandPtr> graph_operands,
        std::vector<mojom::OperationPtr> graph_operations,
        std::vector<OperandId> graph_input_operand_ids,
        base::flat_map<OperandId, std::vector<uint8_t>> shape_constant_data,
        base::PassKey<WebNNGraphBuilderImpl> pass_key);
    ~ComputeResourceInfo();

    ComputeResourceInfo(const ComputeResourceInfo&) = delete;
    ComputeResourceInfo& operator=(const ComputeResourceInfo&) = delete;

    ComputeResourceInfo(ComputeResourceInfo&&);
    ComputeResourceInfo& operator=(ComputeResourceInfo&&);

    base::flat_map<std::string, OperandDescriptor> input_names_to_descriptors;
    base::flat_map<std::string, OperandDescriptor> output_names_to_descriptors;
    base::flat_map<OperandId, base::flat_set<OperationId>>
        operand_to_dependent_operations;
    base::flat_map<OperandId, OperationId> operand_to_producing_operation;

    // True if any input descriptor contains a DynamicDimension.
    // Used as a fast-path gate: when false, dispatch skips re-validation.
    bool has_dynamic_inputs = false;

    // Graph structure for dispatch-time re-validation (Phase B Steps 3-5).
    // Only populated when has_dynamic_inputs is true.
    std::vector<mojom::OperandPtr> graph_operands;
    std::vector<mojom::OperationPtr> graph_operations;
    std::vector<OperandId> graph_input_operand_ids;

    // Raw byte data of the constant operands on a shape-computation chain
    // (integer or float), for dispatch-time shape folding (Phase B Step 2).
    // Collected by walking back from dynamic ops' shape operands, so weight
    // tensors are excluded. Only populated when has_dynamic_inputs is true.
    base::flat_map<OperandId, std::vector<uint8_t>> shape_constant_data;
  };

  // Constructs a graph where the receiver and implementation are owned by the
  // context.
  WebNNGraphImpl(mojo::PendingReceiver<mojom::WebNNGraph> receiver,
                 WebNNContextImpl& context,
                 ComputeResourceInfo compute_resource_info,
                 std::vector<mojom::Device> devices);

  WebNNGraphImpl(const WebNNGraphImpl&) = delete;
  WebNNGraphImpl& operator=(const WebNNGraphImpl&) = delete;

  const ComputeResourceInfo& compute_resource_info() const {
    return compute_resource_info_;
  }

  const std::vector<mojom::Device>& devices() { return devices_; }

  // mojom::WebNNGraph:
  void ComputeShapes(const base::flat_map<std::string, std::vector<uint32_t>>&
                         named_input_shapes,
                     ComputeShapesCallback callback) override;

  // Execute the dispatch on the GPU sequence (or directly if no GPU sequence).
  // Called by WebNNContextImpl::Dispatch() after input/output tensors have been
  // validated and resolved. Schedules the backend's DispatchImpl() on the GPU
  // sequence, checking that no tensors are exported before running.
  void RunDispatch(
      base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>> named_inputs,
      base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>> named_outputs,
      webnn::ScopedTrace scoped_trace,
      mojo::ReportBadMessageCallback bad_message_cb);

  // Runs forward shape inference for a graph with dynamic input dimensions:
  // given the concrete shapes of the graph's inputs (keyed by input name),
  // substitutes the symbolic dimensions and infers a concrete
  // `OperandDescriptor` for every graph output.
  //
  // Returns the inferred output descriptors keyed by output name, or an error
  // message if the output shapes cannot be resolved from the input shapes
  // alone (e.g. a shape that would require reading input tensor data). Callers
  // must validate the input shapes against the graph's input constraints
  // beforehand, and must only call this when
  // `compute_resource_info().has_dynamic_inputs` is true.
  base::expected<base::flat_map<std::string, OperandDescriptor>, std::string>
  InferConcreteOutputShapes(
      const base::flat_map<std::string, std::vector<uint32_t>>&
          named_input_shapes) const;

 protected:
  ~WebNNGraphImpl() override;

  // The `WebNNContextImpl` which owns and will outlive this object.
  const base::raw_ref<WebNNContextImpl> context_;

 private:
  void OnDisconnect() override;

  // Execute the compiled platform graph. The `named_inputs` and `named_outputs`
  // were validated in WebNNContextImpl::Dispatch().
  virtual void DispatchImpl(
      base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>> named_inputs,
      base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>>
          named_outputs) = 0;

  // The validator is to make sure the inputs from a compute call match the
  // built graph's expected.
  ComputeResourceInfo compute_resource_info_;

  const std::vector<mojom::Device> devices_;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_WEBNN_GRAPH_IMPL_H_
