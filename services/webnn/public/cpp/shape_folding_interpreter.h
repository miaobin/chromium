// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_PUBLIC_CPP_SHAPE_FOLDING_INTERPRETER_H_
#define SERVICES_WEBNN_PUBLIC_CPP_SHAPE_FOLDING_INTERPRETER_H_

#include <optional>
#include <vector>

#include "base/component_export.h"
#include "base/containers/flat_map.h"
#include "base/containers/span.h"
#include "base/memory/raw_ref.h"
#include "base/memory/raw_span.h"
#include "services/webnn/public/cpp/webnn_types.h"
#include "services/webnn/public/mojom/webnn_graph.mojom-forward.h"

namespace webnn {

// Evaluates integer operand values at dispatch time by interpreting the shape
// computation chain backward from a target operand. Used for dynamic* ops
// (dynamicReshape, dynamicExpand, etc.) whose output shape depends on runtime
// tensor values.
//
// Supported chain operations:
//   - Constant operands (integer types)
//   - shape() — returns input tensor dimensions as int64 values
//   - concat, gather, slice, reshape, reverse, transpose
//   - add, sub, mul, div, min, max (element-wise integer arithmetic)
//   - cast to integer types
//
// Returns nullopt if the chain contains unsupported operations or non-integer
// data types.
class COMPONENT_EXPORT(WEBNN_PUBLIC_CPP) ShapeFoldingInterpreter {
 public:
  ShapeFoldingInterpreter(
      base::span<const mojom::OperandPtr> operands,
      const std::vector<mojom::OperationPtr>& operations,
      const base::flat_map<OperandId, OperationId>&
          operand_to_producing_operation,
      const base::flat_map<OperandId, std::vector<uint8_t>>&
          constant_data);

  ~ShapeFoldingInterpreter();

  ShapeFoldingInterpreter(const ShapeFoldingInterpreter&) = delete;
  ShapeFoldingInterpreter& operator=(const ShapeFoldingInterpreter&) = delete;

  // Evaluates the concrete int64 values of the given operand.
  // Returns nullopt if the operand cannot be evaluated (unsupported op chain,
  // non-integer dtype, etc.).
  std::optional<std::vector<int64_t>> Evaluate(OperandId operand_id);

 private:
  std::optional<std::vector<int64_t>> EvaluateImpl(OperandId operand_id);

  // Reads integer values from raw constant bytes based on data type.
  std::optional<std::vector<int64_t>> ReadConstantValues(
      OperandId operand_id) const;

  // Interprets a single operation and returns the output values.
  std::optional<std::vector<int64_t>> InterpretOperation(
      const mojom::Operation& operation,
      OperandId output_operand_id);

  base::raw_span<const mojom::OperandPtr> operands_;
  const base::raw_ref<const std::vector<mojom::OperationPtr>> operations_;
  const base::raw_ref<const base::flat_map<OperandId, OperationId>>
      operand_to_producing_operation_;
  const base::raw_ref<const base::flat_map<OperandId, std::vector<uint8_t>>>
      constant_data_;

  // Memoization cache for evaluated operand values.
  base::flat_map<OperandId, std::optional<std::vector<int64_t>>> cache_;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_PUBLIC_CPP_SHAPE_FOLDING_INTERPRETER_H_
