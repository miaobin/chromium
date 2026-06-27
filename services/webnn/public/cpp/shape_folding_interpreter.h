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
// Values are carried as `double`. Integer operands represent exact integral
// values: WebNN dimensions are uint32 (< 2^32), well within the 53-bit exact
// integer range of double, so concrete shapes lose no precision. Intermediate
// integer products (e.g. ReduceProduct of a large shape) that exceed 2^53 can
// lose precision; such values would also overflow a uint32 dimension and are
// rejected at the conversion boundary, so this does not produce wrong shapes.
// Floating-point operands (e.g. Resize scales) carry real values, enabling
// folds like floor(dim * scale) that integer-only arithmetic cannot express.
//
// Arithmetic semantics follow the producing operand's data type: integer
// operands use integer division (truncating) and modulo; float operands use
// real division. cast to an integer type truncates toward zero.
//
// Supported chain operations:
//   - Constant operands (integer and float32/float16 types)
//   - shape() — returns input tensor dimensions
//   - concat, gather, slice, reshape, reverse, transpose
//   - squeeze, unsqueeze, flatten (rank-only changes; identity on values)
//   - split (1-D shape vectors along axis 0)
//   - expand (broadcast a 1-D shape vector to a concrete 1-D target)
//   - add, sub, mul, div, min, max, mod (element-wise arithmetic)
//   - reciprocal, floor, ceil, abs, neg (element-wise unary)
//   - equal, notEqual, greater(OrEqual), lesser(OrEqual), logicalAnd/Or/Xor
//     (element-wise comparison/logical, producing 0/1)
//   - reduce (sum, product, mean, min, max on 1-D)
//   - cast between integer and floating-point types
//
// Returns nullopt if the chain contains unsupported operations or data types.
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

  // Evaluates the concrete values of the given operand as doubles. Integer
  // operands carry exact integral values (up to 2^53; see class comment);
  // floating-point operands carry real values. Returns nullopt if the operand
  // cannot be evaluated (unsupported op chain, unsupported dtype, etc.).
  //
  // Callers that need integer dimensions must convert at the boundary, after
  // validating the value is finite and in range (see EvaluateShapeOperand).
  std::optional<std::vector<double>> Evaluate(OperandId operand_id);

 private:
  std::optional<std::vector<double>> EvaluateImpl(OperandId operand_id);

  // Reads constant values from raw bytes based on data type, as doubles.
  std::optional<std::vector<double>> ReadConstantValues(
      OperandId operand_id) const;

  // Interprets a single operation and returns the output values as doubles.
  std::optional<std::vector<double>> InterpretOperation(
      const mojom::Operation& operation,
      OperandId output_operand_id);

  // Returns true if `operand_id` is a valid operand with an integer data type.
  // Out-of-range ids return false.
  //
  // CONTRACT for operator handlers: values flow as `double` regardless of the
  // operand's declared type, so any handler whose result differs between
  // integer and real arithmetic MUST restore integer semantics for integer
  // outputs by gating on this predicate (keyed on the *output* operand). This
  // is the one invariant the type system cannot enforce for us; missing it on a
  // new handler yields a fractional value that silently survives until the
  // int64 boundary rejects it (or, worse, an off-by-one shape). Today's
  // restoration points: truncating division/mean (std::trunc), modulo
  // (std::fmod), integer cast (std::trunc), and comparison/logical results
  // (0.0/1.0). Handlers that are exact over doubles for all integral inputs
  // (add/sub/mul/min/max, reshape/concat/gather/slice, ...) need no gating.
  bool IsIntegerOperand(OperandId operand_id) const;

  base::raw_span<const mojom::OperandPtr> operands_;
  const base::raw_ref<const std::vector<mojom::OperationPtr>> operations_;
  const base::raw_ref<const base::flat_map<OperandId, OperationId>>
      operand_to_producing_operation_;
  const base::raw_ref<const base::flat_map<OperandId, std::vector<uint8_t>>>
      constant_data_;

  // Memoization cache for evaluated operand values.
  base::flat_map<OperandId, std::optional<std::vector<double>>> cache_;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_PUBLIC_CPP_SHAPE_FOLDING_INTERPRETER_H_
