// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/public/cpp/shape_folding_interpreter.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "base/containers/span.h"
#include "base/containers/span_reader.h"
#include "base/numerics/checked_math.h"
#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"

// The fp16.h header triggers narrowing and sign conversion warnings.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif
#include "third_party/fp16/src/include/fp16.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace webnn {

namespace {

bool IsIntegerDataType(OperandDataType dtype) {
  switch (dtype) {
    case OperandDataType::kInt8:
    case OperandDataType::kUint8:
    case OperandDataType::kInt32:
    case OperandDataType::kUint32:
    case OperandDataType::kInt64:
    case OperandDataType::kUint64:
    case OperandDataType::kInt4:
    case OperandDataType::kUint4:
      return true;
    case OperandDataType::kFloat16:
    case OperandDataType::kFloat32:
      return false;
  }
}

// Reads constant values from raw bytes as doubles. Integer types convert
// exactly (WebNN dimensions are <= uint32, within double's 53-bit exact
// range); float32/float16 decode to their real values.
std::optional<std::vector<double>> ReadConstantValuesAsDouble(
    OperandDataType dtype,
    base::span<const uint8_t> bytes) {
  std::vector<double> values;
  base::SpanReader reader(bytes);

  switch (dtype) {
    case OperandDataType::kInt8: {
      while (reader.remaining() > 0) {
        int8_t v;
        if (!reader.ReadI8NativeEndian(v)) {
          return std::nullopt;
        }
        values.push_back(static_cast<double>(v));
      }
      break;
    }
    case OperandDataType::kUint8: {
      while (reader.remaining() > 0) {
        uint8_t v;
        if (!reader.ReadU8NativeEndian(v)) {
          return std::nullopt;
        }
        values.push_back(static_cast<double>(v));
      }
      break;
    }
    case OperandDataType::kInt32: {
      while (reader.remaining() > 0) {
        int32_t v;
        if (!reader.ReadI32NativeEndian(v)) {
          return std::nullopt;
        }
        values.push_back(static_cast<double>(v));
      }
      break;
    }
    case OperandDataType::kUint32: {
      while (reader.remaining() > 0) {
        uint32_t v;
        if (!reader.ReadU32NativeEndian(v)) {
          return std::nullopt;
        }
        values.push_back(static_cast<double>(v));
      }
      break;
    }
    case OperandDataType::kInt64: {
      while (reader.remaining() > 0) {
        int64_t v;
        if (!reader.ReadI64NativeEndian(v)) {
          return std::nullopt;
        }
        values.push_back(static_cast<double>(v));
      }
      break;
    }
    case OperandDataType::kUint64: {
      while (reader.remaining() > 0) {
        uint64_t v;
        if (!reader.ReadU64NativeEndian(v)) {
          return std::nullopt;
        }
        values.push_back(static_cast<double>(v));
      }
      break;
    }
    case OperandDataType::kFloat32: {
      while (reader.remaining() > 0) {
        float v;
        if (!reader.ReadFloatNativeEndian(v)) {
          return std::nullopt;
        }
        values.push_back(static_cast<double>(v));
      }
      break;
    }
    case OperandDataType::kFloat16: {
      while (reader.remaining() > 0) {
        uint16_t v;
        if (!reader.ReadU16NativeEndian(v)) {
          return std::nullopt;
        }
        values.push_back(static_cast<double>(fp16_ieee_to_fp32_value(v)));
      }
      break;
    }
    default:
      return std::nullopt;
  }
  return values;
}

}  // namespace

ShapeFoldingInterpreter::ShapeFoldingInterpreter(
    base::span<const mojom::OperandPtr> operands,
    const std::vector<mojom::OperationPtr>& operations,
    const base::flat_map<OperandId, OperationId>&
        operand_to_producing_operation,
    const base::flat_map<OperandId, std::vector<uint8_t>>& constant_data)
    : operands_(operands),
      operations_(operations),
      operand_to_producing_operation_(operand_to_producing_operation),
      constant_data_(constant_data) {}

ShapeFoldingInterpreter::~ShapeFoldingInterpreter() = default;

std::optional<std::vector<double>> ShapeFoldingInterpreter::Evaluate(
    OperandId operand_id) {
  auto cache_it = cache_.find(operand_id);
  if (cache_it != cache_.end()) {
    return cache_it->second;
  }

  auto result = EvaluateImpl(operand_id);
  // Only cache successes. Failures may become resolvable after more
  // operations have their output shapes inferred (e.g., Shape op on a
  // tensor whose dims were dynamic but have since been resolved).
  if (result) {
    cache_[operand_id] = result;
  }
  return result;
}

std::optional<std::vector<double>> ShapeFoldingInterpreter::ReadConstantValues(
    OperandId operand_id) const {
  auto data_it = constant_data_->find(operand_id);
  if (data_it == constant_data_->end()) {
    return std::nullopt;
  }

  if (operand_id.value() >= operands_.size() || !operands_[operand_id.value()]) {
    return std::nullopt;
  }
  const auto& operand = operands_[operand_id.value()];
  return ReadConstantValuesAsDouble(operand->descriptor.data_type(),
                                    data_it->second);
}

bool ShapeFoldingInterpreter::IsIntegerOperand(OperandId operand_id) const {
  if (operand_id.value() >= operands_.size() ||
      !operands_[operand_id.value()]) {
    return false;
  }
  return IsIntegerDataType(
      operands_[operand_id.value()]->descriptor.data_type());
}

std::optional<std::vector<double>> ShapeFoldingInterpreter::EvaluateImpl(
    OperandId operand_id) {
  if (operand_id.value() >= operands_.size() ||
      !operands_[operand_id.value()]) {
    return std::nullopt;
  }

  const auto& operand = operands_[operand_id.value()];

  // Constants: read values directly from stored data.
  if (operand->kind == mojom::Operand::Kind::kConstant) {
    return ReadConstantValues(operand_id);
  }

  // Input operands: not evaluable as values (we know their shape but not
  // their tensor data at validation time).
  if (operand->kind == mojom::Operand::Kind::kInput) {
    return std::nullopt;
  }

  // Output operand: look up producing operation and interpret it.
  auto prod_it = operand_to_producing_operation_->find(operand_id);
  if (prod_it == operand_to_producing_operation_->end()) {
    return std::nullopt;
  }

  OperationId op_id = prod_it->second;
  if (op_id >= operations_->size()) {
    return std::nullopt;
  }

  return InterpretOperation(*(*operations_)[op_id], operand_id);
}

std::optional<std::vector<double>> ShapeFoldingInterpreter::InterpretOperation(
    const mojom::Operation& operation,
    OperandId output_operand_id) {
  // shape() — returns input tensor dimensions.
  if (operation.is_shape()) {
    const auto& shape_op = *operation.get_shape();
    if (shape_op.input_operand_id.value() >= operands_.size() ||
        !operands_[shape_op.input_operand_id.value()]) {
      return std::nullopt;
    }
    const auto& input_operand = operands_[shape_op.input_operand_id.value()];
    const auto& shape = input_operand->descriptor.shape();
    std::vector<double> values;
    values.reserve(shape.size());
    for (const auto& dim : shape) {
      if (!std::holds_alternative<uint32_t>(dim)) {
        // Input shape should be concrete at this point in infer mode.
        return std::nullopt;
      }
      values.push_back(static_cast<double>(std::get<uint32_t>(dim)));
    }
    return values;
  }

  // concat — concatenate evaluated inputs along axis.
  if (operation.is_concat()) {
    const auto& concat_op = *operation.get_concat();
    std::vector<double> result;
    // For 1-D shape tensors, concat just appends values.
    // For multi-dimensional concat, we need axis handling, but shape tensors
    // are typically 1-D so axis=0 concatenation is the common case.
    for (OperandId input_id : concat_op.input_operand_ids) {
      auto input_values = Evaluate(input_id);
      if (!input_values) {
        return std::nullopt;
      }
      result.insert(result.end(), input_values->begin(), input_values->end());
    }
    return result;
  }

  // gather — extract elements from data using indices.
  if (operation.is_gather()) {
    const auto& gather_op = *operation.get_gather();
    auto data_values = Evaluate(gather_op.input_operand_id);
    auto indices_values = Evaluate(gather_op.indices_operand_id);
    if (!data_values || !indices_values) {
      return std::nullopt;
    }
    // For 1-D data (typical shape tensor), gather just indexes into it.
    std::vector<double> result;
    result.reserve(indices_values->size());
    for (double idx_d : *indices_values) {
      // Indices are integral; reject non-finite or non-integer values.
      if (!std::isfinite(idx_d) || idx_d != std::trunc(idx_d)) {
        return std::nullopt;
      }
      int64_t idx = static_cast<int64_t>(idx_d);
      // Handle negative indices.
      if (idx < 0) {
        idx += static_cast<int64_t>(data_values->size());
      }
      if (idx < 0 || static_cast<size_t>(idx) >= data_values->size()) {
        return std::nullopt;
      }
      result.push_back((*data_values)[static_cast<size_t>(idx)]);
    }
    return result;
  }

  // Element-wise binary arithmetic. Division and modulo follow the output
  // operand's data type: integer operands keep integer (truncating) semantics,
  // float operands use real arithmetic.
  if (operation.is_element_wise_binary()) {
    const auto& binary_op = *operation.get_element_wise_binary();
    auto lhs_values = Evaluate(binary_op.lhs_operand_id);
    auto rhs_values = Evaluate(binary_op.rhs_operand_id);
    if (!lhs_values || !rhs_values) {
      return std::nullopt;
    }

    // See IsIntegerOperand's contract: div/mod below diverge between integer
    // and real arithmetic, so they must restore integer semantics for integer
    // outputs. add/sub/mul/min/max are exact over doubles for integral inputs
    // and need no gating.
    const bool is_integer = IsIntegerOperand(output_operand_id);

    // Handle broadcasting: if sizes differ, one must be scalar (size 1).
    const std::vector<double>* a = &*lhs_values;
    const std::vector<double>* b = &*rhs_values;
    size_t result_size = std::max(a->size(), b->size());
    if (a->size() != result_size && a->size() != 1) {
      return std::nullopt;
    }
    if (b->size() != result_size && b->size() != 1) {
      return std::nullopt;
    }

    std::vector<double> result;
    result.reserve(result_size);
    for (size_t i = 0; i < result_size; ++i) {
      double av = (*a)[a->size() == 1 ? 0 : i];
      double bv = (*b)[b->size() == 1 ? 0 : i];
      double rv;
      switch (binary_op.kind) {
        case mojom::ElementWiseBinary::Kind::kAdd:
          rv = av + bv;
          break;
        case mojom::ElementWiseBinary::Kind::kSub:
          rv = av - bv;
          break;
        case mojom::ElementWiseBinary::Kind::kMul:
          rv = av * bv;
          break;
        case mojom::ElementWiseBinary::Kind::kDiv:
          if (bv == 0) {
            return std::nullopt;
          }
          // Integer operands truncate toward zero (matching C++ integer
          // division); float operands use real division.
          rv = is_integer ? std::trunc(av / bv) : (av / bv);
          break;
        case mojom::ElementWiseBinary::Kind::kMax:
          rv = std::max(av, bv);
          break;
        case mojom::ElementWiseBinary::Kind::kMin:
          rv = std::min(av, bv);
          break;
        case mojom::ElementWiseBinary::Kind::kMod:
          if (bv == 0) {
            return std::nullopt;
          }
          // std::fmod truncates toward zero, matching C++ integer % for the
          // integral values shape arithmetic uses.
          rv = std::fmod(av, bv);
          break;
        // Comparison and logical ops produce a boolean (0/1) result. These are
        // common in shape computations (e.g. Equal/Where chains that resolve a
        // -1 or dynamic dimension).
        case mojom::ElementWiseBinary::Kind::kEqual:
          rv = (av == bv) ? 1.0 : 0.0;
          break;
        case mojom::ElementWiseBinary::Kind::kNotEqual:
          rv = (av != bv) ? 1.0 : 0.0;
          break;
        case mojom::ElementWiseBinary::Kind::kGreater:
          rv = (av > bv) ? 1.0 : 0.0;
          break;
        case mojom::ElementWiseBinary::Kind::kGreaterOrEqual:
          rv = (av >= bv) ? 1.0 : 0.0;
          break;
        case mojom::ElementWiseBinary::Kind::kLesser:
          rv = (av < bv) ? 1.0 : 0.0;
          break;
        case mojom::ElementWiseBinary::Kind::kLesserOrEqual:
          rv = (av <= bv) ? 1.0 : 0.0;
          break;
        case mojom::ElementWiseBinary::Kind::kLogicalAnd:
          rv = (av != 0 && bv != 0) ? 1.0 : 0.0;
          break;
        case mojom::ElementWiseBinary::Kind::kLogicalOr:
          rv = (av != 0 || bv != 0) ? 1.0 : 0.0;
          break;
        case mojom::ElementWiseBinary::Kind::kLogicalXor:
          rv = ((av != 0) != (bv != 0)) ? 1.0 : 0.0;
          break;
        case mojom::ElementWiseBinary::Kind::kPow:
        default:
          // kPow and any future kinds are not folded (essentially absent from
          // shape arithmetic).
          return std::nullopt;
      }
      result.push_back(rv);
    }
    return result;
  }

  // reshape — reinterpret values with a different shape (values unchanged).
  if (operation.is_reshape()) {
    const auto& reshape_op = *operation.get_reshape();
    return Evaluate(reshape_op.input_operand_id);
  }

  // squeeze / unsqueeze / flatten — these only change the rank (inserting or
  // removing size-1 axes, or collapsing axes into two). They never reorder or
  // alter the underlying elements, so on the flattened value list used for
  // shape folding they are identity pass-throughs, like reshape.
  if (operation.is_squeeze()) {
    return Evaluate(operation.get_squeeze()->input_operand_id);
  }
  if (operation.is_unsqueeze()) {
    return Evaluate(operation.get_unsqueeze()->input_operand_id);
  }
  if (operation.is_flatten()) {
    return Evaluate(operation.get_flatten()->input_operand_id);
  }

  // reverse — reverse values along an axis.
  if (operation.is_reverse()) {
    const auto& reverse_op = *operation.get_reverse();
    auto values = Evaluate(reverse_op.input_operand_id);
    if (!values) {
      return std::nullopt;
    }
    // For 1-D tensors, just reverse the vector.
    std::reverse(values->begin(), values->end());
    return values;
  }

  // transpose — for 1-D, this is a no-op.
  if (operation.is_transpose()) {
    const auto& transpose_op = *operation.get_transpose();
    return Evaluate(transpose_op.input_operand_id);
  }

  // slice — extract a contiguous sub-range.
  if (operation.is_slice()) {
    const auto& slice_op = *operation.get_slice();
    auto values = Evaluate(slice_op.input_operand_id);
    if (!values) {
      return std::nullopt;
    }
    // For 1-D slicing: starts[0] and sizes[0].
    if (slice_op.ranges.size() != 1) {
      // Multi-dimensional slice on shape tensor is uncommon; bail out.
      return std::nullopt;
    }
    uint32_t start = slice_op.ranges[0].start;
    uint32_t size = slice_op.ranges[0].size;
    if (static_cast<size_t>(start) + size > values->size()) {
      return std::nullopt;
    }
    return std::vector<double>(values->begin() + start,
                               values->begin() + start + size);
  }

  // Element-wise unary — cast, floor/ceil, reciprocal, abs, neg. These are the
  // operations Resize/interpolate size chains use, e.g. floor(dim * scale) or
  // floor(side / max(H, W) * target).
  if (operation.is_element_wise_unary()) {
    const auto& unary_op = *operation.get_element_wise_unary();
    auto input_values = Evaluate(unary_op.input_operand_id);
    if (!input_values) {
      return std::nullopt;
    }
    switch (unary_op.kind) {
      case mojom::ElementWiseUnary::Kind::kCast: {
        // Cast is a type-level operation; values are already double. If the
        // output type is integer, truncate toward zero (matching an integer
        // cast); a cast to a float type passes the values through unchanged.
        // This is the integer-semantics restoration point for cast; see
        // IsIntegerOperand's contract.
        if (IsIntegerOperand(output_operand_id)) {
          std::vector<double> result;
          result.reserve(input_values->size());
          for (double v : *input_values) {
            // A non-finite value cannot become a concrete integer.
            if (!std::isfinite(v)) {
              return std::nullopt;
            }
            result.push_back(std::trunc(v));
          }
          return result;
        }
        return input_values;
      }
      case mojom::ElementWiseUnary::Kind::kFloor: {
        std::vector<double> result;
        result.reserve(input_values->size());
        for (double v : *input_values) {
          result.push_back(std::floor(v));
        }
        return result;
      }
      case mojom::ElementWiseUnary::Kind::kCeil: {
        std::vector<double> result;
        result.reserve(input_values->size());
        for (double v : *input_values) {
          result.push_back(std::ceil(v));
        }
        return result;
      }
      case mojom::ElementWiseUnary::Kind::kReciprocal: {
        std::vector<double> result;
        result.reserve(input_values->size());
        for (double v : *input_values) {
          if (v == 0.0) {
            return std::nullopt;
          }
          result.push_back(1.0 / v);
        }
        return result;
      }
      case mojom::ElementWiseUnary::Kind::kAbs: {
        std::vector<double> result;
        result.reserve(input_values->size());
        for (double v : *input_values) {
          result.push_back(std::abs(v));
        }
        return result;
      }
      case mojom::ElementWiseUnary::Kind::kNeg: {
        std::vector<double> result;
        result.reserve(input_values->size());
        for (double v : *input_values) {
          result.push_back(-v);
        }
        return result;
      }
      default:
        return std::nullopt;
    }
  }

  // range — generate a sequence [start, start+delta, ...) up to limit. Inputs
  // are integral for shape arithmetic.
  if (operation.is_range()) {
    const auto& range_op = *operation.get_range();
    auto start_values = Evaluate(range_op.start_operand_id);
    auto limit_values = Evaluate(range_op.limit_operand_id);
    auto delta_values = Evaluate(range_op.delta_operand_id);
    if (!start_values || !limit_values || !delta_values) {
      return std::nullopt;
    }
    // All inputs must be scalar and finite.
    if (start_values->size() != 1 || limit_values->size() != 1 ||
        delta_values->size() != 1) {
      return std::nullopt;
    }
    double start = (*start_values)[0];
    double limit = (*limit_values)[0];
    double delta = (*delta_values)[0];
    if (!std::isfinite(start) || !std::isfinite(limit) ||
        !std::isfinite(delta) || delta == 0.0) {
      return std::nullopt;
    }
    double count_d = std::ceil((limit - start) / delta);
    int64_t count = static_cast<int64_t>(std::max(0.0, count_d));
    // Safety limit to prevent huge allocations during shape folding.
    constexpr int64_t kMaxRangeElements = 100000;
    if (count > kMaxRangeElements) {
      return std::nullopt;
    }
    std::vector<double> result;
    result.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
      result.push_back(start + static_cast<double>(i) * delta);
    }
    return result;
  }

  // dynamic_reshape — values pass through unchanged (same as reshape).
  if (operation.is_dynamic_reshape()) {
    const auto& reshape_op = *operation.get_dynamic_reshape();
    return Evaluate(reshape_op.input_operand_id);
  }

  // dynamic_slice — extract a sub-range with dynamic starts/sizes. Like the
  // static slice operator, `sizes` is the window span (the sliced range is
  // [start, start + size)) and strides is a build-time constant attribute.
  if (operation.is_dynamic_slice()) {
    const auto& slice_op = *operation.get_dynamic_slice();
    auto values = Evaluate(slice_op.input_operand_id);
    auto starts = Evaluate(slice_op.starts_operand_id);
    auto sizes = Evaluate(slice_op.sizes_operand_id);
    if (!values || !starts || !sizes) {
      return std::nullopt;
    }
    // Only support 1-D inputs (typical for shape tensors).
    if (starts->size() != 1 || sizes->size() != 1) {
      return std::nullopt;
    }
    int64_t stride = 1;
    if (!slice_op.strides.empty()) {
      if (slice_op.strides.size() != 1) {
        return std::nullopt;
      }
      stride = slice_op.strides[0];
    }
    if (stride <= 0) {
      return std::nullopt;
    }
    // starts and sizes are uint32 (non-negative) integral indices; no
    // negative-index handling.
    double start_d = (*starts)[0];
    double size_d = (*sizes)[0];
    if (!std::isfinite(start_d) || start_d != std::trunc(start_d) ||
        !std::isfinite(size_d) || size_d != std::trunc(size_d)) {
      return std::nullopt;
    }
    int64_t dim_size = static_cast<int64_t>(values->size());
    int64_t start = static_cast<int64_t>(start_d);
    int64_t end = start + static_cast<int64_t>(size_d);
    start = std::clamp(start, int64_t{0}, dim_size);
    end = std::clamp(end, int64_t{0}, dim_size);

    std::vector<double> result;
    for (int64_t i = start; i < end; i += stride) {
      result.push_back((*values)[static_cast<size_t>(i)]);
    }
    return result;
  }

  // reduce — apply reduction along axes on integer value tensors.
  // For shape-folding, we support common reductions on 1-D tensors.
  if (operation.is_reduce()) {
    const auto& reduce_op = *operation.get_reduce();
    auto input_values = Evaluate(reduce_op.input_operand_id);
    if (!input_values || input_values->empty()) {
      return std::nullopt;
    }

    // Get input shape to determine dimensionality.
    if (reduce_op.input_operand_id.value() >= operands_.size() ||
        !operands_[reduce_op.input_operand_id.value()]) {
      return std::nullopt;
    }
    const auto& input_shape =
        operands_[reduce_op.input_operand_id.value()]->descriptor.shape();
    const size_t rank = input_shape.size();

    // For shape folding, handle 1-D full reduction (the common case for
    // computing shape values like ReduceProduct of a shape tensor).
    if (rank == 1) {
      bool reduces_axis_0 =
          reduce_op.axes.empty() ||
          (reduce_op.axes.size() == 1 && reduce_op.axes[0] == 0);
      if (reduces_axis_0) {
        // Only kMean diverges between integer and real arithmetic here (it
        // averages); sum/product/min/max are exact over doubles for integral
        // inputs. See IsIntegerOperand's contract.
        const bool is_integer = IsIntegerOperand(output_operand_id);
        double acc = 0;
        switch (reduce_op.kind) {
          case mojom::Reduce::Kind::kSum:
            for (double v : *input_values) {
              acc += v;
            }
            break;
          case mojom::Reduce::Kind::kMean:
            for (double v : *input_values) {
              acc += v;
            }
            acc /= static_cast<double>(input_values->size());
            // Integer mean truncates toward zero (matches integer division).
            if (is_integer) {
              acc = std::trunc(acc);
            }
            break;
          case mojom::Reduce::Kind::kProduct:
            acc = 1;
            for (double v : *input_values) {
              acc *= v;
            }
            break;
          case mojom::Reduce::Kind::kMax:
            acc = (*input_values)[0];
            for (size_t i = 1; i < input_values->size(); ++i) {
              acc = std::max(acc, (*input_values)[i]);
            }
            break;
          case mojom::Reduce::Kind::kMin:
            acc = (*input_values)[0];
            for (size_t i = 1; i < input_values->size(); ++i) {
              acc = std::min(acc, (*input_values)[i]);
            }
            break;
          default:
            return std::nullopt;
        }
        // keep_dimensions does not change the single reduced value for a 1-D
        // full reduction.
        return std::vector<double>{acc};
      }
    }

    // Higher-rank or partial-axis reductions: not supported for shape folding.
    return std::nullopt;
  }

  // where — element-wise conditional selection:
  //   output[i] = condition[i] ? true_value[i] : false_value[i]
  if (operation.is_where()) {
    const auto& where_op = *operation.get_where();
    auto condition_values = Evaluate(where_op.condition_operand_id);
    auto true_values = Evaluate(where_op.true_value_operand_id);
    auto false_values = Evaluate(where_op.false_value_operand_id);
    if (!condition_values || !true_values || !false_values) {
      return std::nullopt;
    }

    // Handle broadcasting: determine result size and validate.
    size_t result_size =
        std::max({condition_values->size(), true_values->size(),
                  false_values->size()});
    auto broadcast_ok = [result_size](size_t s) {
      return s == result_size || s == 1;
    };
    if (!broadcast_ok(condition_values->size()) ||
        !broadcast_ok(true_values->size()) ||
        !broadcast_ok(false_values->size())) {
      return std::nullopt;
    }

    std::vector<double> result;
    result.reserve(result_size);
    for (size_t i = 0; i < result_size; ++i) {
      double cond = (*condition_values)[condition_values->size() == 1 ? 0 : i];
      double tv = (*true_values)[true_values->size() == 1 ? 0 : i];
      double fv = (*false_values)[false_values->size() == 1 ? 0 : i];
      result.push_back(cond != 0 ? tv : fv);
    }
    return result;
  }

  // expand — broadcast a 1-D shape vector to a concrete 1-D target length.
  // ONNX ConstantOfShape and broadcasts inside shape arithmetic lower to
  // mojom::Expand. The target length is read from the (already concretized in
  // infer mode) output operand shape rather than the op's `new_shape`
  // attribute, which may still carry symbolic dimensions.
  if (operation.is_expand()) {
    const auto& expand_op = *operation.get_expand();
    if (output_operand_id.value() >= operands_.size() ||
        !operands_[output_operand_id.value()]) {
      return std::nullopt;
    }
    const auto& out_shape =
        operands_[output_operand_id.value()]->descriptor.shape();
    if (out_shape.size() != 1 ||
        !std::holds_alternative<uint32_t>(out_shape[0])) {
      return std::nullopt;
    }
    const size_t target = std::get<uint32_t>(out_shape[0]);
    auto input_values = Evaluate(expand_op.input_operand_id);
    if (!input_values) {
      return std::nullopt;
    }
    if (input_values->size() == target) {
      return input_values;  // Identity broadcast.
    }
    if (input_values->size() == 1) {
      return std::vector<double>(target, (*input_values)[0]);
    }
    return std::nullopt;
  }

  // split — partition a 1-D shape vector along axis 0 and return the segment
  // that corresponds to the requested output. mojom::Split carries no explicit
  // split sizes, so per-segment lengths are taken from each output operand's
  // concrete length when available, falling back to an equal split otherwise.
  if (operation.is_split()) {
    const auto& split_op = *operation.get_split();

    // Locate which output this evaluation targets.
    size_t target_index = split_op.output_operand_ids.size();
    for (size_t i = 0; i < split_op.output_operand_ids.size(); ++i) {
      if (split_op.output_operand_ids[i] == output_operand_id) {
        target_index = i;
        break;
      }
    }
    if (target_index == split_op.output_operand_ids.size()) {
      return std::nullopt;
    }

    // Only 1-D shape vectors split along axis 0 are foldable; higher-rank
    // splits are not shape computations.
    if (split_op.input_operand_id.value() >= operands_.size() ||
        !operands_[split_op.input_operand_id.value()]) {
      return std::nullopt;
    }
    const auto& input_shape =
        operands_[split_op.input_operand_id.value()]->descriptor.shape();
    if (split_op.axis != 0 || input_shape.size() != 1) {
      return std::nullopt;
    }

    auto input_values = Evaluate(split_op.input_operand_id);
    if (!input_values) {
      return std::nullopt;
    }
    const size_t output_count = split_op.output_operand_ids.size();

    // Prefer per-output concrete lengths; if any is non-concrete, fall back to
    // an equal split of the input.
    std::vector<size_t> sizes;
    sizes.reserve(output_count);
    bool all_concrete = true;
    size_t total = 0;
    for (OperandId out_id : split_op.output_operand_ids) {
      if (out_id.value() >= operands_.size() || !operands_[out_id.value()]) {
        return std::nullopt;
      }
      const auto& out_shape = operands_[out_id.value()]->descriptor.shape();
      if (out_shape.size() == 1 &&
          std::holds_alternative<uint32_t>(out_shape[0])) {
        size_t seg = std::get<uint32_t>(out_shape[0]);
        sizes.push_back(seg);
        total += seg;
      } else {
        all_concrete = false;
        break;
      }
    }
    if (!all_concrete || total != input_values->size()) {
      if (output_count == 0 || input_values->size() % output_count != 0) {
        return std::nullopt;
      }
      sizes.assign(output_count, input_values->size() / output_count);
    }

    size_t offset = 0;
    for (size_t i = 0; i < target_index; ++i) {
      offset += sizes[i];
    }
    const size_t target_size = sizes[target_index];
    if (offset + target_size > input_values->size()) {
      return std::nullopt;
    }
    std::vector<double> result;
    result.reserve(target_size);
    for (size_t i = 0; i < target_size; ++i) {
      result.push_back((*input_values)[offset + i]);
    }
    return result;
  }

  // Unsupported operation — cannot evaluate.
  return std::nullopt;
}

}  // namespace webnn
