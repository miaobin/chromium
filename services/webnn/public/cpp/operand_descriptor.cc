// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/public/cpp/operand_descriptor.h"

#include <algorithm>
#include <numeric>

#include "base/containers/to_vector.h"
#include "base/numerics/checked_math.h"
#include "base/types/expected_macros.h"
#include "services/webnn/public/cpp/context_properties.h"
#include "services/webnn/public/cpp/webnn_errors.h"

namespace webnn {

namespace {

#define ASSIGN_OR_RETURN_ERROR_WITH_LABEL_IF_ERROR(lhs, rexpr, label) \
  ASSIGN_OR_RETURN(lhs, rexpr, [&label](std::string error) {          \
    return ErrorWithLabel(label, error);                              \
  });

base::expected<void, std::string> IsValidPermutation(
    base::span<const uint32_t> permutation,
    OperandDataType data_type,
    base::span<const Dimension> shape) {
  // TODO(crbug.com/428232161): Support sub-byte transposes.
  if (OperandDescriptor::GetBitsPerElement(data_type) < 8u) {
    return base::unexpected(
        "Invalid descriptor: Permutation is not supported for sub-byte data "
        "types.");
  }
  if (permutation.size() != shape.size()) {
    return base::unexpected(
        "Invalid descriptor: Permutation size doesn't match with shape.");
  }
  std::vector<uint32_t> sorted_permutation = base::ToVector(permutation);
  std::ranges::sort(sorted_permutation);
  for (size_t i = 0; i < sorted_permutation.size(); ++i) {
    if (sorted_permutation[i] != i) {
      return base::unexpected(
          "Invalid descriptor: Permutation contains invalid dimension.");
    }
  }
  return base::ok();
}

// Collects a fully-static shape from a dimension vector. Returns nullopt if any
// dimension is dynamic.
std::optional<std::vector<uint32_t>> ToStaticShapeVector(
    base::span<const Dimension> dimensions) {
  std::vector<uint32_t> shape;
  shape.reserve(dimensions.size());
  for (const auto& dim : dimensions) {
    const uint32_t* size = std::get_if<uint32_t>(&dim);
    if (!size) {
      return std::nullopt;
    }
    shape.push_back(*size);
  }
  return shape;
}

}  // namespace

std::vector<Dimension> ToDimensionVector(base::span<const uint32_t> shape) {
  std::vector<Dimension> dimension_shape;
  dimension_shape.reserve(shape.size());
  for (uint32_t dim : shape) {
    dimension_shape.push_back(dim);
  }
  return dimension_shape;
}

// static
base::expected<OperandDescriptor, std::string> OperandDescriptor::Create(
    const ContextProperties& context_properties,
    OperandDataType data_type,
    base::span<const uint32_t> shape,
    std::string_view label) {
  return Create(context_properties, data_type, ToDimensionVector(shape), label);
}

// static
base::expected<OperandDescriptor, std::string> OperandDescriptor::Create(
    const ContextProperties& context_properties,
    OperandDataType data_type,
    base::span<const Dimension> shape,
    std::string_view label) {
  const std::optional<std::vector<uint32_t>> shape_uint32 =
      ToStaticShapeVector(shape);
  // Only validate byte length for fully-static shapes. Dynamic shapes defer
  // size validation to dispatch, when concrete input sizes are known.
  if (shape_uint32.has_value()) {
    ASSIGN_OR_RETURN_ERROR_WITH_LABEL_IF_ERROR(
        uint64_t byte_length,
        ValidateAndGetByteLength(
            OperandDescriptor::GetBitsPerElement(data_type),
            base::span<const uint32_t>(*shape_uint32)),
        label);

    if (byte_length > context_properties.tensor_byte_length_limit) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedTensorSizeError(
                     byte_length,
                     context_properties.tensor_byte_length_limit)));
    }
  } else {
    // For dynamic shapes, still validate rank.
    if (shape.size() > 8) {
      return base::unexpected(ErrorWithLabel(
          label,
          "Invalid descriptor: The maximum rank of an operand is 8."));
    }
  }
  return OperandDescriptor(data_type, base::ToVector(shape));
}

// static
base::expected<OperandDescriptor, std::string>
OperandDescriptor::CreateForDeserialization(
    OperandDataType data_type,
    base::span<const uint32_t> shape,
    base::span<const uint32_t> pending_permutation) {
  return CreateForDeserialization(data_type, ToDimensionVector(shape),
                                  pending_permutation);
}

// static
base::expected<OperandDescriptor, std::string>
OperandDescriptor::CreateForDeserialization(
    OperandDataType data_type,
    base::span<const Dimension> shape,
    base::span<const uint32_t> pending_permutation) {
  const std::optional<std::vector<uint32_t>> shape_uint32 =
      ToStaticShapeVector(shape);
  if (shape_uint32.has_value()) {
    RETURN_IF_ERROR(ValidateAndGetByteLength(
        OperandDescriptor::GetBitsPerElement(data_type),
        base::span<const uint32_t>(*shape_uint32)));
  } else {
    // For dynamic shapes, still validate rank.
    if (shape.size() > 8) {
      return base::unexpected(
          "Invalid descriptor: The maximum rank of an operand is 8.");
    }
  }
  if (!pending_permutation.empty()) {
    RETURN_IF_ERROR(IsValidPermutation(pending_permutation, data_type, shape));
  }
  return OperandDescriptor(data_type, base::ToVector(shape),
                           base::ToVector(pending_permutation));
}

// static
OperandDescriptor OperandDescriptor::UnsafeCreateForTesting(
    OperandDataType data_type,
    base::span<const uint32_t> shape,
    base::span<const uint32_t> pending_permutation) {
  return OperandDescriptor(data_type, ToDimensionVector(shape),
                           base::ToVector(pending_permutation));
}

// static
OperandDescriptor OperandDescriptor::UnsafeCreateForTesting(
    OperandDataType data_type,
    base::span<const Dimension> shape,
    base::span<const uint32_t> pending_permutation) {
  return OperandDescriptor(data_type, base::ToVector(shape),
                           base::ToVector(pending_permutation));
}

// static
size_t OperandDescriptor::GetBitsPerElement(OperandDataType data_type) {
  switch (data_type) {
    case OperandDataType::kFloat32:
      return sizeof(float) * 8;
    case OperandDataType::kFloat16:
      return sizeof(uint16_t) * 8;
    case OperandDataType::kInt32:
      return sizeof(int32_t) * 8;
    case OperandDataType::kUint32:
      return sizeof(uint32_t) * 8;
    case OperandDataType::kInt64:
      return sizeof(int64_t) * 8;
    case OperandDataType::kUint64:
      return sizeof(uint64_t) * 8;
    case OperandDataType::kInt8:
      return sizeof(int8_t) * 8;
    case OperandDataType::kUint8:
      return sizeof(uint8_t) * 8;
    case OperandDataType::kInt4:
    case OperandDataType::kUint4:
      return 4;
  }
}

OperandDescriptor::OperandDescriptor(mojo::DefaultConstruct::Tag) {}

OperandDescriptor::OperandDescriptor()
    : data_type_(OperandDataType::kFloat32) {}

OperandDescriptor::OperandDescriptor(OperandDataType data_type,
                                     std::vector<Dimension> shape)
    : data_type_(data_type), shape_(std::move(shape)) {}

OperandDescriptor::OperandDescriptor(OperandDataType data_type,
                                     std::vector<Dimension> shape,
                                     std::vector<uint32_t> pending_permutation)
    : data_type_(data_type),
      shape_(std::move(shape)),
      pending_permutation_(std::move(pending_permutation)) {}

OperandDescriptor::OperandDescriptor(const OperandDescriptor&) = default;
OperandDescriptor& OperandDescriptor::operator=(const OperandDescriptor&) =
    default;
OperandDescriptor::OperandDescriptor(OperandDescriptor&&) noexcept = default;
OperandDescriptor& OperandDescriptor::operator=(OperandDescriptor&&) noexcept =
    default;

OperandDescriptor::~OperandDescriptor() = default;

size_t OperandDescriptor::PackedByteLength() const {
  // Overflow checks are not needed here because this same calculation is
  // performed with overflow checking in `Create()`. `this` would not exist if
  // those checks failed.
  auto num_elements = NumberOfElements();
  CHECK(num_elements.has_value())
      << "PackedByteLength() called on unbounded descriptor";
  base::CheckedNumeric<uint64_t> checked_number_of_bytes =
      (base::CheckedNumeric<uint64_t>(GetBitsPerElement(data_type_)) *
           *num_elements +
       7) /
      8;
  return checked_number_of_bytes.ValueOrDie<size_t>();
}

std::optional<size_t> OperandDescriptor::NumberOfElements() const {
  size_t result = 1;
  for (const auto& dim : shape_) {
    const uint32_t* size = std::get_if<uint32_t>(&dim);
    if (!size) {
      return std::nullopt;
    }
    base::CheckedNumeric<size_t> checked =
        base::CheckedNumeric<size_t>(result) * *size;
    if (!checked.AssignIfValid(&result)) {
      return std::nullopt;
    }
  }
  return result;
}

void OperandDescriptor::SetPendingPermutation(
    base::span<const uint32_t> permutation) {
  CHECK(IsValidPermutation(permutation, data_type_, shape_).has_value());
  pending_permutation_.assign(permutation.begin(), permutation.end());
}

std::optional<std::vector<uint32_t>> OperandDescriptor::StaticShape() const {
  return ToStaticShapeVector(shape_);
}
}  // namespace webnn
