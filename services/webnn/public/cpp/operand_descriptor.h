// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_PUBLIC_CPP_OPERAND_DESCRIPTOR_H_
#define SERVICES_WEBNN_PUBLIC_CPP_OPERAND_DESCRIPTOR_H_

#include <algorithm>
#include <compare>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>  // For std::variant
#include <vector>

#include "base/component_export.h"
#include "base/containers/span.h"
#include "base/numerics/checked_math.h"
#include "base/types/expected.h"
#include "mojo/public/cpp/bindings/default_construct_tag.h"

namespace webnn {

// An unbounded dynamic dimension. `name` carries the symbolic name when the dim
// is named (mapped to an ONNX symbolic dimension); `nullopt` means an unnamed
// dynamic dim (ORT's `?`).
//
// `operator==` is purely structural (and therefore reflexive): it answers "are
// these the same descriptor?", which `OperandDescriptor`'s equality and Mojo
// round-trip checks rely on. It does NOT mean "do these resolve to the same
// runtime value" — two unnamed dims are independent unknowns even though they
// compare equal structurally. Code that needs runtime-value identity (e.g.
// reshape dimension cancellation) must additionally require `name.has_value()`.
struct DynamicDimension {
  std::optional<std::string> name;

  friend constexpr bool operator==(const DynamicDimension&,
                                   const DynamicDimension&) = default;
};

using Dimension = std::variant<uint32_t, DynamicDimension>;

// Helper to convert a uint32_t vector to a Dimension vector (all static
// dimensions).
std::vector<Dimension> COMPONENT_EXPORT(WEBNN_PUBLIC_CPP)
    ToDimensionVector(base::span<const uint32_t> shape);

enum class OperandDataType {
  kFloat32,
  kFloat16,
  kInt32,
  kUint32,
  kInt64,
  kUint64,
  kInt8,
  kUint8,
  kInt4,
  kUint4,

  kMinValue = kFloat32,
  kMaxValue = kUint4,
};

struct ContextProperties;

class COMPONENT_EXPORT(WEBNN_PUBLIC_CPP) OperandDescriptor {
 public:
  // Validates the tensor size limit and returns the byte length of the tensor.
  // The `T` must be an integral type. The `bits_per_element` should be the bit
  // size of the data type, such as from `GetBitsPerElement`.
  template <typename T>
  static base::expected<uint64_t, std::string> ValidateAndGetByteLength(
      size_t bits_per_element,
      base::span<const T> shape) {
    static_assert(std::is_integral_v<T>,
                  "Shape type must be an integral type.");
    // TODO(crbug.com/329482489): Specify the max rank of an operand. Consider
    // exposing different ranks for different backends (e.g. Core ML supports
    // only up to rank 5).
    if (shape.size() > 8) {
      return base::unexpected(
          "Invalid descriptor: The maximum rank of an operand is 8.");
    }

    // Enforce dimension range according to
    // https://www.w3.org/TR/webnn/#valid-dimension.
    if (std::ranges::any_of(shape, [](T dimension) {
          return !base::CheckedNumeric<int32_t>(dimension).IsValid();
        })) {
      return base::unexpected(
          "Invalid descriptor: All dimensions must be in the range of "
          "int32_t.");
    }

    base::CheckedNumeric<int32_t> checked_number_of_elements =
        std::accumulate(shape.begin(), shape.end(),
                        base::CheckedNumeric<int32_t>(1), std::multiplies());
    if (!checked_number_of_elements.IsValid()) {
      return base::unexpected(
          "Invalid descriptor: The number of elements is too large.");
    }

    // Since the data stored in memory are in 8-bits bytes, here we need to make
    // up an integer multiple of 8 to calculate the `checked_number_of_bytes`.
    base::CheckedNumeric<uint64_t> checked_number_of_bytes =
        (checked_number_of_elements.Cast<uint64_t>() * bits_per_element + 7) /
        8;

    size_t number_of_bytes;
    if (!checked_number_of_bytes.AssignIfValid(&number_of_bytes)) {
      return base::unexpected(
          "Invalid descriptor: The byte length is too large.");
    }

    if (number_of_bytes == 0) {
      // TODO(crbug.com/329471677): Consider supporting size 0 dimensions.
      return base::unexpected(
          "Invalid descriptor: All dimensions should be positive.");
    }

    return number_of_bytes;
  }

  // Creates a valid `OperandDescriptor` or returns an error message which may
  // be returned to script if the inputs are not valid.
  static base::expected<OperandDescriptor, std::string> Create(
      const ContextProperties& context_properties,
      OperandDataType data_type,
      base::span<const uint32_t> shape,
      std::string_view label);

  // Creates a valid `OperandDescriptor` with dynamic dimensions.
  static base::expected<OperandDescriptor, std::string> Create(
      const ContextProperties& context_properties,
      OperandDataType data_type,
      base::span<const Dimension> shape,
      std::string_view label);

  // This function is called by `OperandDescriptor` mojom traits that need to be
  // validated tensor size limit later.
  static base::expected<OperandDescriptor, std::string>
  CreateForDeserialization(OperandDataType data_type,
                           base::span<const uint32_t> shape,
                           base::span<const uint32_t> pending_permutation);

  // Same as above, but support dynamic dimensions.
  static base::expected<OperandDescriptor, std::string>
  CreateForDeserialization(OperandDataType data_type,
                           base::span<const Dimension> shape,
                           base::span<const uint32_t> pending_permutation);

  // Creates an unranked descriptor (rank unknown). Used for dynamic-rank graphs
  // (Phase B): the rank is resolved to a concrete value at
  // computeShapes/dispatch. Unranked operands carry only a data type.
  static OperandDescriptor CreateUnranked(OperandDataType data_type);

  // Same as above, but skip validation checks. This may be used to create an
  // invalid descriptor to test that its deserialization fails.
  static OperandDescriptor UnsafeCreateForTesting(
      OperandDataType data_type,
      base::span<const uint32_t> shape,
      base::span<const uint32_t> pending_permutation = {});

  // Overload that accepts Dimension span for testing with dynamic dimensions.
  static OperandDescriptor UnsafeCreateForTesting(
      OperandDataType data_type,
      base::span<const Dimension> shape,
      base::span<const uint32_t> pending_permutation = {});

  static size_t GetBitsPerElement(OperandDataType data_type);

  // Creates an invalid instance for use with Mojo deserialization, which
  // requires types to be default-constructible.
  explicit OperandDescriptor(mojo::DefaultConstruct::Tag);

  // Creates an instance with default-initialized data_type_, needed for
  // WTF::HashMap value type support in blink mojom bindings.
  OperandDescriptor();

  // Copyable and movable.
  OperandDescriptor(const OperandDescriptor&);
  OperandDescriptor& operator=(const OperandDescriptor&);
  OperandDescriptor(OperandDescriptor&&) noexcept;
  OperandDescriptor& operator=(OperandDescriptor&&) noexcept;

  ~OperandDescriptor();

  OperandDataType data_type() const { return data_type_; }
  // Returns the shape's dimensions. CHECK-fails if this descriptor is unranked
  // (rank unknown); callers that may handle unranked operands must first guard
  // on `HasRank()`. Per Phase B, the rank is known or unknown (binary); code
  // paths that have already established a known rank (the common case) keep
  // using `shape()` directly.
  const std::vector<Dimension>& shape() const {
    CHECK(shape_.has_value()) << "shape() called on unranked descriptor";
    return *shape_;
  }
  const std::vector<uint32_t>& pending_permutation() const {
    return pending_permutation_;
  }

  // Whether this descriptor has a known rank. An unranked descriptor (rank
  // unknown) is only produced for dynamic-rank graphs (Phase B) and is resolved
  // to a concrete rank at computeShapes/dispatch.
  bool HasRank() const { return shape_.has_value(); }

  // Returns the shape as an optional (nullopt when unranked). Used by the Mojo
  // traits, which serialize the shape as a nullable array.
  const std::optional<std::vector<Dimension>>& shape_optional() const {
    return shape_;
  }

  // Returns the rank, or nullopt if the descriptor is unranked. Callers must
  // handle the unranked case explicitly (defer to dispatch) before relying on
  // a concrete rank.
  std::optional<uint32_t> Rank() const {
    if (!shape_.has_value()) {
      return std::nullopt;
    }
    return static_cast<uint32_t>(shape_->size());
  }
  // Total byte length assuming perfect packing. Some tensors described by this
  // `OperandDescriptor` may be stored with more bytes.
  // CHECK-fails if the shape has any dynamic dimension (tensor descriptors are
  // always static by design).
  size_t PackedByteLength() const;
  // Returns the total number of elements, or nullopt if any dimension is
  // dynamic.
  std::optional<size_t> NumberOfElements() const;

  void SetPendingPermutation(base::span<const uint32_t> permutation);

  // Returns the shape as a vector of uint32_t, returns nullopt if any dynamic
  // dimension is present.
  std::optional<std::vector<uint32_t>> StaticShape() const;

  // Note: only equality is defined. `Dimension` (specifically the unnamed
  // dynamic dim) has no total order, and `OperandDescriptor` is only ever used
  // as a `flat_map` value, never as a key, so no ordering is needed.
  friend constexpr bool operator==(const OperandDescriptor& lhs,
                                   const OperandDescriptor& rhs) {
    return lhs.data_type_ == rhs.data_type_ && lhs.shape_ == rhs.shape_;
  }

 private:
  OperandDescriptor(OperandDataType data_type, std::vector<Dimension> shape);
  OperandDescriptor(OperandDataType data_type,
                    std::vector<Dimension> shape,
                    std::vector<uint32_t> permutation);

  OperandDataType data_type_;
  // `nullopt` means the descriptor is unranked (rank unknown). A present-but-
  // empty vector is a scalar (rank 0).
  std::optional<std::vector<Dimension>> shape_;
  std::vector<uint32_t> pending_permutation_;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_PUBLIC_CPP_OPERAND_DESCRIPTOR_H_
