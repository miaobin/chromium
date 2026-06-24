// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/public/mojom/operand_descriptor_mojom_traits.h"

#include "base/notreached.h"
#include "base/types/expected.h"

namespace mojo {

// static
webnn::OperandDataType StructTraits<webnn::mojom::OperandDescriptorDataView,
                                    webnn::OperandDescriptor>::
    data_type(const webnn::OperandDescriptor& descriptor) {
  return descriptor.data_type();
}

// static
bool StructTraits<webnn::mojom::OperandDescriptorDataView,
                  webnn::OperandDescriptor>::
    Read(webnn::mojom::OperandDescriptorDataView data,
         webnn::OperandDescriptor* out) {
  // A null `shape` denotes an unranked operand (dynamic rank).
  std::optional<std::vector<webnn::Dimension>> shape;
  if (!data.ReadShape(&shape)) {
    return false;
  }

  mojo::ArrayDataView<uint32_t> pending_permutation;
  data.GetPendingPermutationDataView(&pending_permutation);

  webnn::OperandDataType data_type;
  if (!data.ReadDataType(&data_type)) {
    return false;
  }

  if (!shape.has_value()) {
    // Unranked operands carry no permutation (only constants do, and constants
    // are always ranked).
    if (pending_permutation.size() != 0) {
      return false;
    }
    *out = webnn::OperandDescriptor::CreateUnranked(data_type);
    return true;
  }

  base::expected<webnn::OperandDescriptor, std::string> descriptor =
      webnn::OperandDescriptor::CreateForDeserialization(
          data_type, *shape, base::span(pending_permutation));

  if (!descriptor.has_value()) {
    return false;
  }

  *out = *std::move(descriptor);
  return true;
}

// static
webnn::mojom::DataType
EnumTraits<webnn::mojom::DataType, webnn::OperandDataType>::ToMojom(
    webnn::OperandDataType input) {
  switch (input) {
    case webnn::OperandDataType::kFloat32:
      return webnn::mojom::DataType::kFloat32;
    case webnn::OperandDataType::kFloat16:
      return webnn::mojom::DataType::kFloat16;
    case webnn::OperandDataType::kInt32:
      return webnn::mojom::DataType::kInt32;
    case webnn::OperandDataType::kUint32:
      return webnn::mojom::DataType::kUint32;
    case webnn::OperandDataType::kInt64:
      return webnn::mojom::DataType::kInt64;
    case webnn::OperandDataType::kUint64:
      return webnn::mojom::DataType::kUint64;
    case webnn::OperandDataType::kInt8:
      return webnn::mojom::DataType::kInt8;
    case webnn::OperandDataType::kUint8:
      return webnn::mojom::DataType::kUint8;
    case webnn::OperandDataType::kInt4:
      return webnn::mojom::DataType::kInt4;
    case webnn::OperandDataType::kUint4:
      return webnn::mojom::DataType::kUint4;
  }
  NOTREACHED();
}

// static
webnn::OperandDataType
EnumTraits<webnn::mojom::DataType, webnn::OperandDataType>::FromMojom(
    webnn::mojom::DataType input) {
  switch (input) {
    case webnn::mojom::DataType::kFloat32:
      return webnn::OperandDataType::kFloat32;
    case webnn::mojom::DataType::kFloat16:
      return webnn::OperandDataType::kFloat16;
    case webnn::mojom::DataType::kInt32:
      return webnn::OperandDataType::kInt32;
    case webnn::mojom::DataType::kUint32:
      return webnn::OperandDataType::kUint32;
    case webnn::mojom::DataType::kInt64:
      return webnn::OperandDataType::kInt64;
    case webnn::mojom::DataType::kUint64:
      return webnn::OperandDataType::kUint64;
    case webnn::mojom::DataType::kInt8:
      return webnn::OperandDataType::kInt8;
    case webnn::mojom::DataType::kUint8:
      return webnn::OperandDataType::kUint8;
    case webnn::mojom::DataType::kInt4:
      return webnn::OperandDataType::kInt4;
    case webnn::mojom::DataType::kUint4:
      return webnn::OperandDataType::kUint4;
  }
  NOTREACHED();
}

// static
webnn::mojom::DimensionDataView::Tag
UnionTraits<webnn::mojom::DimensionDataView, webnn::Dimension>::GetTag(
    const webnn::Dimension& dimension) {
  if (std::holds_alternative<uint32_t>(dimension)) {
    return webnn::mojom::DimensionDataView::Tag::kSize;
  }
  return webnn::mojom::DimensionDataView::Tag::kDynamicName;
}

// static
uint32_t UnionTraits<webnn::mojom::DimensionDataView, webnn::Dimension>::size(
    const webnn::Dimension& dimension) {
  return std::get<uint32_t>(dimension);
}

// static
const std::optional<std::string>&
UnionTraits<webnn::mojom::DimensionDataView, webnn::Dimension>::dynamic_name(
    const webnn::Dimension& dimension) {
  return std::get<webnn::DynamicDimension>(dimension).name;
}

// static
bool UnionTraits<webnn::mojom::DimensionDataView, webnn::Dimension>::Read(
    webnn::mojom::DimensionDataView data,
    webnn::Dimension* out) {
  switch (data.tag()) {
    case webnn::mojom::DimensionDataView::Tag::kSize:
      *out = data.size();
      return true;
    case webnn::mojom::DimensionDataView::Tag::kDynamicName: {
      std::optional<std::string> name;
      if (!data.ReadDynamicName(&name)) {
        return false;
      }
      *out = webnn::DynamicDimension{.name = std::move(name)};
      return true;
    }
  }
  return false;
}

}  // namespace mojo
