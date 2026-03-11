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
  std::vector<webnn::Dimension> shape;
  if (!data.ReadShape(&shape)) {
    return false;
  }

  mojo::ArrayDataView<uint32_t> pending_permutation;
  data.GetPendingPermutationDataView(&pending_permutation);

  webnn::OperandDataType data_type;
  if (!data.ReadDataType(&data_type)) {
    return false;
  }
  base::expected<webnn::OperandDescriptor, std::string> descriptor =
      webnn::OperandDescriptor::CreateForDeserialization(
          data_type, shape, base::span(pending_permutation));

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
  return webnn::mojom::DimensionDataView::Tag::kDynamicDimension;
}

// static
uint32_t UnionTraits<webnn::mojom::DimensionDataView, webnn::Dimension>::size(
    const webnn::Dimension& dimension) {
  return std::get<uint32_t>(dimension);
}

// static
const webnn::DynamicDimension& UnionTraits<
    webnn::mojom::DimensionDataView,
    webnn::Dimension>::dynamic_dimension(const webnn::Dimension& dimension) {
  return std::get<webnn::DynamicDimension>(dimension);
}

// static
bool UnionTraits<webnn::mojom::DimensionDataView, webnn::Dimension>::Read(
    webnn::mojom::DimensionDataView data,
    webnn::Dimension* out) {
  switch (data.tag()) {
    case webnn::mojom::DimensionDataView::Tag::kSize:
      *out = data.size();
      return true;
    case webnn::mojom::DimensionDataView::Tag::kDynamicDimension: {
      webnn::DynamicDimension dynamic_dimension;
      if (!data.ReadDynamicDimension(&dynamic_dimension)) {
        return false;
      }
      *out = std::move(dynamic_dimension);
      return true;
    }
  }
  return false;
}

// static
bool StructTraits<
    webnn::mojom::DynamicDimensionDataView,
    webnn::DynamicDimension>::Read(webnn::mojom::DynamicDimensionDataView data,
                                   webnn::DynamicDimension* out) {
  if (!data.ReadName(&out->name)) {
    return false;
  }
  out->max_size = data.max_size();
  out->min_size = data.min_size();
  return true;
}

}  // namespace mojo
