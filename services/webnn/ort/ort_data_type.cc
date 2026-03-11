// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/ort/ort_data_type.h"

namespace webnn::ort {

ONNXTensorElementDataType WebnnToOnnxDataType(OperandDataType data_type) {
  switch (data_type) {
    case OperandDataType::kFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case OperandDataType::kFloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case OperandDataType::kInt32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case OperandDataType::kUint32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case OperandDataType::kInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case OperandDataType::kUint64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case OperandDataType::kInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case OperandDataType::kUint8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case OperandDataType::kInt4:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;
    case OperandDataType::kUint4:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4;
  }
}

std::vector<int64_t> WebnnToOnnxShape(
    base::span<const webnn::Dimension> shape) {
  std::vector<int64_t> onnx_shape;
  onnx_shape.reserve(shape.size());
  for (const auto& dim : shape) {
    if (std::holds_alternative<uint32_t>(dim)) {
      onnx_shape.push_back(std::get<uint32_t>(dim));
    } else {
      onnx_shape.push_back(-1);
    }
  }
  return onnx_shape;
}

}  // namespace webnn::ort
