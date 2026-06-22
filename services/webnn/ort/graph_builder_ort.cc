// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/ort/graph_builder_ort.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <ranges>

#include "base/notreached.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/string_util.h"
#include "base/types/expected_macros.h"
#include "base/types/fixed_array.h"
#include "services/webnn/ort/ort_data_type.h"
#include "services/webnn/public/cpp/graph_validation_utils.h"
#include "services/webnn/public/cpp/supported_data_types.h"
#include "services/webnn/public/mojom/webnn_error.mojom.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"
#include "services/webnn/webnn_constant_operand.h"
#include "third_party/fp16/src/include/fp16.h"

namespace webnn::ort {

namespace {

// ArgMin/Max ops
constexpr base::cstring_view kOpTypeArgMin = "ArgMin";
constexpr base::cstring_view kOpTypeArgMax = "ArgMax";

// Element-wise binary ops
constexpr base::cstring_view kOpTypeAdd = "Add";
constexpr base::cstring_view kOpTypeSub = "Sub";
constexpr base::cstring_view kOpTypeMul = "Mul";
constexpr base::cstring_view kOpTypeDiv = "Div";
constexpr base::cstring_view kOpTypeMod = "Mod";
constexpr base::cstring_view kOpTypeMax = "Max";
constexpr base::cstring_view kOpTypeMin = "Min";
constexpr base::cstring_view kOpTypePow = "Pow";
constexpr base::cstring_view kOpTypeEqual = "Equal";
constexpr base::cstring_view kOpTypeGreater = "Greater";
constexpr base::cstring_view kOpTypeGreaterOrEqual = "GreaterOrEqual";
constexpr base::cstring_view kOpTypeLesser = "Less";
constexpr base::cstring_view kOpTypeLesserOrEqual = "LessOrEqual";
constexpr base::cstring_view kOpTypeLogicalAnd = "And";
constexpr base::cstring_view kOpTypeLogicalOr = "Or";
constexpr base::cstring_view kOpTypeLogicalXor = "Xor";

// Element-wise unary ops
constexpr base::cstring_view kOpTypeAbs = "Abs";
constexpr base::cstring_view kOpTypeCeil = "Ceil";
constexpr base::cstring_view kOpTypeCos = "Cos";
constexpr base::cstring_view kOpTypeExp = "Exp";
constexpr base::cstring_view kOpTypeFloor = "Floor";
constexpr base::cstring_view kOpTypeLog = "Log";
constexpr base::cstring_view kOpTypeIsNaN = "IsNaN";
constexpr base::cstring_view kOpTypeIsInfinite = "IsInf";
constexpr base::cstring_view kOpTypeLogicalNot = "Not";
constexpr base::cstring_view kOpTypeNeg = "Neg";
constexpr base::cstring_view kOpTypeRoundEven = "Round";
constexpr base::cstring_view kOpTypeSign = "Sign";
constexpr base::cstring_view kOpTypeSin = "Sin";
constexpr base::cstring_view kOpTypeTan = "Tan";
constexpr base::cstring_view kOpTypeIdentity = "Identity";
constexpr base::cstring_view kOpTypeSqrt = "Sqrt";
constexpr base::cstring_view kOpTypeErf = "Erf";
constexpr base::cstring_view kOpTypeReciprocal = "Reciprocal";
constexpr base::cstring_view kOpTypeCast = "Cast";

constexpr base::cstring_view kOpTypeBatchNormalization = "BatchNormalization";
constexpr base::cstring_view kOpTypeClamp = "Clip";
constexpr base::cstring_view kOpTypeConcat = "Concat";
constexpr base::cstring_view kOpTypeConv2d = "Conv";
constexpr base::cstring_view kOpTypeConvTranspose2d = "ConvTranspose";
constexpr base::cstring_view kOpTypeCumulativeSum = "CumSum";
constexpr base::cstring_view kOpTypeDequantizeLinear = "DequantizeLinear";
constexpr base::cstring_view kOpTypeElu = "Elu";
constexpr base::cstring_view kOpTypeExpand = "Expand";
constexpr base::cstring_view kOpTypeGather = "Gather";
constexpr base::cstring_view kOpTypeGatherElements = "GatherElements";
constexpr base::cstring_view kOpTypeGatherND = "GatherND";
constexpr base::cstring_view kOpTypeGelu = "Gelu";
constexpr base::cstring_view kOpTypeGemm = "Gemm";
constexpr base::cstring_view kOpTypeGru = "GRU";
constexpr base::cstring_view kOpTypeHardSigmoid = "HardSigmoid";
constexpr base::cstring_view kOpTypeHardSwish = "HardSwish";
constexpr base::cstring_view kOpTypeInstanceNormalization =
    "InstanceNormalization";
constexpr base::cstring_view kOpTypeLayerNormalization = "LayerNormalization";
constexpr base::cstring_view kOpTypeLeakyRelu = "LeakyRelu";
constexpr base::cstring_view kOpTypeLstm = "LSTM";
constexpr base::cstring_view kOpTypeMatMul = "MatMul";
constexpr base::cstring_view kOpTypePad = "Pad";
constexpr base::cstring_view kOpTypePRelu = "PRelu";
constexpr base::cstring_view kOpTypeQuantizeLinear = "QuantizeLinear";
constexpr base::cstring_view kOpTypeRelu = "Relu";
constexpr base::cstring_view kOpTypeResize = "Resize";
constexpr base::cstring_view kOpTypeReshape = "Reshape";
constexpr base::cstring_view kOpTypeScatterElements = "ScatterElements";
constexpr base::cstring_view kOpTypeShape = "Shape";
constexpr base::cstring_view kOpTypeSqueeze = "Squeeze";
constexpr base::cstring_view kOpTypeUnsqueeze = "Unsqueeze";
constexpr base::cstring_view kOpTypeScatterND = "ScatterND";
constexpr base::cstring_view kOpTypeSigmoid = "Sigmoid";
constexpr base::cstring_view kOpTypeSlice = "Slice";
constexpr base::cstring_view kOpTypeSoftmax = "Softmax";
constexpr base::cstring_view kOpTypeSoftplus = "Softplus";
constexpr base::cstring_view kOpTypeSoftsign = "Softsign";
constexpr base::cstring_view kOpTypeSplit = "Split";
constexpr base::cstring_view kOpTypeTanh = "Tanh";
constexpr base::cstring_view kOpTypeTile = "Tile";
constexpr base::cstring_view kOpTypeTranspose = "Transpose";
constexpr base::cstring_view kOpTypeTriangular = "Trilu";
constexpr base::cstring_view kOpTypeWhere = "Where";

// Pooling operations
constexpr base::cstring_view kOpTypeAveragePool2d = "AveragePool";
constexpr base::cstring_view kOpTypeMaxPool2d = "MaxPool";
constexpr base::cstring_view kOpTypeLpPool2d = "LpPool";

// Reduction operations
constexpr base::cstring_view kOpTypeReduceL1 = "ReduceL1";
constexpr base::cstring_view kOpTypeReduceL2 = "ReduceL2";
constexpr base::cstring_view kOpTypeReduceLogSum = "ReduceLogSum";
constexpr base::cstring_view kOpTypeReduceLogSumExp = "ReduceLogSumExp";
constexpr base::cstring_view kOpTypeReduceMax = "ReduceMax";
constexpr base::cstring_view kOpTypeReduceMean = "ReduceMean";
constexpr base::cstring_view kOpTypeReduceMin = "ReduceMin";
constexpr base::cstring_view kOpTypeReduceProd = "ReduceProd";
constexpr base::cstring_view kOpTypeReduceSum = "ReduceSum";
constexpr base::cstring_view kOpTypeReduceSumSquare = "ReduceSumSquare";

// Attributes
constexpr base::cstring_view kAttrActivations = "activations";
constexpr base::cstring_view kAttrAlpha = "alpha";
constexpr base::cstring_view kAttrAxis = "axis";
constexpr base::cstring_view kAttrBeta = "beta";
constexpr base::cstring_view kAttrBlockSize = "block_size";
constexpr base::cstring_view kAttrCeilMode = "ceil_mode";
constexpr base::cstring_view kAttrDilations = "dilations";
constexpr base::cstring_view kAttrDirection = "direction";
constexpr base::cstring_view kAttrEpsilon = "epsilon";
constexpr base::cstring_view kAttrExclusive = "exclusive";
constexpr base::cstring_view kAttrGroup = "group";
constexpr base::cstring_view kAttrHiddenSize = "hidden_size";
constexpr base::cstring_view kAttrKeepDims = "keepdims";
constexpr base::cstring_view kAttrKernelShape = "kernel_shape";
constexpr base::cstring_view kAttrLinearBeforeReset = "linear_before_reset";
constexpr base::cstring_view kAttrMode = "mode";
constexpr base::cstring_view kAttrNoopWithEmptyAxes = "noop_with_empty_axes";
constexpr base::cstring_view kAttrNumOutputs = "num_outputs";
constexpr base::cstring_view kAttrOutputPadding = "output_padding";
constexpr base::cstring_view kAttrP = "p";
constexpr base::cstring_view kAttrPads = "pads";
constexpr base::cstring_view kAttrPerm = "perm";
constexpr base::cstring_view kAttrReverse = "reverse";
constexpr base::cstring_view kAttrStrides = "strides";
constexpr base::cstring_view kAttrTo = "to";
constexpr base::cstring_view kAttrTransA = "transA";
constexpr base::cstring_view kAttrTransB = "transB";
constexpr base::cstring_view kAttrUpper = "upper";

constexpr base::cstring_view kInserted = "Inserted";
constexpr base::cstring_view kToEmulate = "ToEmulate";
constexpr base::cstring_view kUnderscore = "_";
constexpr std::string_view kNullCharacter("\0", 1);

std::string SanitizeName(std::string_view name) {
  std::string sanitized_name(name);
  base::ReplaceChars(sanitized_name, kNullCharacter, kUnderscore,
                     &sanitized_name);
  return sanitized_name;
}

std::string GetOperandName(std::string_view name, OperandId id) {
  // ORT CreateValueInfo API rejects name starting with null character:
  // https://github.com/microsoft/onnxruntime/blob/7b5a93ef5f71ca58a1b6e4ae81b250e767756c68/onnxruntime/core/session/model_editor_c_api.cc#L29
  return base::JoinString(
      {SanitizeName(name), base::NumberToString(id.value())}, kUnderscore);
}

// Maps a DataType to a `ONNXTensorElementDataType`. Other `TensorTypeMap`
// overloads may be declared below as needed.
//
// Example: TensorTypeMap<uint32_t>::value ->
// ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
template <typename DataType>
  requires internal::IsSupportedTensorType<DataType>
struct TensorTypeMap;

template <>
struct TensorTypeMap<float> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

// Use uint16_t to carry bits of float16.
template <>
struct TensorTypeMap<uint16_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
};

template <>
struct TensorTypeMap<int32_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
};

template <>
struct TensorTypeMap<uint32_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
};

template <>
struct TensorTypeMap<int64_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

template <>
struct TensorTypeMap<uint64_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
};

template <>
struct TensorTypeMap<int8_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
};

template <>
struct TensorTypeMap<uint8_t> {
  static constexpr ONNXTensorElementDataType value =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
};

// Calculate the output_padding according to the ONNX ConvTranspose2d
// documentation:
// https://onnx.ai/onnx/operators/onnx__ConvTranspose.html#summary
int64_t CalculateOutputPaddingSize(int64_t input_size,
                                   int64_t filter_size,
                                   int64_t stride,
                                   int64_t dilation,
                                   int64_t pad_begin,
                                   int64_t pad_end,
                                   int64_t output_size) {
  const auto output_padding =
      base::CheckedNumeric(output_size) - stride * (input_size - 1) -
      ((filter_size - 1) * dilation + 1) + pad_begin + pad_end;
  // `output_padding` is validated by
  // `ValidateAndCalculateConvTranspose2dOutputSizes()`. Because Conv2d mojo
  // struct doesn't include `output_padding`, for ORT backend, we need to
  // re-compute it by using other attributes.
  CHECK(output_padding.IsValid());
  return output_padding.ValueOrDie();
}

void CheckReduceInputSupported(const DataTypeLimits& data_type_limits,
                               mojom::Reduce::Kind kind,
                               const OperandDescriptor& input_descriptor) {
  switch (kind) {
    case mojom::Reduce::Kind::kL1:
      CHECK(data_type_limits.reduce_l1_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kL2:
      CHECK(data_type_limits.reduce_l2_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kLogSum:
      CHECK(data_type_limits.reduce_log_sum_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kLogSumExp:
      CHECK(
          data_type_limits.reduce_log_sum_exp_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kMax:
      CHECK(data_type_limits.reduce_max_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kMean:
      CHECK(data_type_limits.reduce_mean_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kMin:
      CHECK(data_type_limits.reduce_min_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kProduct:
      CHECK(data_type_limits.reduce_product_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kSum:
      CHECK(data_type_limits.reduce_sum_input.Supports(input_descriptor));
      break;
    case mojom::Reduce::Kind::kSumSquare:
      CHECK(
          data_type_limits.reduce_sum_square_input.Supports(input_descriptor));
      break;
  }
}

base::cstring_view MapReduceKindToOrtOpType(mojom::Reduce::Kind kind) {
  switch (kind) {
    case mojom::Reduce::Kind::kL1:
      return kOpTypeReduceL1;
    case mojom::Reduce::Kind::kL2:
      return kOpTypeReduceL2;
    case mojom::Reduce::Kind::kLogSum:
      return kOpTypeReduceLogSum;
    case mojom::Reduce::Kind::kLogSumExp:
      return kOpTypeReduceLogSumExp;
    case mojom::Reduce::Kind::kMax:
      return kOpTypeReduceMax;
    case mojom::Reduce::Kind::kMean:
      return kOpTypeReduceMean;
    case mojom::Reduce::Kind::kMin:
      return kOpTypeReduceMin;
    case mojom::Reduce::Kind::kProduct:
      return kOpTypeReduceProd;
    case mojom::Reduce::Kind::kSum:
      return kOpTypeReduceSum;
    case mojom::Reduce::Kind::kSumSquare:
      return kOpTypeReduceSumSquare;
  }
}

const std::vector<base::cstring_view> GetRecurrentNetworkActivations(
    std::vector<mojom::RecurrentNetworkActivation> activations,
    bool is_bidirectional) {
  std::vector<base::cstring_view> activation_list;
  for (const auto& activation : activations) {
    switch (activation) {
      case mojom::RecurrentNetworkActivation::kRelu:
        activation_list.push_back("Relu");
        break;
      case mojom::RecurrentNetworkActivation::kSigmoid:
        activation_list.push_back("Sigmoid");
        break;
      case mojom::RecurrentNetworkActivation::kTanh:
        activation_list.push_back("Tanh");
        break;
      default:
        NOTREACHED() << "Unsupported recurrent network activation function.";
    }
  }
  if (is_bidirectional) {
    activation_list.insert(activation_list.end(), activation_list.begin(),
                           activation_list.end());
  }
  return activation_list;
}

const base::cstring_view GetRecurrentNetworkDirection(
    mojom::RecurrentNetworkDirection direction) {
  switch (direction) {
    case mojom::RecurrentNetworkDirection::kForward:
      return "forward";
    case mojom::RecurrentNetworkDirection::kBackward:
      return "reverse";
    case mojom::RecurrentNetworkDirection::kBoth:
      return "bidirectional";
    default:
      NOTREACHED() << "Unsupported recurrent network activation direction.";
  }
}

}  // namespace

// static
base::expected<std::unique_ptr<ModelEditor::ModelInfo>, mojom::ErrorPtr>
GraphBuilderOrt::CreateAndBuild(
    const mojom::GraphInfo& graph_info,
    ContextProperties context_properties,
    base::flat_map<OperandId, std::unique_ptr<WebNNConstantOperand>>
        constant_operands,
    std::optional<uint32_t> batched_matmul_k_dimension_limit) {
  GraphBuilderOrt graph_builder(graph_info, std::move(context_properties),
                                std::move(constant_operands),
                                std::move(batched_matmul_k_dimension_limit));
  return graph_builder.BuildModel();
}

DynamicDimensionInfo::DynamicDimensionInfo() = default;
DynamicDimensionInfo::DynamicDimensionInfo(std::string input_operand_name,
                                           uint32_t axis)
    : input_operand_name(std::move(input_operand_name)), axis(axis) {}
DynamicDimensionInfo::~DynamicDimensionInfo() = default;
DynamicDimensionInfo::DynamicDimensionInfo(const DynamicDimensionInfo&) =
    default;
DynamicDimensionInfo& DynamicDimensionInfo::operator=(
    const DynamicDimensionInfo&) = default;
DynamicDimensionInfo::DynamicDimensionInfo(DynamicDimensionInfo&&) = default;
DynamicDimensionInfo& DynamicDimensionInfo::operator=(DynamicDimensionInfo&&) =
    default;

GraphBuilderOrt::GraphBuilderOrt(
    const mojom::GraphInfo& graph_info,
    ContextProperties context_properties,
    base::flat_map<OperandId, std::unique_ptr<WebNNConstantOperand>>
        constant_operands,
    std::optional<uint32_t> batched_matmul_k_dimension_limit)
    : graph_info_(graph_info),
      constant_operands_(std::move(constant_operands)),
      context_properties_(std::move(context_properties)),
      batched_matmul_k_dimension_limit_(
          std::move(batched_matmul_k_dimension_limit)) {
  // Register dynamic dimensions from graph input operands. Dynamic dims that
  // first appear on intermediate operands (e.g. the output of concat) are
  // registered lazily in BuildModel() as each operation is processed, so that
  // only operands that have already been produced are used as Shape sources.
  for (OperandId input_operand_id : graph_info_->input_operands) {
    RegisterOperandDynamicDims(input_operand_id);
  }
}

GraphBuilderOrt::~GraphBuilderOrt() = default;

const mojom::Operand& GraphBuilderOrt::GetOperand(OperandId operand_id) const {
  return *graph_info_->operands.at(operand_id.value());
}

std::string GraphBuilderOrt::GetOperandNameById(OperandId operand_id) const {
  const mojom::Operand& operand = GetOperand(operand_id);
  return GetOperandName(operand.name.has_value() ? *operand.name : "",
                        operand_id);
}

std::string GraphBuilderOrt::GenerateNodeName(std::string_view label) {
  return base::JoinString(
      {SanitizeName(label), base::NumberToString(next_operation_id_++)},
      kUnderscore);
}

std::string GraphBuilderOrt::GenerateEmulatedOpLabel(
    base::cstring_view op_type,
    std::string_view original_label,
    std::string_view additional_tag) {
  return base::JoinString({kInserted, op_type, additional_tag, kToEmulate,
                           SanitizeName(original_label)},
                          kUnderscore);
}

std::string GraphBuilderOrt::GenerateOperandName() {
  next_operand_id_++;
  CHECK(next_operand_id_.IsValid());
  return base::JoinString(
      {kInserted, base::NumberToString(
                      static_cast<uint32_t>(next_operand_id_.ValueOrDie()))},
      kUnderscore);
}

template <typename DataType>
  requires internal::IsSupportedTensorType<DataType>
std::string GraphBuilderOrt::CreateInitializer(
    base::span<const int64_t> shape,
    base::span<const DataType> data) {
  std::string name = GenerateOperandName();
  base::span<const uint8_t> byte_span;
  if constexpr (std::floating_point<DataType>) {
    // Floating point types do not have unique object representations, but
    // this code appears to be using a byte span to type-erase, which is fine.
    byte_span = base::as_byte_span(base::allow_nonunique_obj, data);
  } else {
    byte_span = base::as_byte_span(data);
  }

  model_editor_.AddInitializer(name, TensorTypeMap<DataType>::value, shape,
                               byte_span);
  return name;
}

template <typename DataType>
  requires internal::IsSupportedTensorType<DataType>
std::string GraphBuilderOrt::CreateScalarInitializer(const DataType& value) {
  return CreateInitializer<DataType>(
      /*shape=*/{}, base::span_from_ref(value));
}

template <typename DataType>
  requires internal::IsSupportedTensorType<DataType>
std::string GraphBuilderOrt::Create1DInitializer(
    base::span<const DataType> data) {
  std::array<int64_t, 1> shape = {base::checked_cast<int64_t>(data.size())};
  return CreateInitializer<DataType>(shape, data);
}

std::string GraphBuilderOrt::CreateInt64InitializerForUint32Array(
    base::span<const uint32_t> array) {
  std::array<int64_t, 1> array_dims = {
      base::checked_cast<int64_t>(array.size())};
  base::FixedArray<int64_t> array_value(array.begin(), array.end());
  return CreateInitializer<int64_t>(array_dims, array_value);
}

std::string GraphBuilderOrt::CreateInitializerWithInputShapeAndDataTypeForFloat(
    OperandId input_operand_id,
    float value,
    std::optional<base::span<const uint32_t>> axes) {
  const std::string input_name = GetOperandNameById(input_operand_id);
  const OperandDataType input_data_type =
      GetOperand(input_operand_id).descriptor.data_type();

  // Step 1: Create a scalar initializer with the input's data type.
  const std::string scalar =
      CreateScalarInitializer(input_data_type, MLNumber::FromFloat64(value));

  // Step 2: Get the input shape at runtime using Shape operator.
  const std::string input_shape_name = GenerateOperandName();
  {
    std::array<const char*, 1> shape_inputs = {input_name.c_str()};
    std::array<const char*, 1> shape_outputs = {input_shape_name.c_str()};
    const std::string shape_node_name = GenerateNodeName(
        base::JoinString({kInserted, kOpTypeShape}, kUnderscore));
    model_editor_.AddNode(kOpTypeShape, shape_node_name, shape_inputs,
                          shape_outputs);
  }

  // Step 3: If axes is specified, gather only those dimensions from input
  // shape.
  std::string target_shape = input_shape_name;
  if (axes.has_value()) {
    base::FixedArray<int64_t> axes_int64(axes->size());
    std::ranges::transform(*axes, axes_int64.begin(), [](uint32_t axis) {
      return static_cast<int64_t>(axis);
    });
    const std::string axes_initializer =
        Create1DInitializer<int64_t>(axes_int64);

    target_shape = GenerateOperandName();
    std::array<const char*, 2> gather_inputs = {input_shape_name.c_str(),
                                                axes_initializer.c_str()};
    std::array<const char*, 1> gather_outputs = {target_shape.c_str()};
    std::array<ScopedOrtOpAttr, 1> gather_attrs = {
        model_editor_.CreateAttribute(kAttrAxis, static_cast<int64_t>(0))};
    const std::string gather_node_name = GenerateNodeName(
        base::JoinString({kInserted, kOpTypeGather}, kUnderscore));
    model_editor_.AddNode(kOpTypeGather, gather_node_name, gather_inputs,
                          gather_outputs, gather_attrs);
  }

  // Step 4: Use Expand to expand the scalar to match the target shape.
  const std::string output = GenerateOperandName();
  {
    std::array<const char*, 2> expand_inputs = {scalar.c_str(),
                                                target_shape.c_str()};
    std::array<const char*, 1> expand_outputs = {output.c_str()};
    const std::string expand_node_name = GenerateNodeName(
        base::JoinString({kInserted, kOpTypeExpand}, kUnderscore));
    model_editor_.AddNode(kOpTypeExpand, expand_node_name, expand_inputs,
                          expand_outputs);
  }

  return output;
}

std::string GraphBuilderOrt::CreateInitializerForFloat(
    OperandDataType data_type,
    base::span<const uint32_t> shape,
    float value) {
  base::CheckedNumeric<size_t> checked_operand_size =
      std::accumulate(shape.begin(), shape.end(),
                      base::CheckedNumeric<size_t>(1), std::multiplies());
  size_t operand_size = checked_operand_size.ValueOrDie();
  base::FixedArray<int64_t> int64_shape(shape.begin(), shape.end());
  switch (data_type) {
    case OperandDataType::kFloat32: {
      base::FixedArray<float> data(operand_size, value);
      return CreateInitializer<float>(int64_shape, data);
    }
    case OperandDataType::kFloat16: {
      base::FixedArray<uint16_t> data(operand_size,
                                      fp16_ieee_from_fp32_value(value));
      return CreateInitializer<uint16_t>(int64_shape, data);
    }
    case OperandDataType::kInt32: {
      base::FixedArray<int32_t> data(operand_size,
                                     base::saturated_cast<int32_t>(value));
      return CreateInitializer<int32_t>(int64_shape, data);
    }
    case OperandDataType::kUint32: {
      base::FixedArray<uint32_t> data(operand_size,
                                      base::saturated_cast<uint32_t>(value));
      return CreateInitializer<uint32_t>(int64_shape, data);
    }
    case OperandDataType::kInt64: {
      base::FixedArray<int64_t> data(operand_size,
                                     base::saturated_cast<int64_t>(value));
      return CreateInitializer<int64_t>(int64_shape, data);
    }
    case OperandDataType::kUint64: {
      base::FixedArray<uint64_t> data(operand_size,
                                      base::saturated_cast<uint64_t>(value));
      return CreateInitializer<uint64_t>(int64_shape, data);
    }
    case OperandDataType::kInt8: {
      base::FixedArray<int8_t> data(operand_size,
                                    base::saturated_cast<int8_t>(value));
      return CreateInitializer<int8_t>(int64_shape, data);
    }
    case OperandDataType::kUint8: {
      base::FixedArray<uint8_t> data(operand_size,
                                     base::saturated_cast<uint8_t>(value));
      return CreateInitializer<uint8_t>(int64_shape, data);
    }
    case OperandDataType::kInt4:
    case OperandDataType::kUint4: {
      NOTREACHED();
    }
  }
}

std::string GraphBuilderOrt::CreateScalarInitializer(OperandDataType data_type,
                                                     const MLNumber& value) {
  switch (data_type) {
    case OperandDataType::kFloat32:
      return CreateScalarInitializer(value.AsFloat32());
    case OperandDataType::kFloat16:
      return CreateScalarInitializer(value.AsFloat16());
    case OperandDataType::kInt32:
      return CreateScalarInitializer(value.AsInt32());
    case OperandDataType::kUint32:
      return CreateScalarInitializer(value.AsUint32());
    case OperandDataType::kInt64:
      return CreateScalarInitializer(value.AsInt64());
    case OperandDataType::kUint64:
      return CreateScalarInitializer(value.AsUint64());
    case OperandDataType::kInt8:
      return CreateScalarInitializer(value.AsInt8());
    case OperandDataType::kUint8:
      return CreateScalarInitializer(value.AsUint8());
    case OperandDataType::kInt4:
    case OperandDataType::kUint4: {
      NOTREACHED();
    }
  }
}

std::string GraphBuilderOrt::CreateOneInitializer(
    OperandDataType data_type,
    base::span<const uint32_t> shape) {
  return CreateInitializerForFloat(data_type, shape, 1.0f);
}

std::string GraphBuilderOrt::CreateZeroInitializer(
    OperandDataType data_type,
    base::span<const uint32_t> shape) {
  return CreateInitializerForFloat(data_type, shape, 0.0f);
}

std::string GraphBuilderOrt::TransposeRnnWeightOrBiasLayout(
    base::cstring_view weight_or_bias,
    base::span<const uint32_t> permutation) {
  size_t num_gates = permutation.size();

  // Use Split operator to split the weight/bias into num_gates slices.
  std::vector<std::string> gate_names;
  gate_names.reserve(num_gates);
  for (size_t i = 0; i < num_gates; i++) {
    gate_names.push_back(GenerateOperandName());
  }
  constexpr int64_t axis = 1;
  std::array<ScopedOrtOpAttr, 2> split_attrs = {
      model_editor_.CreateAttribute(kAttrAxis, axis),
      model_editor_.CreateAttribute(kAttrNumOutputs,
                                    static_cast<int64_t>(num_gates))};
  std::array<const char*, 1> split_inputs = {weight_or_bias.c_str()};
  std::vector<const char*> split_outputs;
  split_outputs.reserve(num_gates);
  for (const auto& gate_name : gate_names) {
    split_outputs.push_back(gate_name.c_str());
  }
  std::string split_node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeSplit}, kUnderscore));
  model_editor_.AddNode(kOpTypeSplit, split_node_name, split_inputs,
                        split_outputs, split_attrs);

  // Use Concat operator to concatenate the slices in the order of permutation.
  std::vector<const char*> concat_inputs;
  concat_inputs.reserve(num_gates);
  for (uint32_t index : permutation) {
    concat_inputs.push_back(gate_names[index].c_str());
  }
  std::string concat_output = GenerateOperandName();
  std::array<const char*, 1> concat_outputs = {concat_output.c_str()};
  std::array<ScopedOrtOpAttr, 1> concat_attrs = {
      model_editor_.CreateAttribute(kAttrAxis, axis)};
  std::string concat_node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeConcat}, kUnderscore));
  model_editor_.AddNode(kOpTypeConcat, concat_node_name, concat_inputs,
                        concat_outputs, concat_attrs);

  return concat_output;
}

void GraphBuilderOrt::AddCastNode(base::cstring_view node_name,
                                  base::cstring_view input,
                                  base::cstring_view output,
                                  ONNXTensorElementDataType to_data_type) {
  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};
  int64_t attr_to_data = static_cast<int64_t>(to_data_type);
  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrTo, attr_to_data)};

  model_editor_.AddNode(kOpTypeCast, node_name, inputs, outputs, attributes);
}

std::string GraphBuilderOrt::CreateCastNode(
    base::cstring_view input,
    ONNXTensorElementDataType to_data_type) {
  const std::string output = GenerateOperandName();
  InsertCastNode(input, output, to_data_type);
  return output;
}

void GraphBuilderOrt::InsertCastNode(base::cstring_view input,
                                     base::cstring_view output,
                                     ONNXTensorElementDataType to_data_type) {
  const std::string node_name =
      GenerateNodeName(base::JoinString({kInserted, kOpTypeCast}, kUnderscore));
  AddCastNode(node_name, input, output, to_data_type);
}

void GraphBuilderOrt::AddExpandNode(base::cstring_view node_name,
                                    base::cstring_view input,
                                    base::cstring_view output,
                                    base::span<const uint32_t> shape) {
  // `new_shape` should be the name of an int64 tensor that specifies the
  // output's shape.
  const std::string new_shape = CreateInt64InitializerForUint32Array(shape);

  std::array<const char*, 2> inputs = {input.c_str(), new_shape.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeExpand, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddExpandNode(
    base::cstring_view node_name,
    base::cstring_view input,
    base::cstring_view output,
    base::span<const Dimension> shape,
    base::span<const Dimension> input_shape,
    const base::flat_map<std::string, DynamicDimensionInfo>&
        known_dynamic_dims) {
  // Output shape has dynamic dimensions. Build the shape tensor at runtime
  // using Shape, Gather, and Concat operators.

  // Cache for Shape node outputs to avoid creating duplicate Shape nodes
  // for the same input operand.
  base::flat_map<std::string, std::string> input_to_shape_map;

  // For each dimension in new shape, either use a constant (for static dims)
  // or gather from input shape (for dynamic dims).
  std::vector<std::string> dimension_names;
  dimension_names.reserve(shape.size());

  for (size_t i = 0; i < shape.size(); ++i) {
    const auto& dim = shape[i];
    if (std::holds_alternative<uint32_t>(dim)) {
      // Static dimension: create a 1-D constant with shape [1].
      uint32_t static_value = std::get<uint32_t>(dim);
      std::array<int64_t, 1> value_array = {static_cast<int64_t>(static_value)};
      std::string const_name = Create1DInitializer<int64_t>(value_array);
      dimension_names.push_back(std::move(const_name));
    } else {
      // Dynamic dimension: first check if it comes from the expand input tensor
      // shape, then fall back to known_dynamic_dims.
      CHECK(std::holds_alternative<DynamicDimension>(dim));
      const auto& dynamic_dim = std::get<DynamicDimension>(dim);

      std::string source_operand_name;
      int64_t axis;

      // First check if the dynamic dimension is present in the expand input's
      // shape.
      auto input_shape_it = std::ranges::find(input_shape, dim);
      if (input_shape_it != input_shape.end()) {
        // Dynamic dimension comes from the expand input tensor itself.
        source_operand_name = std::string(input);
        axis = static_cast<int64_t>(
            std::distance(input_shape.begin(), input_shape_it));
      } else {
        // Fall back to known_dynamic_dims. Only a named dynamic dim can be
        // resolved this way; an unnamed dim must have matched the input shape
        // above.
        CHECK(dynamic_dim.name.has_value())
            << "Unnamed dynamic dimension not found in expand input shape";
        auto it = known_dynamic_dims.find(*dynamic_dim.name);
        CHECK(it != known_dynamic_dims.end())
            << "Dynamic dimension '" << *dynamic_dim.name
            << "' not found in input shape or known_dynamic_dims";
        const DynamicDimensionInfo& dyn_dim_info = it->second;
        source_operand_name = dyn_dim_info.input_operand_name;
        axis = static_cast<int64_t>(dyn_dim_info.axis);
      }

      // Create Shape node for this input operand if not already created.
      std::string input_shape_name;
      auto shape_it = input_to_shape_map.find(source_operand_name);
      if (shape_it != input_to_shape_map.end()) {
        input_shape_name = shape_it->second;
      } else {
        input_shape_name = GenerateOperandName();
        std::array<const char*, 1> shape_inputs = {source_operand_name.c_str()};
        std::array<const char*, 1> shape_outputs = {input_shape_name.c_str()};
        const std::string shape_node_name = GenerateNodeName(base::JoinString(
            {kInserted, "Shape", kToEmulate, node_name}, kUnderscore));
        model_editor_.AddNode("Shape", shape_node_name, shape_inputs,
                              shape_outputs);
        input_to_shape_map[source_operand_name] = input_shape_name;
      }

      // Create a Gather node to extract this dimension from the Shape output.
      // Use axis=0 since the Shape output is 1-D, and indices as a 1-D tensor
      // with shape [1] to get output shape [1].
      const std::string gather_output = GenerateOperandName();
      std::array<int64_t, 1> indices_array = {axis};
      const std::string indices_const =
          Create1DInitializer<int64_t>(indices_array);

      std::array<const char*, 2> gather_inputs = {input_shape_name.c_str(),
                                                  indices_const.c_str()};
      std::array<const char*, 1> gather_outputs = {gather_output.c_str()};
      std::array<ScopedOrtOpAttr, 1> gather_attributes = {
          model_editor_.CreateAttribute(kAttrAxis, static_cast<int64_t>(0))};
      const std::string gather_node_name = GenerateNodeName(
          base::JoinString({kInserted, kOpTypeGather, kToEmulate, node_name,
                            base::NumberToString(axis)},
                           kUnderscore));
      model_editor_.AddNode(kOpTypeGather, gather_node_name, gather_inputs,
                            gather_outputs, gather_attributes);

      dimension_names.push_back(std::move(gather_output));
    }
  }

  // Step 3: Concatenate all dimension values to create the final shape tensor.
  std::string final_shape_name;
  if (dimension_names.size() == 1) {
    // Single dimension, no need to concatenate.
    final_shape_name = dimension_names[0];
  } else {
    // Multiple dimensions, concatenate them.
    final_shape_name = GenerateOperandName();
    std::vector<const char*> concat_inputs;
    concat_inputs.reserve(dimension_names.size());
    for (const auto& name : dimension_names) {
      concat_inputs.push_back(name.c_str());
    }
    std::array<const char*, 1> concat_outputs = {final_shape_name.c_str()};
    std::array<ScopedOrtOpAttr, 1> concat_attributes = {
        model_editor_.CreateAttribute(kAttrAxis, static_cast<int64_t>(0))};
    const std::string concat_node_name = GenerateNodeName(base::JoinString(
        {kInserted, kOpTypeConcat, kToEmulate, node_name}, kUnderscore));
    model_editor_.AddNode(kOpTypeConcat, concat_node_name, concat_inputs,
                          concat_outputs, concat_attributes);
  }

  // Step 4: Use the constructed shape tensor for the Expand operation.
  std::array<const char*, 2> inputs = {input.c_str(), final_shape_name.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeExpand, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddResizeNode(base::cstring_view node_name,
                                    base::cstring_view input,
                                    base::cstring_view scales,
                                    base::cstring_view sizes,
                                    base::cstring_view mode,
                                    base::cstring_view output) {
  // Skip the input roi, which only takes effect when the coordinate
  // transformation mode is set to "tf_crop_and_resize". Currently WebNN only
  // supports "half_pixel", which is the default mode.
  const std::string roi;
  std::array<const char*, 4> inputs = {input.c_str(), roi.c_str(),
                                       scales.c_str(), sizes.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrMode, mode)};

  model_editor_.AddNode(kOpTypeResize, node_name, inputs, outputs, attributes);
}

std::string GraphBuilderOrt::BlockwiseExpand(base::cstring_view input,
                                             base::span<const uint32_t> shape) {
  const std::string sizes = CreateInt64InitializerForUint32Array(shape);
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeResize}, kUnderscore));
  const std::string output = GenerateOperandName();
  AddResizeNode(node_name, input, /*scales=*/"", sizes,
                /*mode=*/"nearest", output);

  return output;
}

void GraphBuilderOrt::AddReshapeNode(base::cstring_view node_name,
                                     base::cstring_view input,
                                     base::cstring_view output,
                                     base::span<const uint32_t> shape) {
  // `new_shape` should be the name of an int64 tensor that specifies the
  // output's shape.
  const std::string new_shape = CreateInt64InitializerForUint32Array(shape);

  std::array<const char*, 2> inputs = {input.c_str(), new_shape.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeReshape, node_name, inputs, outputs);
}

std::string GraphBuilderOrt::CreateReshapeNode(
    base::cstring_view input,
    base::span<const uint32_t> shape) {
  const std::string output = GenerateOperandName();
  InsertReshapeNode(input, output, shape);
  return output;
}

void GraphBuilderOrt::InsertReshapeNode(base::cstring_view input,
                                        base::cstring_view output,
                                        base::span<const uint32_t> shape) {
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeReshape}, kUnderscore));
  AddReshapeNode(node_name, input, output, shape);
}

void GraphBuilderOrt::AddUnsqueezeNode(base::cstring_view node_name,
                                       base::cstring_view input,
                                       base::cstring_view output,
                                       base::span<const int64_t> axes) {
  // ONNX Unsqueeze op's `axes` is an operand of data type int64.
  const std::string axes_operand = Create1DInitializer<int64_t>(axes);

  std::array<const char*, 2> inputs = {input.c_str(), axes_operand.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeUnsqueeze, node_name, inputs, outputs);
}

std::string GraphBuilderOrt::CreateUnsqueezeNode(
    base::cstring_view input,
    base::span<const int64_t> axes) {
  const std::string output = GenerateOperandName();
  InsertUnsqueezeNode(input, output, axes);
  return output;
}

void GraphBuilderOrt::InsertUnsqueezeNode(base::cstring_view input,
                                          base::cstring_view output,
                                          base::span<const int64_t> axes) {
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeUnsqueeze}, kUnderscore));
  AddUnsqueezeNode(node_name, input, output, axes);
}

void GraphBuilderOrt::AddSqueezeNode(base::cstring_view node_name,
                                     base::cstring_view input,
                                     base::cstring_view output,
                                     base::span<const int64_t> axes) {
  // ONNX Squeeze op's `axes` is an operand of data type int64.
  const std::string axes_operand = Create1DInitializer<int64_t>(axes);

  std::array<const char*, 2> inputs = {input.c_str(), axes_operand.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeSqueeze, node_name, inputs, outputs);
}

void GraphBuilderOrt::InsertSqueezeNode(base::cstring_view input,
                                        base::cstring_view output,
                                        base::span<const int64_t> axes) {
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeSqueeze}, kUnderscore));
  AddSqueezeNode(node_name, input, output, axes);
}

void GraphBuilderOrt::AddSliceNode(base::cstring_view node_name,
                                   base::cstring_view input,
                                   base::cstring_view output,
                                   base::span<const int64_t> axes_value,
                                   base::span<const int64_t> starts_value,
                                   base::span<const int64_t> ends_value,
                                   base::span<const int64_t> steps_value) {
  // ONNX `Slice` op's `axes`, `starts`， `ends` and `steps` are operands of
  // data type int64 rather than attributes.
  const std::string axes = Create1DInitializer<int64_t>(axes_value);
  const std::string starts = Create1DInitializer<int64_t>(starts_value);
  const std::string ends = Create1DInitializer<int64_t>(ends_value);
  const std::string steps = Create1DInitializer<int64_t>(steps_value);

  std::array<const char*, 5> inputs = {
      input.c_str(), starts.c_str(), ends.c_str(), axes.c_str(), steps.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeSlice, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddTransposeNode(base::cstring_view node_name,
                                       base::cstring_view input,
                                       base::cstring_view output,
                                       base::span<const uint32_t> perm_value) {
  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  base::FixedArray<int64_t> perm(perm_value.begin(), perm_value.end());
  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrPerm, perm)};
  model_editor_.AddNode(kOpTypeTranspose, node_name, inputs, outputs,
                        attributes);
}

std::string GraphBuilderOrt::CreateTransposeNode(
    base::cstring_view input,
    base::span<const uint32_t> perm_value) {
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeTranspose}, kUnderscore));
  const std::string output = GenerateOperandName();

  AddTransposeNode(node_name, input, output, perm_value);
  return output;
}

void GraphBuilderOrt::EmulateWithIdentityNode(base::cstring_view label,
                                              base::cstring_view input,
                                              base::cstring_view output) {
  const std::string node_name = GenerateNodeName(base::JoinString(
      {kInserted, kOpTypeIdentity, kToEmulate, label}, kUnderscore));

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeIdentity, node_name, inputs, outputs);
}

std::string GraphBuilderOrt::ClampIndices(base::cstring_view indices,
                                          OperandDataType data_type,
                                          uint32_t dim_size) {
  const std::string node_name = GenerateNodeName(
      base::JoinString({kInserted, kOpTypeClamp}, kUnderscore));
  const std::string output = GenerateOperandName();

  // The dimension size must be greater than 0.
  CHECK_GT(dim_size, 0u);

  std::string min;
  std::string max;
  switch (data_type) {
    case OperandDataType::kInt32: {
      // A valid dimension must be in the range of int32.
      // https://www.w3.org/TR/webnn/#valid-dimension
      min = CreateScalarInitializer(-base::checked_cast<int32_t>(dim_size));
      max = CreateScalarInitializer(base::checked_cast<int32_t>(dim_size - 1));
      break;
    }
    case OperandDataType::kInt64: {
      min = CreateScalarInitializer(-static_cast<int64_t>(dim_size));
      max = CreateScalarInitializer(static_cast<int64_t>(dim_size - 1));
      break;
    }
    default:
      NOTREACHED() << "[WebNN] Indices can only be one of the int32 and int64 "
                      "data types.";
  }

  std::array<const char*, 3> inputs = {indices.data(), min.c_str(),
                                       max.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeClamp, node_name, inputs, outputs);
  return output;
}

std::string GraphBuilderOrt::ClampIndicesForDynamicShape(
    base::cstring_view node_name,
    base::cstring_view input,
    base::cstring_view indices,
    uint32_t axis,
    OperandDataType indices_data_type) {
  // Step 1: Get the input shape at runtime using Shape operator.
  const std::string input_shape_name = GenerateOperandName();
  {
    std::array<const char*, 1> shape_inputs = {input.c_str()};
    std::array<const char*, 1> shape_outputs = {input_shape_name.c_str()};
    const std::string shape_node_name = GenerateNodeName(base::JoinString(
        {kInserted, "Shape", kToEmulate, node_name}, kUnderscore));
    model_editor_.AddNode("Shape", shape_node_name, shape_inputs,
                          shape_outputs);
  }

  // Step 2: Gather the dimension size at the axis.
  const std::string axis_dim_size_name = GenerateOperandName();
  {
    const std::string axis_index =
        CreateScalarInitializer<int64_t>(static_cast<int64_t>(axis));
    std::array<const char*, 2> gather_inputs = {input_shape_name.c_str(),
                                                axis_index.c_str()};
    std::array<const char*, 1> gather_outputs = {axis_dim_size_name.c_str()};
    const std::string gather_node_name = GenerateNodeName(base::JoinString(
        {kInserted, "Gather", kToEmulate, node_name}, kUnderscore));
    std::array<ScopedOrtOpAttr, 1> gather_attrs = {
        model_editor_.CreateAttribute(kAttrAxis, static_cast<int64_t>(0))};
    model_editor_.AddNode(kOpTypeGather, gather_node_name, gather_inputs,
                          gather_outputs, gather_attrs);
  }

  // Step 3: Cast indices to int64 if needed (for arithmetic operations).
  std::string indices_int64 = std::string(indices);
  bool need_cast_back = false;
  if (indices_data_type != OperandDataType::kInt64) {
    indices_int64 =
        CreateCastNode(indices, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    need_cast_back = true;
  }

  // Step 4: Compute min = -axis_dim_size and max = axis_dim_size - 1
  // min = Neg(axis_dim_size)
  const std::string neg_dim_size_name = GenerateOperandName();
  {
    std::array<const char*, 1> neg_inputs = {axis_dim_size_name.c_str()};
    std::array<const char*, 1> neg_outputs = {neg_dim_size_name.c_str()};
    const std::string neg_node_name = GenerateNodeName(base::JoinString(
        {kInserted, kOpTypeNeg, kToEmulate, node_name}, kUnderscore));
    model_editor_.AddNode(kOpTypeNeg, neg_node_name, neg_inputs, neg_outputs);
  }

  // max = axis_dim_size - 1
  const std::string one_scalar = CreateScalarInitializer<int64_t>(1);
  const std::string max_value_name = GenerateOperandName();
  {
    std::array<const char*, 2> sub_inputs = {axis_dim_size_name.c_str(),
                                             one_scalar.c_str()};
    std::array<const char*, 1> sub_outputs = {max_value_name.c_str()};
    const std::string sub_node_name = GenerateNodeName(base::JoinString(
        {kInserted, kOpTypeSub, kToEmulate, node_name}, kUnderscore));
    model_editor_.AddNode(kOpTypeSub, sub_node_name, sub_inputs, sub_outputs);
  }

  // Step 5: Clamp using Clip operator
  const std::string clamped_indices_int64 = GenerateOperandName();
  {
    std::array<const char*, 3> clip_inputs = {indices_int64.c_str(),
                                              neg_dim_size_name.c_str(),
                                              max_value_name.c_str()};
    std::array<const char*, 1> clip_outputs = {clamped_indices_int64.c_str()};
    const std::string clip_node_name = GenerateNodeName(base::JoinString(
        {kInserted, kOpTypeClamp, kToEmulate, node_name}, kUnderscore));
    model_editor_.AddNode(kOpTypeClamp, clip_node_name, clip_inputs,
                          clip_outputs);
  }

  // Step 6: Cast back to original data type if needed.
  if (need_cast_back) {
    return CreateCastNode(clamped_indices_int64,
                          WebnnToOnnxDataType(indices_data_type));
  }
  return clamped_indices_int64;
}

std::string GraphBuilderOrt::ClampGatherNDIndices(
    base::cstring_view input_operand_name,
    base::cstring_view indices_operand_name,
    const OperandDescriptor& input_descriptor,
    const OperandDescriptor& indices_descriptor) {
  CHECK_GT(input_descriptor.shape().size(), 0u);
  CHECK_GT(indices_descriptor.shape().size(), 0u);

  // The indices last dimension must be static (validated earlier).
  const Dimension& indices_last_dim =
      indices_descriptor.shape()[indices_descriptor.shape().size() - 1];
  CHECK(std::holds_alternative<uint32_t>(indices_last_dim));
  const uint32_t num_axes = std::get<uint32_t>(indices_last_dim);

  // Check if all relevant input dimensions are static.
  bool relevant_input_dims_static = true;
  for (uint32_t axis = 0; axis < num_axes; ++axis) {
    if (!std::holds_alternative<uint32_t>(input_descriptor.shape()[axis])) {
      relevant_input_dims_static = false;
      break;
    }
  }

  if (relevant_input_dims_static) {
    // Static path: create constant min/max tensors.
    std::array<int64_t, 1> min_max_shape = {static_cast<int64_t>(num_axes)};

    base::FixedArray<int64_t> min_value(num_axes);
    base::FixedArray<int64_t> max_value(num_axes);
    for (uint32_t axis = 0; axis < num_axes; ++axis) {
      uint32_t input_dim_size =
          std::get<uint32_t>(input_descriptor.shape()[axis]);
      min_value[axis] = -static_cast<int64_t>(input_dim_size);
      max_value[axis] = static_cast<int64_t>(input_dim_size) - 1;
    }

    // ONNX Clip can only have `min` and `max` as scalars, so here use Min and
    // Max to emulate a clamp operation.
    std::string min = CreateInitializer<int64_t>(min_max_shape, min_value);
    const std::string max_node_name = GenerateNodeName(
        base::JoinString({kInserted, kOpTypeMax}, kUnderscore));
    const std::string max_output = GenerateOperandName();
    std::array<const char*, 2> max_inputs = {indices_operand_name.c_str(),
                                             min.c_str()};
    std::array<const char*, 1> max_outputs = {max_output.c_str()};
    model_editor_.AddNode(kOpTypeMax, max_node_name, max_inputs, max_outputs);

    std::string max = CreateInitializer<int64_t>(min_max_shape, max_value);
    const std::string min_node_name = GenerateNodeName(
        base::JoinString({kInserted, kOpTypeMin}, kUnderscore));
    const std::string output = GenerateOperandName();
    std::array<const char*, 2> min_inputs = {max_output.c_str(), max.c_str()};
    std::array<const char*, 1> min_outputs = {output.c_str()};
    model_editor_.AddNode(kOpTypeMin, min_node_name, min_inputs, min_outputs);

    return output;
  } else {
    // Dynamic path: build min/max tensors at runtime.
    const std::string node_base_name = GenerateNodeName(
        base::JoinString({kInserted, "ClampGatherND"}, kUnderscore));

    // Step 1: Get the input shape at runtime.
    const std::string input_shape_name = GenerateOperandName();
    {
      std::array<const char*, 1> shape_inputs = {input_operand_name.c_str()};
      std::array<const char*, 1> shape_outputs = {input_shape_name.c_str()};
      const std::string shape_node_name = GenerateNodeName(base::JoinString(
          {kInserted, "Shape", kToEmulate, node_base_name}, kUnderscore));
      model_editor_.AddNode("Shape", shape_node_name, shape_inputs,
                            shape_outputs);
    }

    // Step 2: Slice input_shape to get the first num_axes dimensions.
    const std::string relevant_input_dims_name = GenerateOperandName();
    {
      const std::string starts = Create1DInitializer<int64_t>({0});
      const std::string ends =
          Create1DInitializer<int64_t>({static_cast<int64_t>(num_axes)});
      const std::string axes = Create1DInitializer<int64_t>({0});
      const std::string steps = Create1DInitializer<int64_t>({1});

      std::array<const char*, 5> slice_inputs = {input_shape_name.c_str(),
                                                 starts.c_str(), ends.c_str(),
                                                 axes.c_str(), steps.c_str()};
      std::array<const char*, 1> slice_outputs = {
          relevant_input_dims_name.c_str()};
      const std::string slice_node_name = GenerateNodeName(base::JoinString(
          {kInserted, kOpTypeSlice, kToEmulate, node_base_name}, kUnderscore));
      model_editor_.AddNode(kOpTypeSlice, slice_node_name, slice_inputs,
                            slice_outputs);
    }

    // Step 3: Compute min = -relevant_input_dims using Neg.
    const std::string min_values_name = GenerateOperandName();
    {
      std::array<const char*, 1> neg_inputs = {
          relevant_input_dims_name.c_str()};
      std::array<const char*, 1> neg_outputs = {min_values_name.c_str()};
      const std::string neg_node_name = GenerateNodeName(base::JoinString(
          {kInserted, kOpTypeNeg, kToEmulate, node_base_name}, kUnderscore));
      model_editor_.AddNode(kOpTypeNeg, neg_node_name, neg_inputs, neg_outputs);
    }

    // Step 4: Compute max = relevant_input_dims - 1 using Sub.
    const std::string max_values_name = GenerateOperandName();
    {
      const std::string one_scalar = CreateScalarInitializer<int64_t>(1);
      std::array<const char*, 2> sub_inputs = {relevant_input_dims_name.c_str(),
                                               one_scalar.c_str()};
      std::array<const char*, 1> sub_outputs = {max_values_name.c_str()};
      const std::string sub_node_name = GenerateNodeName(base::JoinString(
          {kInserted, kOpTypeSub, kToEmulate, node_base_name}, kUnderscore));
      model_editor_.AddNode(kOpTypeSub, sub_node_name, sub_inputs, sub_outputs);
    }

    // Step 5: Clamp indices using Max and Min operations.
    const std::string after_max_name = GenerateOperandName();
    {
      std::array<const char*, 2> max_inputs = {indices_operand_name.c_str(),
                                               min_values_name.c_str()};
      std::array<const char*, 1> max_outputs = {after_max_name.c_str()};
      const std::string max_node_name = GenerateNodeName(base::JoinString(
          {kInserted, kOpTypeMax, kToEmulate, node_base_name}, kUnderscore));
      model_editor_.AddNode(kOpTypeMax, max_node_name, max_inputs, max_outputs);
    }

    const std::string clamped_indices = GenerateOperandName();
    {
      std::array<const char*, 2> min_inputs = {after_max_name.c_str(),
                                               max_values_name.c_str()};
      std::array<const char*, 1> min_outputs = {clamped_indices.c_str()};
      const std::string min_node_name = GenerateNodeName(base::JoinString(
          {kInserted, kOpTypeMin, kToEmulate, node_base_name}, kUnderscore));
      model_editor_.AddNode(kOpTypeMin, min_node_name, min_inputs, min_outputs);
    }

    return clamped_indices;
  }
}

template <typename T>
void GraphBuilderOrt::AddBinaryOperation(const T& operation,
                                         base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(operation.label);
  const std::string lhs = GetOperandNameById(operation.lhs_operand_id);
  const std::string rhs = GetOperandNameById(operation.rhs_operand_id);
  const std::string output = GetOperandNameById(operation.output_operand_id);

  std::array<const char*, 2> inputs = {lhs.c_str(), rhs.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(op_type, node_name, inputs, outputs);
}

template <typename T>
void GraphBuilderOrt::AddUnaryOperation(const T& operation,
                                        base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(operation.label);
  const std::string input = GetOperandNameById(operation.input_operand_id);
  const std::string output = GetOperandNameById(operation.output_operand_id);

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(op_type, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddArgMinMaxOperation(
    const mojom::ArgMinMax& arg_min_max) {
  const std::string node_name = GenerateNodeName(arg_min_max.label);
  const std::string input = GetOperandNameById(arg_min_max.input_operand_id);
  const std::string output = GetOperandNameById(arg_min_max.output_operand_id);

  CHECK(context_properties_.data_type_limits.arg_min_max_input.Supports(
      GetOperand(arg_min_max.input_operand_id).descriptor));
  CHECK(context_properties_.data_type_limits.arg_min_max_output.Supports(
      GetOperand(arg_min_max.output_operand_id).descriptor));

  std::array<ScopedOrtOpAttr, 2> attributes = {
      model_editor_.CreateAttribute(kAttrAxis,
                                    static_cast<int64_t>(arg_min_max.axis)),
      model_editor_.CreateAttribute(
          kAttrKeepDims, static_cast<int64_t>(arg_min_max.keep_dimensions))};

  // ONNX ArgMin/Max only supports int64 output.
  OperandDataType output_data_type =
      GetOperand(arg_min_max.output_operand_id).descriptor.data_type();
  bool need_cast = output_data_type != OperandDataType::kInt64;
  const std::string int64_output = need_cast ? GenerateOperandName() : output;

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {int64_output.c_str()};

  model_editor_.AddNode(arg_min_max.kind == mojom::ArgMinMax::Kind::kMax
                            ? kOpTypeArgMax
                            : kOpTypeArgMin,
                        node_name, inputs, outputs, attributes);

  if (need_cast) {
    // Here cast ArgMin/Max output from int64 to int32 is safe since WebNN
    // operand dimension must be in the range of int32.
    // https://www.w3.org/TR/webnn/#valid-dimension
    CHECK_EQ(output_data_type, OperandDataType::kInt32);
    InsertCastNode(int64_output, output, WebnnToOnnxDataType(output_data_type));
  }
}

void GraphBuilderOrt::AddBatchNormalizationOperation(
    const mojom::BatchNormalization& batch_normalization) {
  const std::string node_name = GenerateNodeName(batch_normalization.label);
  std::string input = GetOperandNameById(batch_normalization.input_operand_id);
  const std::string mean =
      GetOperandNameById(batch_normalization.mean_operand_id);
  const std::string variance =
      GetOperandNameById(batch_normalization.variance_operand_id);
  std::string output =
      GetOperandNameById(batch_normalization.output_operand_id);

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  CHECK(data_type_limits.batch_normalization_input.Supports(
      GetOperand(batch_normalization.input_operand_id).descriptor));
  CHECK(data_type_limits.batch_normalization_mean.Supports(
      GetOperand(batch_normalization.mean_operand_id).descriptor));
  // TODO(crbug.com/431952809): Rename DataTypeLimits fields to be more generic
  // or encompassing.
  CHECK(data_type_limits.batch_normalization_mean.Supports(
      GetOperand(batch_normalization.variance_operand_id).descriptor));

  const OperandDescriptor& input_descriptor =
      GetOperand(batch_normalization.input_operand_id).descriptor;
  const OperandDataType input_data_type = input_descriptor.data_type();
  const std::vector<Dimension>& input_shape = input_descriptor.shape();
  // ONNX BatchNormalization expects NCHW layout, channel is at index 1. In
  // addition it also accepts single dimension input of size N in which case C
  // is assumed to be 1.
  // https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#inputs
  //
  // WebNN BatchNormalization supports 1D input of shape [C], but ONNX requires
  // at least 2D input. To handle this, we reshape [C] to [1, C] before passing
  // to ONNX, then reshape the output back to [C].
  bool needs_reshape_for_1d = input_shape.size() == 1;
  Dimension input_channels = uint32_t{1};

  if (needs_reshape_for_1d) {
    // Unsqueeze 1D [C] -> 2D [1, C] for ONNX BatchNorm by adding dimension at
    // axis 0.
    input_channels = input_shape[0];
    std::array<int64_t, 1> unsqueeze_axes = {0};
    input = CreateUnsqueezeNode(input, unsqueeze_axes);
  } else if (input_shape.size() > 1) {
    // For multi-dimensional inputs, channel is at index 1 (NCHW layout).
    input_channels = input_shape[1];
  }

  // ONNX BatchNormalization requires 5 inputs: input, scale, bias, mean and
  // variance. WebNN allows optional scale/bias, so create default ones if not
  // provided. Default scale = 1.0 (no scaling), default bias = 0.0 (no offset).
  std::string scale, bias;
  if (batch_normalization.scale_operand_id) {
    CHECK(data_type_limits.batch_normalization_mean.Supports(
        GetOperand(batch_normalization.scale_operand_id.value()).descriptor));
    scale = GetOperandNameById(batch_normalization.scale_operand_id.value());
  } else {
    if (std::holds_alternative<uint32_t>(input_channels)) {
      std::vector<uint32_t> scale_and_bias_shape = {
          std::get<uint32_t>(input_channels)};
      scale = CreateOneInitializer(input_data_type, scale_and_bias_shape);
    } else {
      // Dynamic channel dimension: create scale at runtime with channel axis.
      std::array<uint32_t, 1> channel_axis = {
          static_cast<uint32_t>(needs_reshape_for_1d ? 0 : 1)};
      scale = CreateInitializerWithInputShapeAndDataTypeForFloat(
          batch_normalization.input_operand_id, 1.0f, channel_axis);
    }
  }
  if (batch_normalization.bias_operand_id) {
    CHECK(data_type_limits.batch_normalization_mean.Supports(
        GetOperand(batch_normalization.bias_operand_id.value()).descriptor));
    bias = GetOperandNameById(batch_normalization.bias_operand_id.value());
  } else {
    if (std::holds_alternative<uint32_t>(input_channels)) {
      std::vector<uint32_t> scale_and_bias_shape = {
          std::get<uint32_t>(input_channels)};
      bias = CreateZeroInitializer(input_data_type, scale_and_bias_shape);
    } else {
      // Dynamic channel dimension: create bias at runtime with channel axis.
      std::array<uint32_t, 1> channel_axis = {
          static_cast<uint32_t>(needs_reshape_for_1d ? 0 : 1)};
      bias = CreateInitializerWithInputShapeAndDataTypeForFloat(
          batch_normalization.input_operand_id, 0.0f, channel_axis);
    }
  }

  // If we reshaped input from 1D to 2D, we need to reshape output back to 1D.
  std::string batchnorm_output = output;
  if (needs_reshape_for_1d) {
    batchnorm_output = GenerateOperandName();
  }

  std::array<const char*, 5> inputs = {input.c_str(), scale.c_str(),
                                       bias.c_str(), mean.c_str(),
                                       variance.c_str()};
  std::array<const char*, 1> outputs = {batchnorm_output.c_str()};
  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrEpsilon, batch_normalization.epsilon)};
  model_editor_.AddNode(kOpTypeBatchNormalization, node_name, inputs, outputs,
                        attributes);

  // Squeeze output back from 2D [1, C] -> 1D [C] for 1D inputs by removing
  // dimension at axis 0.
  if (needs_reshape_for_1d) {
    std::array<int64_t, 1> squeeze_axes = {0};
    InsertSqueezeNode(batchnorm_output, output, squeeze_axes);
  }
}

void GraphBuilderOrt::AddCastOperation(const mojom::ElementWiseUnary& cast) {
  const std::string node_name = GenerateNodeName(cast.label);
  const std::string input = GetOperandNameById(cast.input_operand_id);
  const std::string output = GetOperandNameById(cast.output_operand_id);
  const OperandDataType output_data_type =
      GetOperand(cast.output_operand_id).descriptor.data_type();
  AddCastNode(node_name, input, output, WebnnToOnnxDataType(output_data_type));
}

void GraphBuilderOrt::AddConv2dOperation(const mojom::Conv2d& conv2d) {
  const std::string node_name = GenerateNodeName(conv2d.label);
  const std::string input = GetOperandNameById(conv2d.input_operand_id);
  const std::string filter = GetOperandNameById(conv2d.filter_operand_id);
  const std::string output = GetOperandNameById(conv2d.output_operand_id);

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  CHECK(data_type_limits.conv2d_input.Supports(
      GetOperand(conv2d.input_operand_id).descriptor));
  CHECK(data_type_limits.conv2d_input.Supports(
      GetOperand(conv2d.filter_operand_id).descriptor));
  std::vector<const char*> inputs = {input.c_str(), filter.c_str()};
  std::string bias;
  if (conv2d.bias_operand_id.has_value()) {
    CHECK(data_type_limits.conv2d_bias.Supports(
        GetOperand(conv2d.bias_operand_id.value()).descriptor));
    bias = GetOperandNameById(conv2d.bias_operand_id.value());
    inputs.push_back(bias.c_str());
  }
  std::array<const char*, 1> outputs = {output.c_str()};

  std::vector<ScopedOrtOpAttr> attributes;
  attributes.reserve(5);
  std::array<int64_t, 2> dilations = {
      base::checked_cast<int64_t>(conv2d.dilations->height),
      base::checked_cast<int64_t>(conv2d.dilations->width)};
  attributes.push_back(
      model_editor_.CreateAttribute(kAttrDilations, dilations));

  int64_t group = base::checked_cast<int64_t>(conv2d.groups);
  attributes.push_back(model_editor_.CreateAttribute(kAttrGroup, group));

  std::array<int64_t, 4> pads = {
      base::checked_cast<int64_t>(conv2d.padding->beginning->height),
      base::checked_cast<int64_t>(conv2d.padding->beginning->width),
      base::checked_cast<int64_t>(conv2d.padding->ending->height),
      base::checked_cast<int64_t>(conv2d.padding->ending->width)};
  attributes.push_back(model_editor_.CreateAttribute(kAttrPads, pads));

  std::array<int64_t, 2> strides = {
      base::checked_cast<int64_t>(conv2d.strides->height),
      base::checked_cast<int64_t>(conv2d.strides->width)};
  attributes.push_back(model_editor_.CreateAttribute(kAttrStrides, strides));

  switch (conv2d.kind) {
    case mojom::Conv2d::Kind::kDirect:
      model_editor_.AddNode(kOpTypeConv2d, node_name, inputs, outputs,
                            attributes);
      break;
    case mojom::Conv2d::Kind::kTransposed:
      // According to the ONNX ConvTranspose2d documentation, `output_padding`
      // is a zero vector if not specified and `pads` will be auto generated if
      // `output_shape` is specified. So we need to calculate the
      // `output_padding` and explicitly set it to ensure that the attributes
      // information is not missing. Since the `pads` attribute has already been
      // set, there is no need to set `output_size` attribute.
      // https://onnx.ai/onnx/operators/onnx__ConvTranspose.html#attributes
      const std::vector<webnn::Dimension> input_shape =
          GetOperand(conv2d.input_operand_id).descriptor.shape();
      const std::vector<webnn::Dimension> filter_shape =
          GetOperand(conv2d.filter_operand_id).descriptor.shape();
      const std::vector<webnn::Dimension> output_shape =
          GetOperand(conv2d.output_operand_id).descriptor.shape();
      // Since ONNX Runtime uses nchw input layout and oihw filter layout，
      // input/filter/output_shape[2] and input/filter/output_shape[3] are used
      // here to access the height and width dimensions of the
      // input/filter/output_shape tensor shape.
      // Filter dims are always static. For input/output spatial dims, use the
      // static value if known, otherwise 0 (a dynamic dim is handled at
      // runtime).
      auto get_size = [](const webnn::Dimension& dim) -> int64_t {
        const uint32_t* val = std::get_if<uint32_t>(&dim);
        return val ? base::checked_cast<int64_t>(*val) : 0;
      };
      std::array<int64_t, 2> input_size = {
          get_size(input_shape[2]), get_size(input_shape[3])};
      std::array<int64_t, 2> filter_size = {
          get_size(filter_shape[2]), get_size(filter_shape[3])};
      std::array<int64_t, 2> output_size = {
          get_size(output_shape[2]), get_size(output_shape[3])};

      int64_t output_padding_height = CalculateOutputPaddingSize(
          input_size[0], filter_size[0], strides[0], dilations[0], pads[0],
          pads[2], output_size[0]);
      int64_t output_padding_width = CalculateOutputPaddingSize(
          input_size[1], filter_size[1], strides[1], dilations[1], pads[1],
          pads[3], output_size[1]);
      std::array<int64_t, 2> output_padding = {output_padding_height,
                                               output_padding_width};

      attributes.push_back(
          model_editor_.CreateAttribute(kAttrOutputPadding, output_padding));

      model_editor_.AddNode(kOpTypeConvTranspose2d, node_name, inputs, outputs,
                            attributes);
      break;
  }
}

void GraphBuilderOrt::AddCumulativeSumOperation(
    const mojom::CumulativeSum& cumulative_sum) {
  const std::string node_name = GenerateNodeName(cumulative_sum.label);
  const std::string input = GetOperandNameById(cumulative_sum.input_operand_id);
  const std::string output =
      GetOperandNameById(cumulative_sum.output_operand_id);

  CHECK(context_properties_.data_type_limits.cumulative_sum_input.Supports(
      GetOperand(cumulative_sum.input_operand_id).descriptor));

  const std::string axis =
      CreateScalarInitializer(base::checked_cast<int64_t>(cumulative_sum.axis));

  std::array<ScopedOrtOpAttr, 2> attributes = {
      model_editor_.CreateAttribute(
          kAttrExclusive,
          base::checked_cast<int64_t>(cumulative_sum.exclusive)),
      model_editor_.CreateAttribute(
          kAttrReverse, base::checked_cast<int64_t>(cumulative_sum.reversed))};

  std::array<const char*, 2> inputs = {input.c_str(), axis.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};
  model_editor_.AddNode(kOpTypeCumulativeSum, node_name, inputs, outputs,
                        attributes);
}

template <typename T>
  requires(std::is_same_v<T, mojom::DequantizeLinear> ||
           std::is_same_v<T, mojom::QuantizeLinear>)
void GraphBuilderOrt::AddDequantizeOrQuantizeLinearOperation(
    const T& operation,
    base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(operation.label);
  std::string input = GetOperandNameById(operation.input_operand_id);
  std::string scale = GetOperandNameById(operation.scale_operand_id);
  std::string zero_point = GetOperandNameById(operation.zero_point_operand_id);
  std::string output = GetOperandNameById(operation.output_operand_id);

  const std::vector<uint32_t> input_shape =
      GetOperand(operation.input_operand_id).descriptor.StaticShape().value();
  // ZeroPoint has the same shape as the scale.
  const std::vector<uint32_t> scale_zero_point_shape =
      GetOperand(operation.scale_operand_id).descriptor.StaticShape().value();
  CHECK_EQ(scale_zero_point_shape.size(), input_shape.size());

  std::optional<int64_t> axis;
  uint32_t scale_not_size_one_dimension_count = 0;
  for (size_t i = 0; i < scale_zero_point_shape.size(); i++) {
    if (scale_zero_point_shape[i] != 1) {
      scale_not_size_one_dimension_count++;
      if (scale_zero_point_shape[i] == input_shape[i]) {
        axis = i;
      }
    }
  }

  bool is_per_axis =
      axis.has_value() && scale_not_size_one_dimension_count == 1;

  std::optional<int64_t> block_size;
  if (scale_not_size_one_dimension_count == 0) {
    // For per-tensor(per-layer) quantization and dequantization, scale should
    // be a scalar.
    if (!scale_zero_point_shape.empty()) {
      // The numbers in scale shape are all 1, scale and zeroPoint should be
      // reshaped to a scalar.
      scale = CreateReshapeNode(scale, {});
      zero_point = CreateReshapeNode(zero_point, {});
    }
  } else if (is_per_axis) {
    // For per-axis quantization and dequantization, scale and zeroPoint should
    // be a 1-D Tensor.
    if (scale_zero_point_shape.size() != 1) {
      scale = CreateReshapeNode(scale, {input_shape[axis.value()]});
      zero_point = CreateReshapeNode(zero_point, {input_shape[axis.value()]});
    }
  } else {
    // For blockwise quantization and dequantization, scale should has the same
    // shape as the input or except for one dimension in which blocking is
    // performed.
    // The default values are used if scale has the same shape as the input.
    axis = 0;
    block_size = 1;
    uint32_t blockwise_axis_count = 0;
    for (size_t i = 0; i < scale_zero_point_shape.size(); i++) {
      if (scale_zero_point_shape[i] != input_shape[i]) {
        CHECK_EQ(input_shape[i] % scale_zero_point_shape[i], 0u);
        block_size = input_shape[i] / scale_zero_point_shape[i];
        axis = i;
        blockwise_axis_count++;
      }
    }

    if (blockwise_axis_count > 1) {
      // The data type of zero point can be int4/uint4, which is not
      // supported by `resize` operator. So cast it to int8/uint8 before
      // `resize` and cast back to int4/uint4 after `resize`.
      const OperandDataType zero_point_data_type =
          GetOperand(operation.zero_point_operand_id).descriptor.data_type();
      if (zero_point_data_type == OperandDataType::kInt4) {
        zero_point =
            CreateCastNode(zero_point, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
      } else if (zero_point_data_type == OperandDataType::kUint4) {
        zero_point =
            CreateCastNode(zero_point, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      }

      scale = BlockwiseExpand(scale, input_shape);
      zero_point = BlockwiseExpand(zero_point, input_shape);

      if (zero_point_data_type == OperandDataType::kInt4) {
        zero_point =
            CreateCastNode(zero_point, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4);
      } else if (zero_point_data_type == OperandDataType::kUint4) {
        zero_point =
            CreateCastNode(zero_point, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4);
      }

      // Reset the axis and block_size back to default values, because scale and
      // zeroPoint now have the same shape as input.
      axis = 0;
      block_size = 1;
    }
  }

  std::array<const char*, 3> inputs = {input.c_str(), scale.c_str(),
                                       zero_point.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  std::vector<ScopedOrtOpAttr> attributes;
  if (axis.has_value()) {
    attributes.push_back(
        model_editor_.CreateAttribute(kAttrAxis, axis.value()));
  }

  if (block_size.has_value()) {
    attributes.push_back(
        model_editor_.CreateAttribute(kAttrBlockSize, block_size.value()));
  }

  model_editor_.AddNode(op_type, node_name, inputs, outputs, attributes);
}

void GraphBuilderOrt::AddEluOperation(const mojom::Elu& elu) {
  const std::string node_name = GenerateNodeName(elu.label);
  const std::string input = GetOperandNameById(elu.input_operand_id);
  const std::string output = GetOperandNameById(elu.output_operand_id);

  CHECK(context_properties_.data_type_limits.elu_input.Supports(
      GetOperand(elu.input_operand_id).descriptor));

  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrAlpha, elu.alpha)};

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};
  model_editor_.AddNode(kOpTypeElu, node_name, inputs, outputs, attributes);
}

// TODO(crbug.com/426228071): Eliminate redundant cast ops for bool and uint8
// data types conversion.
void GraphBuilderOrt::AddLogicalBinaryOperation(
    const mojom::ElementWiseBinary& logical_binary,
    base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(logical_binary.label);
  std::string lhs = GetOperandNameById(logical_binary.lhs_operand_id);
  std::string rhs = GetOperandNameById(logical_binary.rhs_operand_id);

  // Some ONNX logical binary operations only support bool input.
  if (logical_binary.kind == mojom::ElementWiseBinary::Kind::kLogicalAnd ||
      logical_binary.kind == mojom::ElementWiseBinary::Kind::kLogicalOr ||
      logical_binary.kind == mojom::ElementWiseBinary::Kind::kLogicalXor) {
    CHECK_EQ(GetOperand(logical_binary.lhs_operand_id).descriptor.data_type(),
             OperandDataType::kUint8);
    lhs = CreateCastNode(lhs, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);

    CHECK_EQ(GetOperand(logical_binary.rhs_operand_id).descriptor.data_type(),
             OperandDataType::kUint8);
    rhs = CreateCastNode(rhs, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
  }
  std::array<const char*, 2> inputs = {lhs.c_str(), rhs.c_str()};

  const std::string bool_output = GenerateOperandName();
  std::array<const char*, 1> outputs = {bool_output.c_str()};
  model_editor_.AddNode(op_type, node_name, inputs, outputs);

  // ONNX logical operators only support bool output. WebNN logical operators
  // support uint8 output. It is necessary to insert a cast operator after a
  // logical operator.
  const OperandDataType output_data_type =
      GetOperand(logical_binary.output_operand_id).descriptor.data_type();
  const std::string output =
      GetOperandNameById(logical_binary.output_operand_id);
  CHECK_EQ(output_data_type, OperandDataType::kUint8);
  InsertCastNode(bool_output, output, WebnnToOnnxDataType(output_data_type));
}

void GraphBuilderOrt::AddLogicalUnaryOperation(
    const mojom::ElementWiseUnary& logical_unary,
    base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(logical_unary.label);

  std::string input = GetOperandNameById(logical_unary.input_operand_id);

  // LogicalNot operation in ONNX only supports bool input.
  if (op_type == kOpTypeLogicalNot) {
    CHECK_EQ(GetOperand(logical_unary.input_operand_id).descriptor.data_type(),
             OperandDataType::kUint8);
    input = CreateCastNode(input, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
  }

  const std::string bool_output = GenerateOperandName();

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {bool_output.c_str()};
  model_editor_.AddNode(op_type, node_name, inputs, outputs);

  // ONNX logical operators only support bool output, while WebNN logical
  // operators support uint8 output. Insert a `Cast` operator for type
  // conversion.
  const OperandDataType output_data_type =
      GetOperand(logical_unary.output_operand_id).descriptor.data_type();
  const std::string output =
      GetOperandNameById(logical_unary.output_operand_id);
  CHECK_EQ(output_data_type, OperandDataType::kUint8);
  InsertCastNode(bool_output, output, WebnnToOnnxDataType(output_data_type));
}

void GraphBuilderOrt::AddLogicalNotEqualOperation(
    const mojom::ElementWiseBinary& not_equal) {
  // Step 1: calculate `equal(a, b)`.
  const std::string equal_node_name =
      GenerateNodeName(GenerateEmulatedOpLabel(kOpTypeEqual, not_equal.label));
  std::string lhs = GetOperandNameById(not_equal.lhs_operand_id);
  std::string rhs = GetOperandNameById(not_equal.rhs_operand_id);
  const std::string equal_output = GenerateOperandName();

  std::array<const char*, 1> equal_outputs = {equal_output.c_str()};
  std::array<const char*, 2> equal_inputs = {lhs.c_str(), rhs.c_str()};
  model_editor_.AddNode(kOpTypeEqual, equal_node_name, equal_inputs,
                        equal_outputs);

  // Step 2: calculate `logicalNot(equal_output)`
  const std::string not_output = GenerateOperandName();
  std::array<const char*, 1> not_outputs = {not_output.c_str()};
  const std::string not_node_name = GenerateNodeName(
      GenerateEmulatedOpLabel(kOpTypeLogicalNot, not_equal.label));
  model_editor_.AddNode(kOpTypeLogicalNot, not_node_name, equal_outputs,
                        not_outputs);

  // ONNX logical operators only support bool output. To support output with the
  // WebNN data type, it is necessary to insert a cast operator after a logical
  // operator.
  OperandId output_operand_id = not_equal.output_operand_id;
  const OperandDataType output_data_type =
      GetOperand(output_operand_id).descriptor.data_type();
  std::string output = GetOperandNameById(output_operand_id);
  CHECK_EQ(output_data_type, OperandDataType::kUint8);
  InsertCastNode(not_output, output, WebnnToOnnxDataType(output_data_type));
}

void GraphBuilderOrt::AddElementWiseBinaryOperation(
    const mojom::ElementWiseBinary& element_wise_binary) {
  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  const OperandDescriptor& lhs_descriptor =
      GetOperand(element_wise_binary.lhs_operand_id).descriptor;
  const OperandDescriptor& rhs_descriptor =
      GetOperand(element_wise_binary.rhs_operand_id).descriptor;
  switch (element_wise_binary.kind) {
    case mojom::ElementWiseBinary::Kind::kAdd: {
      CHECK(data_type_limits.add_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeAdd);
    }
    case mojom::ElementWiseBinary::Kind::kSub: {
      CHECK(data_type_limits.sub_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeSub);
    }
    case mojom::ElementWiseBinary::Kind::kMul: {
      CHECK(data_type_limits.mul_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeMul);
    }
    case mojom::ElementWiseBinary::Kind::kDiv: {
      CHECK(data_type_limits.div_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeDiv);
    }
    case mojom::ElementWiseBinary::Kind::kMax: {
      CHECK(data_type_limits.max_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeMax);
    }
    case mojom::ElementWiseBinary::Kind::kMin: {
      CHECK(data_type_limits.min_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeMin);
    }
    case mojom::ElementWiseBinary::Kind::kPow: {
      CHECK(data_type_limits.pow_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypePow);
    }
    case mojom::ElementWiseBinary::Kind::kEqual: {
      CHECK(data_type_limits.equal_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeEqual);
    }
    case mojom::ElementWiseBinary::Kind::kNotEqual: {
      CHECK(data_type_limits.not_equal_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalNotEqualOperation(element_wise_binary);
    }
    case mojom::ElementWiseBinary::Kind::kGreater: {
      CHECK(data_type_limits.greater_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeGreater);
    }
    case mojom::ElementWiseBinary::Kind::kGreaterOrEqual: {
      CHECK(data_type_limits.greater_or_equal_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary,
                                       kOpTypeGreaterOrEqual);
    }
    case mojom::ElementWiseBinary::Kind::kLesser: {
      CHECK(data_type_limits.lesser_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeLesser);
    }
    case mojom::ElementWiseBinary::Kind::kLesserOrEqual: {
      CHECK(data_type_limits.lesser_or_equal_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary,
                                       kOpTypeLesserOrEqual);
    }
    case mojom::ElementWiseBinary::Kind::kLogicalAnd: {
      CHECK(data_type_limits.logical_and_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeLogicalAnd);
    }
    case mojom::ElementWiseBinary::Kind::kLogicalOr: {
      CHECK(data_type_limits.logical_or_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeLogicalOr);
    }
    case mojom::ElementWiseBinary::Kind::kLogicalXor: {
      CHECK(data_type_limits.logical_xor_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddLogicalBinaryOperation(element_wise_binary, kOpTypeLogicalXor);
    }
    case mojom::ElementWiseBinary::Kind::kMod: {
      CHECK(data_type_limits.mod_input.SupportsAll(
          {lhs_descriptor, rhs_descriptor}));
      return AddBinaryOperation(element_wise_binary, kOpTypeMod);
    }
  }
}

void GraphBuilderOrt::AddElementWiseUnaryOperation(
    const mojom::ElementWiseUnary& element_wise_unary) {
  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  const OperandDescriptor& input_descriptor =
      GetOperand(element_wise_unary.input_operand_id).descriptor;
  switch (element_wise_unary.kind) {
    case mojom::ElementWiseUnary::Kind::kAbs: {
      CHECK(data_type_limits.abs_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeAbs);
    }
    case mojom::ElementWiseUnary::Kind::kCeil: {
      CHECK(data_type_limits.ceil_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeCeil);
    }
    case mojom::ElementWiseUnary::Kind::kCos: {
      CHECK(data_type_limits.cos_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeCos);
    }
    case mojom::ElementWiseUnary::Kind::kExp: {
      CHECK(data_type_limits.exp_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeExp);
    }
    case mojom::ElementWiseUnary::Kind::kFloor: {
      CHECK(data_type_limits.floor_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeFloor);
    }
    case mojom::ElementWiseUnary::Kind::kLog: {
      CHECK(data_type_limits.log_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeLog);
    }
    case mojom::ElementWiseUnary::Kind::kIsNaN: {
      CHECK(data_type_limits.is_nan_input.Supports(input_descriptor));
      return AddLogicalUnaryOperation(element_wise_unary, kOpTypeIsNaN);
    }
    case mojom::ElementWiseUnary::Kind::kIsInfinite: {
      CHECK(data_type_limits.is_infinite_input.Supports(input_descriptor));
      return AddLogicalUnaryOperation(element_wise_unary, kOpTypeIsInfinite);
    }
    case mojom::ElementWiseUnary::Kind::kLogicalNot: {
      CHECK(data_type_limits.logical_not_input.Supports(input_descriptor));
      return AddLogicalUnaryOperation(element_wise_unary, kOpTypeLogicalNot);
    }
    case mojom::ElementWiseUnary::Kind::kNeg: {
      CHECK(data_type_limits.neg_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeNeg);
    }
    case mojom::ElementWiseUnary::Kind::kRoundEven: {
      CHECK(data_type_limits.round_even_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeRoundEven);
    }
    case mojom::ElementWiseUnary::Kind::kSign: {
      CHECK(data_type_limits.sign_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeSign);
    }
    case mojom::ElementWiseUnary::Kind::kSin: {
      CHECK(data_type_limits.sin_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeSin);
    }
    case mojom::ElementWiseUnary::Kind::kTan: {
      CHECK(data_type_limits.tan_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeTan);
    }
    case mojom::ElementWiseUnary::Kind::kIdentity: {
      CHECK(data_type_limits.identity_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeIdentity);
    }
    case mojom::ElementWiseUnary::Kind::kSqrt: {
      CHECK(data_type_limits.sqrt_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeSqrt);
    }
    case mojom::ElementWiseUnary::Kind::kErf: {
      CHECK(data_type_limits.erf_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeErf);
    }
    case mojom::ElementWiseUnary::Kind::kReciprocal: {
      CHECK(data_type_limits.reciprocal_input.Supports(input_descriptor));
      return AddUnaryOperation(element_wise_unary, kOpTypeReciprocal);
    }
    case mojom::ElementWiseUnary::Kind::kCast: {
      CHECK(data_type_limits.cast_input.Supports(input_descriptor));
      return AddCastOperation(element_wise_unary);
    }
  }
}

void GraphBuilderOrt::AddClampOperation(const mojom::Clamp& clamp) {
  const std::string node_name = GenerateNodeName(clamp.label);
  const std::string input = GetOperandNameById(clamp.input_operand_id);
  const std::string output = GetOperandNameById(clamp.output_operand_id);

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  const OperandDescriptor& input_descriptor =
      GetOperand(clamp.input_operand_id).descriptor;
  CHECK(data_type_limits.clamp_input.Supports(input_descriptor));

  const OperandDataType input_data_type = input_descriptor.data_type();

  // Min and max are 0-D operands with the same data type of input.
  const std::string min =
      CreateScalarInitializer(input_data_type, clamp.min_value);
  const std::string max =
      CreateScalarInitializer(input_data_type, clamp.max_value);

  std::array<const char*, 3> inputs = {input.c_str(), min.c_str(), max.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeClamp, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddExpandOperation(const mojom::Expand& expand) {
  const std::string input = GetOperandNameById(expand.input_operand_id);
  const std::string output = GetOperandNameById(expand.output_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(expand.input_operand_id).descriptor;
  CHECK(context_properties_.data_type_limits.expand_input.Supports(
      input_descriptor));

  const OperandDescriptor& output_descriptor =
      GetOperand(expand.output_operand_id).descriptor;

  // Workaround: expanding a scalar to another scalar is supposed to be a no-op,
  // here we map it to an Identity node to avoid the mishandling of some ORT
  // EPs.
  // TODO(crbug.com/500385615): Remove the workaround when the issue is fixed.
  if (input_descriptor.Rank() == 0 && output_descriptor.Rank() == 0) {
    EmulateWithIdentityNode(expand.label, input, output);
    return;
  }

  const std::string node_name = GenerateNodeName(expand.label);
  const std::vector<Dimension>& output_shape =
      GetOperand(expand.output_operand_id).descriptor.shape();
  const std::vector<Dimension>& expand_input_shape =
      GetOperand(expand.input_operand_id).descriptor.shape();

  AddExpandNode(node_name, input, output, output_shape, expand_input_shape,
                known_dynamic_dims_);
}

void GraphBuilderOrt::AddConcatOperation(const mojom::Concat& concat) {
  const std::string node_name = GenerateNodeName(concat.label);

  size_t input_count = concat.input_operand_ids.size();
  base::FixedArray<std::string> inputs_string(input_count);
  base::FixedArray<const char*> inputs(input_count);
  for (size_t i = 0; i < input_count; i++) {
    CHECK(context_properties_.data_type_limits.concat_inputs.Supports(
        GetOperand(concat.input_operand_ids[i]).descriptor));
    inputs_string[i] = GetOperandNameById(concat.input_operand_ids[i]);
    inputs[i] = inputs_string[i].c_str();
  }

  const std::string output = GetOperandNameById(concat.output_operand_id);
  std::array<const char*, 1> outputs = {output.c_str()};

  std::array<ScopedOrtOpAttr, 1> attributes = {model_editor_.CreateAttribute(
      kAttrAxis, base::checked_cast<int64_t>(concat.axis))};

  model_editor_.AddNode(kOpTypeConcat, node_name, inputs, outputs, attributes);
}

template <typename T>
void GraphBuilderOrt::AddGatherOperation(const T& operation,
                                         base::cstring_view op_type) {
  const std::string node_name = GenerateNodeName(operation.label);
  const std::string input = GetOperandNameById(operation.input_operand_id);
  const std::string indices = GetOperandNameById(operation.indices_operand_id);
  const std::string output = GetOperandNameById(operation.output_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(operation.input_operand_id).descriptor;
  const Dimension& axis_dimension = input_descriptor.shape()[operation.axis];

  std::string indices_to_use = indices;
  // Clamp the indices operand to prevent out-of-bounds reading which will cause
  // ORT CPU EP to throw a runtime error.
  if (std::holds_alternative<uint32_t>(axis_dimension)) {
    indices_to_use = ClampIndices(
        indices,
        GetOperand(operation.indices_operand_id).descriptor.data_type(),
        std::get<uint32_t>(axis_dimension));
  } else {
    // For dynamic dimension, clamp indices at runtime.
    indices_to_use = ClampIndicesForDynamicShape(
        node_name, input, indices, operation.axis,
        GetOperand(operation.indices_operand_id).descriptor.data_type());
  }

  std::array<const char*, 2> inputs = {input.c_str(), indices_to_use.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  std::array<ScopedOrtOpAttr, 1> attributes = {model_editor_.CreateAttribute(
      kAttrAxis, static_cast<int64_t>(operation.axis))};

  model_editor_.AddNode(op_type, node_name, inputs, outputs, attributes);
}

void GraphBuilderOrt::AddGatherNDOperation(const mojom::GatherND& gather_nd) {
  const std::string node_name = GenerateNodeName(gather_nd.label);
  const std::string input = GetOperandNameById(gather_nd.input_operand_id);
  const std::string indices = GetOperandNameById(gather_nd.indices_operand_id);
  const std::string output = GetOperandNameById(gather_nd.output_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(gather_nd.input_operand_id).descriptor;
  const OperandDescriptor& indices_descriptor =
      GetOperand(gather_nd.indices_operand_id).descriptor;
  CHECK(context_properties_.data_type_limits.gather_nd_input.Supports(
      input_descriptor));
  CHECK(context_properties_.data_type_limits.gather_nd_indices.Supports(
      indices_descriptor));

  // ONNX GatherND only supports int64 indices.
  std::string int64_indices =
      indices_descriptor.data_type() == OperandDataType::kInt64
          ? indices
          : CreateCastNode(indices, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

  // Clamp the indices operand to prevent out-of-bounds reading which will cause
  // ORT CPU EP to throw a runtime error.
  std::string clamped_indices = ClampGatherNDIndices(
      input, int64_indices, input_descriptor, indices_descriptor);

  std::array<const char*, 2> inputs = {input.c_str(), clamped_indices.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeGatherND, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddGemmOperation(const mojom::Gemm& gemm) {
  const std::string node_name = GenerateNodeName(gemm.label);
  const std::string input_a = GetOperandNameById(gemm.a_operand_id);
  const std::string input_b = GetOperandNameById(gemm.b_operand_id);
  const std::string output = GetOperandNameById(gemm.output_operand_id);

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  const OperandDescriptor& input_a_descriptor =
      GetOperand(gemm.a_operand_id).descriptor;
  const OperandDescriptor& input_b_descriptor =
      GetOperand(gemm.b_operand_id).descriptor;
  CHECK(data_type_limits.gemm_a.SupportsAll(
      {input_a_descriptor, input_b_descriptor}));
  CHECK_EQ(input_a_descriptor.data_type(), input_b_descriptor.data_type());

  std::vector<const char*> inputs = {input_a.c_str(), input_b.c_str()};
  std::string input_c;
  if (gemm.c_operand_id.has_value()) {
    const OperandDescriptor& input_c_descriptor =
        GetOperand(*gemm.c_operand_id).descriptor;
    CHECK(data_type_limits.gemm_c.Supports(input_c_descriptor));
    CHECK_EQ(input_c_descriptor.data_type(), input_a_descriptor.data_type());

    input_c = GetOperandNameById(*gemm.c_operand_id);
    inputs.push_back(input_c.c_str());
  }
  std::array<const char*, 1> outputs = {output.c_str()};

  std::array<ScopedOrtOpAttr, 4> attributes = {
      model_editor_.CreateAttribute(kAttrAlpha, gemm.alpha),
      model_editor_.CreateAttribute(kAttrBeta, gemm.beta),
      model_editor_.CreateAttribute(kAttrTransA,
                                    static_cast<int64_t>(gemm.a_transpose)),
      model_editor_.CreateAttribute(kAttrTransB,
                                    static_cast<int64_t>(gemm.b_transpose))};

  model_editor_.AddNode(kOpTypeGemm, node_name, inputs, outputs, attributes);
}

// `GruType` must be `mojom::Gru` or `mojom::GruCell`.
template <typename GruType>
  requires(std::is_same_v<GruType, mojom::Gru> ||
           std::is_same_v<GruType, mojom::GruCell>)
void GraphBuilderOrt::AddGruOperation(const GruType& gru) {
  const std::string node_name = GenerateNodeName(gru.label);
  std::string input = GetOperandNameById(gru.input_operand_id);
  std::string weight = GetOperandNameById(gru.weight_operand_id);
  std::string recurrent_weight =
      GetOperandNameById(gru.recurrent_weight_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(gru.input_operand_id).descriptor;
  const OperandDescriptor& weight_descriptor =
      GetOperand(gru.weight_operand_id).descriptor;
  const OperandDescriptor& recurrent_weight_descriptor =
      GetOperand(gru.recurrent_weight_operand_id).descriptor;

  uint32_t num_directions = 1;
  if constexpr (std::is_same_v<GruType, mojom::Gru>) {
    CHECK(context_properties_.data_type_limits.gru_input.Supports(
        input_descriptor));
    CHECK(context_properties_.data_type_limits.gru_input.Supports(
        weight_descriptor));
    CHECK(context_properties_.data_type_limits.gru_input.Supports(
        recurrent_weight_descriptor));
    num_directions =
        gru.direction == mojom::RecurrentNetworkDirection::kBoth ? 2 : 1;
  } else {
    CHECK(context_properties_.data_type_limits.gru_cell_input.Supports(
        input_descriptor));
    CHECK(context_properties_.data_type_limits.gru_cell_input.Supports(
        weight_descriptor));
    CHECK(context_properties_.data_type_limits.gru_cell_input.Supports(
        recurrent_weight_descriptor));

    // Reshape the input into a 3-D tensor, since the GRU of ONNX requires
    // the input shape to be [seq_length, batch_size, input_size]. For
    // gruCell, `seq_length` is equal to 1.
    const std::vector<uint32_t> input_shape =
        input_descriptor.StaticShape().value();
    CHECK_EQ(input_shape.size(), 2u);
    input = CreateReshapeNode(input, {1, input_shape[0], input_shape[1]});

    // Reshape the weight into a 3-D tensor, since the GRU of ONNX requires
    // the weight shape to be [num_directions, 3*hidden_size, input_size].
    // For gruCell, `num_directions` is equal to 1.
    const std::vector<uint32_t> weight_shape =
        weight_descriptor.StaticShape().value();
    CHECK_EQ(weight_shape.size(), 2u);
    weight = CreateReshapeNode(weight, {1, weight_shape[0], weight_shape[1]});

    // Reshape the recurrent weight into a 3-D tensor, since the GRU of ONNX
    // requires the recurrent weight shape to be [num_directions,
    // 3*hidden_size, hidden_size]. For gruCell, `num_directions` is equal to 1.
    const std::vector<uint32_t> recurrent_weight_shape =
        recurrent_weight_descriptor.StaticShape().value();
    CHECK_EQ(recurrent_weight_shape.size(), 2u);
    recurrent_weight = CreateReshapeNode(
        recurrent_weight,
        {1, recurrent_weight_shape[0], recurrent_weight_shape[1]});
  }

  constexpr std::array<uint32_t, 3> kRznToZrnPermutation = {1, 0, 2};
  if (gru.layout == mojom::GruWeightLayout::kRzn) {
    weight = TransposeRnnWeightOrBiasLayout(weight, kRznToZrnPermutation);
    recurrent_weight =
        TransposeRnnWeightOrBiasLayout(recurrent_weight, kRznToZrnPermutation);
  }

  std::vector<const char*> inputs = {input.c_str(), weight.c_str(),
                                     recurrent_weight.c_str()};

  const uint32_t hidden_size = gru.hidden_size;
  // Graph validation already checked that hidden_size * 3 would not overflow.
  std::array<uint32_t, 2> bias_dims = {num_directions, hidden_size * 3};
  std::string bias, recurrent_bias, concatenated_bias;
  if (!gru.bias_operand_id.has_value() &&
      !gru.recurrent_bias_operand_id.has_value()) {
    // When both bias and recurrentBias are not present, set ONNX GRU input "B"
    // as not specified.
    inputs.push_back("");
  } else {
    if (gru.bias_operand_id.has_value()) {
      bias = GetOperandNameById(*gru.bias_operand_id);
      if constexpr (std::is_same_v<GruType, mojom::Gru>) {
        CHECK(context_properties_.data_type_limits.gru_bias.Supports(
            GetOperand(*gru.bias_operand_id).descriptor));
      } else {
        CHECK(context_properties_.data_type_limits.gru_cell_bias.Supports(
            GetOperand(*gru.bias_operand_id).descriptor));
        bias = CreateReshapeNode(bias, bias_dims);
      }
      if (gru.layout == mojom::GruWeightLayout::kRzn) {
        bias = TransposeRnnWeightOrBiasLayout(bias, kRznToZrnPermutation);
      }
    } else {
      bias = CreateZeroInitializer(input_descriptor.data_type(), bias_dims);
    }

    if (gru.recurrent_bias_operand_id.has_value()) {
      recurrent_bias = GetOperandNameById(*gru.recurrent_bias_operand_id);
      if constexpr (std::is_same_v<GruType, mojom::Gru>) {
        CHECK(context_properties_.data_type_limits.gru_bias.Supports(
            GetOperand(*gru.recurrent_bias_operand_id).descriptor));
      } else {
        CHECK(context_properties_.data_type_limits.gru_cell_bias.Supports(
            GetOperand(*gru.recurrent_bias_operand_id).descriptor));
        recurrent_bias = CreateReshapeNode(recurrent_bias, bias_dims);
      }
      if (gru.layout == mojom::GruWeightLayout::kRzn) {
        recurrent_bias = TransposeRnnWeightOrBiasLayout(recurrent_bias,
                                                        kRznToZrnPermutation);
      }
    } else {
      recurrent_bias =
          CreateZeroInitializer(input_descriptor.data_type(), bias_dims);
    }

    // Concat bias and recurrent_bias.
    concatenated_bias = GenerateOperandName();
    std::array<const char*, 2> bias_inputs = {bias.c_str(),
                                              recurrent_bias.c_str()};
    std::array<const char*, 1> bias_outputs = {concatenated_bias.c_str()};
    std::array<ScopedOrtOpAttr, 1> concat_attributes = {
        model_editor_.CreateAttribute(kAttrAxis, static_cast<int64_t>(1))};
    std::string concat_node_name = GenerateNodeName(
        base::JoinString({kInserted, kOpTypeConcat}, kUnderscore));
    model_editor_.AddNode(kOpTypeConcat, concat_node_name, bias_inputs,
                          bias_outputs, concat_attributes);
    inputs.push_back(concatenated_bias.c_str());
  }

  // "sequence_lens" is an optional tensor specifying lengths of the sequences
  // in a batch.
  inputs.push_back("");

  std::string hidden_state;
  if constexpr (std::is_same_v<GruType, mojom::Gru>) {
    if (gru.initial_hidden_state_operand_id.has_value()) {
      hidden_state =
          GetOperandNameById(gru.initial_hidden_state_operand_id.value());
      CHECK(context_properties_.data_type_limits.gru_input.Supports(
          GetOperand(gru.initial_hidden_state_operand_id.value()).descriptor));
    }
  } else {
    hidden_state = GetOperandNameById(gru.hidden_state_operand_id);
    const std::vector<uint32_t> hidden_state_shape =
        GetOperand(gru.hidden_state_operand_id)
            .descriptor.StaticShape()
            .value();
    CHECK_EQ(hidden_state_shape.size(), 2u);
    // Reshape the hiddenState into a 3-D tensor, since the GRU of ONNX requires
    // the "initial_h" shape to be [num_directions, batch_size, hidden_size].
    // For gruCell, `num_directions` is equal to 1.
    hidden_state = CreateReshapeNode(
        hidden_state, {1, hidden_state_shape[0], hidden_state_shape[1]});
  }
  inputs.push_back(hidden_state.c_str());

  std::vector<ScopedOrtOpAttr> attributes;
  attributes.reserve(4);
  base::cstring_view direction = "forward";
  if constexpr (std::is_same_v<GruType, mojom::Gru>) {
    direction = GetRecurrentNetworkDirection(gru.direction);
  }
  attributes.push_back(
      model_editor_.CreateAttribute(kAttrDirection, direction));

  const std::vector<base::cstring_view> activations =
      GetRecurrentNetworkActivations(gru.activations,
                                     direction == "bidirectional");
  std::vector<const char*> activations_c_str;
  for (const auto& activation : activations) {
    activations_c_str.push_back(activation.c_str());
  }
  attributes.push_back(
      model_editor_.CreateAttribute(kAttrActivations, activations_c_str));

  attributes.push_back(model_editor_.CreateAttribute(
      kAttrHiddenSize, base::checked_cast<int64_t>(hidden_size)));
  attributes.push_back(model_editor_.CreateAttribute(
      kAttrLinearBeforeReset, static_cast<int64_t>(gru.reset_after)));

  std::string output, output_hidden;
  if constexpr (std::is_same_v<GruType, mojom::Gru>) {
    output_hidden = GetOperandNameById(gru.output_operand_ids[0]);
    if (gru.return_sequence) {
      output = GetOperandNameById(gru.output_operand_ids[1]);
    }
  } else {
    output_hidden = GenerateOperandName();
  }
  std::array<const char*, 2> outputs = {output.c_str(), output_hidden.c_str()};
  model_editor_.AddNode(kOpTypeGru, node_name, inputs, outputs, attributes);

  if constexpr (std::is_same_v<GruType, mojom::GruCell>) {
    // Reshape the ONNX GRU output "Y_h" of shape [num_directions, batch_size,
    // hidden_size] back to a 2-D tensor, since the gruCell of WebNN requires
    // the output shape to be [batchSize, hiddenSize].
    const std::vector<uint32_t> output_shape =
        GetOperand(gru.output_operand_id).descriptor.StaticShape().value();
    CHECK_EQ(output_shape.size(), 2u);
    InsertReshapeNode(output_hidden, GetOperandNameById(gru.output_operand_id),
                      output_shape);
  }
}

template void GraphBuilderOrt::AddGruOperation<mojom::Gru>(const mojom::Gru&);

template void GraphBuilderOrt::AddGruOperation<mojom::GruCell>(
    const mojom::GruCell&);

void GraphBuilderOrt::AddHardSigmoidOperation(
    const mojom::HardSigmoid& hard_sigmoid) {
  const std::string node_name = GenerateNodeName(hard_sigmoid.label);
  const std::string input = GetOperandNameById(hard_sigmoid.input_operand_id);
  const std::string output = GetOperandNameById(hard_sigmoid.output_operand_id);

  CHECK(context_properties_.data_type_limits.hard_sigmoid_input.Supports(
      GetOperand(hard_sigmoid.input_operand_id).descriptor));

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  std::array<ScopedOrtOpAttr, 2> attributes = {
      model_editor_.CreateAttribute(kAttrAlpha, hard_sigmoid.alpha),
      model_editor_.CreateAttribute(kAttrBeta, hard_sigmoid.beta)};
  model_editor_.AddNode(kOpTypeHardSigmoid, node_name, inputs, outputs,
                        attributes);
}

void GraphBuilderOrt::AddInstanceNormalizationOperation(
    const mojom::InstanceNormalization& instance_normalization) {
  const std::string node_name = GenerateNodeName(instance_normalization.label);
  const std::string input =
      GetOperandNameById(instance_normalization.input_operand_id);
  const std::string output =
      GetOperandNameById(instance_normalization.output_operand_id);

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  CHECK(data_type_limits.instance_normalization_input.Supports(
      GetOperand(instance_normalization.input_operand_id).descriptor));
  const OperandDescriptor& input_descriptor =
      GetOperand(instance_normalization.input_operand_id).descriptor;
  const OperandDataType input_data_type = input_descriptor.data_type();
  // ONNX InstanceNormalization expects NCHW layout, channel is at index 1.
  const std::vector<Dimension>& input_shape = input_descriptor.shape();
  CHECK_EQ(input_shape.size(), 4u);
  Dimension input_channels = input_shape[1];

  // ONNX InstanceNormalization requires 3 inputs: input, scale and bias.
  // WebNN allows optional scale/bias, so create default ones if not provided.
  // Default scale = 1.0 (no scaling), default bias = 0.0 (no offset).
  std::string scale, bias;
  if (instance_normalization.scale_operand_id) {
    CHECK(data_type_limits.instance_normalization_scale.Supports(
        GetOperand(instance_normalization.scale_operand_id.value())
            .descriptor));
    scale = GetOperandNameById(instance_normalization.scale_operand_id.value());
  } else {
    if (std::holds_alternative<uint32_t>(input_channels)) {
      std::vector<uint32_t> scale_and_bias_shape = {
          std::get<uint32_t>(input_channels)};
      scale = CreateOneInitializer(input_data_type, scale_and_bias_shape);
    } else {
      // Dynamic channel dimension: create scale at runtime with channel axis.
      std::array<uint32_t, 1> channel_axis = {1};
      scale = CreateInitializerWithInputShapeAndDataTypeForFloat(
          instance_normalization.input_operand_id, 1.0f, channel_axis);
    }
  }
  if (instance_normalization.bias_operand_id) {
    CHECK(data_type_limits.instance_normalization_scale.Supports(
        GetOperand(instance_normalization.bias_operand_id.value()).descriptor));
    bias = GetOperandNameById(instance_normalization.bias_operand_id.value());
  } else {
    if (std::holds_alternative<uint32_t>(input_channels)) {
      std::vector<uint32_t> scale_and_bias_shape = {
          std::get<uint32_t>(input_channels)};
      bias = CreateZeroInitializer(input_data_type, scale_and_bias_shape);
    } else {
      // Dynamic channel dimension: create bias at runtime with channel axis.
      std::array<uint32_t, 1> channel_axis = {1};
      bias = CreateInitializerWithInputShapeAndDataTypeForFloat(
          instance_normalization.input_operand_id, 0.0f, channel_axis);
    }
  }

  std::array<const char*, 3> inputs = {input.c_str(), scale.c_str(),
                                       bias.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};
  std::array<ScopedOrtOpAttr, 1> attributes = {model_editor_.CreateAttribute(
      kAttrEpsilon, instance_normalization.epsilon)};
  model_editor_.AddNode(kOpTypeInstanceNormalization, node_name, inputs,
                        outputs, attributes);
}

void GraphBuilderOrt::AddLayerNormalizationOperation(
    const mojom::LayerNormalization& layer_normalization) {
  const std::string node_name = GenerateNodeName(layer_normalization.label);
  const std::string input =
      GetOperandNameById(layer_normalization.input_operand_id);
  const std::string output =
      GetOperandNameById(layer_normalization.output_operand_id);

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  const OperandDescriptor& input_descriptor =
      GetOperand(layer_normalization.input_operand_id).descriptor;
  CHECK(data_type_limits.layer_normalization_input.Supports(input_descriptor));

  std::string scale, bias;
  if (layer_normalization.scale_operand_id) {
    CHECK(data_type_limits.layer_normalization_input.Supports(
        GetOperand(layer_normalization.scale_operand_id.value()).descriptor));
    scale = GetOperandNameById(layer_normalization.scale_operand_id.value());
  }
  if (layer_normalization.bias_operand_id) {
    CHECK(data_type_limits.layer_normalization_input.Supports(
        GetOperand(layer_normalization.bias_operand_id.value()).descriptor));
    bias = GetOperandNameById(layer_normalization.bias_operand_id.value());
  }

  std::vector<const char*> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};
  const OperandDataType input_data_type = input_descriptor.data_type();
  auto axes = layer_normalization.axes;
  const auto static_input_shape = input_descriptor.StaticShape();
  // ONNX LayerNormalization doesn't support empty axes because it requires to
  // set the first normalization dimension.
  // https://onnx.ai/onnx/operators/onnx__LayerNormalization.html#attributes
  // For WebNN layerNormalization, if axes is empty, no dimensions are reduced
  // and the emulation can be simplified to `output = bias + (scale * 0).
  // https://www.w3.org/TR/webnn/#dom-mllayernormalizationoptions-axes
  if (axes.empty()) {
    if (layer_normalization.bias_operand_id) {
      std::string zero;
      if (static_input_shape.has_value()) {
        zero =
            CreateZeroInitializer(input_data_type, static_input_shape.value());
      } else {
        zero = CreateInitializerWithInputShapeAndDataTypeForFloat(
            layer_normalization.input_operand_id, 0.0f);
      }
      std::array<const char*, 2> add_inputs = {bias.c_str(), zero.c_str()};
      return model_editor_.AddNode(kOpTypeAdd, node_name, add_inputs, outputs);
    } else {
      std::array<const char*, 2> sub_inputs = {input.c_str(), input.c_str()};
      return model_editor_.AddNode(kOpTypeSub, node_name, sub_inputs, outputs);
    }
  }

  const size_t axes_size = axes.size();
  // Sort the indexes of the elements in the axes array based on their values
  // and return the sorted index array for adding a transpose operation if
  // needed. For example input shape is [2, 1, 4, 3], the shape of the scale and
  // bias is [3, 1, 4] if axes is [3, 1, 2], the sorted axes would be [1, 2, 3],
  // then the permutation would be (sorted indices array) [1, 2, 0].
  std::optional<std::vector<uint32_t>> permutation;
  if (!std::ranges::is_sorted(axes)) {
    std::vector<uint32_t> sorted_indices(axes_size);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::ranges::sort(sorted_indices, std::ranges::less(),
                      [&axes](uint32_t index) { return axes[index]; });
    permutation = std::move(sorted_indices);
    std::ranges::sort(axes);
  }

  std::optional<std::vector<uint32_t>> static_scale_shape;
  if (static_input_shape.has_value()) {
    static_scale_shape.emplace();
    static_scale_shape->reserve(axes_size);
    std::ranges::transform(axes, std::back_inserter(static_scale_shape.value()),
                           [&static_input_shape](uint32_t axis) {
                             return static_input_shape.value()[axis];
                           });
  }
  // Because ONNX LayerNormalization only accepts the first normalization
  // dimension, it can only support WebNN layerNormalization whose axes are
  // consecutive til the last dimension. Here we only check beginning and ending
  // of the ascending sorted axes, because the blink validation code ensures
  // axes not having duplicated values.
  if (axes[axes_size - 1] == input_descriptor.shape().size() - 1 &&
      axes[0] == input_descriptor.shape().size() - axes_size) {
    if (layer_normalization.scale_operand_id) {
      if (permutation.has_value()) {
        scale = CreateTransposeNode(scale, permutation.value());
      }
    } else {
      if (static_scale_shape.has_value()) {
        scale =
            CreateOneInitializer(input_data_type, static_scale_shape.value());
      } else {
        // Dynamic scale shape: create scale at runtime with axes.
        scale = CreateInitializerWithInputShapeAndDataTypeForFloat(
            layer_normalization.input_operand_id, 1.0f, axes);
      }
    }
    inputs.push_back(scale.c_str());

    if (layer_normalization.bias_operand_id) {
      if (permutation.has_value()) {
        bias = CreateTransposeNode(bias, permutation.value());
      }
      inputs.push_back(bias.c_str());
    }

    std::array<ScopedOrtOpAttr, 2> attributes = {
        model_editor_.CreateAttribute(kAttrAxis,
                                      base::checked_cast<int64_t>(axes[0])),
        model_editor_.CreateAttribute(kAttrEpsilon,
                                      layer_normalization.epsilon)};

    model_editor_.AddNode(kOpTypeLayerNormalization, node_name, inputs, outputs,
                          attributes);
  } else {
    // Emulate layerNormalization by scale * ((input - mean) / sqrt(variance +
    // epsilon)) + bias. Calculate mean as follows:
    // reduceOptions = {axes, keepDimensions: true};
    // mean = builder.reduceMean(input, reduceOptions).
    const std::string reduce_mean_1_label = GenerateEmulatedOpLabel(
        kOpTypeReduceMean, layer_normalization.label, "1");
    const std::string reduce_mean_1_node_name =
        GenerateNodeName(reduce_mean_1_label);
    const std::string mean_output = GenerateOperandName();
    std::string axes_name = CreateInt64InitializerForUint32Array(axes);
    std::array<const char*, 2> reduce_mean_1_inputs = {input.c_str(),
                                                       axes_name.c_str()};
    std::array<const char*, 1> reduce_mean_1_outputs = {mean_output.c_str()};
    std::array<ScopedOrtOpAttr, 2> reduce_mean_1_attributes = {
        model_editor_.CreateAttribute(kAttrKeepDims, 1),
        model_editor_.CreateAttribute(kAttrNoopWithEmptyAxes, 1)};
    model_editor_.AddNode(kOpTypeReduceMean, reduce_mean_1_node_name,
                          reduce_mean_1_inputs, reduce_mean_1_outputs,
                          reduce_mean_1_attributes);

    // Calculate variance as follows:
    // powValue = builder.constant(input.dataType, 2);
    // variance = builder.reduceMean(builder.pow(builder.sub(input, mean),
    // powValue), reduceOptions);
    const std::string sub_label =
        GenerateEmulatedOpLabel(kOpTypeSub, layer_normalization.label);
    const std::string sub_node_name = GenerateNodeName(sub_label);
    const std::string sub_output = GenerateOperandName();

    std::array<const char*, 2> sub_inputs = {input.c_str(),
                                             mean_output.c_str()};
    std::array<const char*, 1> sub_outputs = {sub_output.c_str()};
    model_editor_.AddNode(kOpTypeSub, sub_node_name, sub_inputs, sub_outputs);

    const std::string pow_label =
        GenerateEmulatedOpLabel(kOpTypePow, layer_normalization.label);
    std::string pow_node_name = GenerateNodeName(pow_label);
    const std::string pow_output = GenerateOperandName();
    std::string pow_value =
        CreateScalarInitializer(input_data_type, MLNumber::FromFloat64(2.0f));
    std::array<const char*, 2> pow_inputs = {sub_output.c_str(),
                                             pow_value.c_str()};
    std::array<const char*, 1> pow_outputs = {pow_output.c_str()};
    model_editor_.AddNode(kOpTypePow, pow_node_name, pow_inputs, pow_outputs);

    const std::string reduce_mean_2_label = GenerateEmulatedOpLabel(
        kOpTypeReduceMean, layer_normalization.label, "2");
    const std::string reduce_mean_2_node_name =
        GenerateNodeName(reduce_mean_2_label);
    const std::string variance_output = GenerateOperandName();
    std::array<const char*, 2> reduce_mean_2_inputs = {pow_output.c_str(),
                                                       axes_name.c_str()};
    std::array<const char*, 1> reduce_mean_2_outputs = {
        variance_output.c_str()};
    std::array<ScopedOrtOpAttr, 2> reduce_mean_2_attributes = {
        model_editor_.CreateAttribute(kAttrKeepDims, 1),
        model_editor_.CreateAttribute(kAttrNoopWithEmptyAxes, 1)};
    model_editor_.AddNode(kOpTypeReduceMean, reduce_mean_2_node_name,
                          reduce_mean_2_inputs, reduce_mean_2_outputs,
                          reduce_mean_2_attributes);

    const std::string add_label =
        GenerateEmulatedOpLabel(kOpTypeAdd, layer_normalization.label);
    const std::string add_node_name = GenerateNodeName(add_label);
    const std::string add_output = GenerateOperandName();
    std::string epsilon_value = CreateScalarInitializer(
        input_data_type, MLNumber::FromFloat64(layer_normalization.epsilon));
    std::array<const char*, 2> add_inputs = {variance_output.c_str(),
                                             epsilon_value.c_str()};
    std::array<const char*, 1> add_outputs = {add_output.c_str()};
    model_editor_.AddNode(kOpTypeAdd, add_node_name, add_inputs, add_outputs);

    const std::string sqrt_label =
        GenerateEmulatedOpLabel(kOpTypeSqrt, layer_normalization.label);
    const std::string sqrt_node_name = GenerateNodeName(sqrt_label);
    const std::string sqrt_output = GenerateOperandName();
    std::array<const char*, 1> sqrt_inputs = {add_output.c_str()};
    std::array<const char*, 1> sqrt_outputs = {sqrt_output.c_str()};
    model_editor_.AddNode(kOpTypeSqrt, sqrt_node_name, sqrt_inputs,
                          sqrt_outputs);

    const std::string div_label =
        GenerateEmulatedOpLabel(kOpTypeDiv, layer_normalization.label);
    const std::string div_node_name = GenerateNodeName(div_label);
    const std::string div_output = GenerateOperandName();
    std::array<const char*, 2> div_inputs = {sub_output.c_str(),
                                             sqrt_output.c_str()};
    std::array<const char*, 1> div_outputs = {div_output.c_str()};
    model_editor_.AddNode(kOpTypeDiv, div_node_name, div_inputs, div_outputs);

    // Create compatible_shape for broadcasting scale and bias with intermediate
    // results such as `div_output` and `mul_output`. Initialize all dimensions
    // to 1, then set normalization axes to match input dimensions for
    // element-wise operations.
    // Example: input_shape=[2,3,4,5], axes=[1,3] -> compatible_shape=[1,3,1,5].
    // Use variant to hold either static shape (vector) or dynamic shape
    // (string).
    std::variant<std::vector<uint32_t>, std::string> compatible_shape;

    if (static_input_shape.has_value()) {
      // Static input shape: create compatible_shape vector.
      std::vector<uint32_t> static_compatible_shape(static_input_shape->size(),
                                                    1);
      for (auto axis : axes) {
        static_compatible_shape[axis] = static_input_shape.value()[axis];
      }
      compatible_shape = std::move(static_compatible_shape);
    } else {
      // Dynamic input shape: create compatible shape at runtime with axes.
      // 1. Get the shape of the input tensor.
      const std::string input_shape_output = GenerateOperandName();
      const std::string shape_node_name = GenerateNodeName(
          GenerateEmulatedOpLabel(kOpTypeShape, layer_normalization.label));
      std::array<const char*, 1> shape_inputs = {input.c_str()};
      std::array<const char*, 1> shape_outputs = {input_shape_output.c_str()};
      model_editor_.AddNode(kOpTypeShape, shape_node_name, shape_inputs,
                            shape_outputs);

      // 2. Create a 1D tensor of all 1s with length = rank of input.
      // First, get the rank by getting the shape of the shape tensor.
      const std::string rank_output = GenerateOperandName();
      const std::string shape_of_shape_node_name =
          GenerateNodeName(GenerateEmulatedOpLabel(
              kOpTypeShape, layer_normalization.label, "rank"));
      std::array<const char*, 1> shape_of_shape_inputs = {
          input_shape_output.c_str()};
      std::array<const char*, 1> shape_of_shape_outputs = {rank_output.c_str()};
      model_editor_.AddNode(kOpTypeShape, shape_of_shape_node_name,
                            shape_of_shape_inputs, shape_of_shape_outputs);

      // Create a tensor of 1s with shape = [rank].
      const std::array<int64_t, 1> scalar_one_data = {1};
      const std::string scalar_one =
          Create1DInitializer<int64_t>(scalar_one_data);
      const std::string ones_shape_output = GenerateOperandName();
      const std::string expand_ones_node_name =
          GenerateNodeName(GenerateEmulatedOpLabel(
              kOpTypeExpand, layer_normalization.label, "ones"));
      std::array<const char*, 2> expand_ones_inputs = {scalar_one.c_str(),
                                                       rank_output.c_str()};
      std::array<const char*, 1> expand_ones_outputs = {
          ones_shape_output.c_str()};
      model_editor_.AddNode(kOpTypeExpand, expand_ones_node_name,
                            expand_ones_inputs, expand_ones_outputs);

      // 3. Gather the dimensions at axes positions from input_shape.
      const std::string axes_indices =
          CreateInt64InitializerForUint32Array(axes);
      const std::string gathered_dims = GenerateOperandName();
      const std::string gather_node_name = GenerateNodeName(
          GenerateEmulatedOpLabel("Gather", layer_normalization.label));
      std::array<const char*, 2> gather_inputs = {input_shape_output.c_str(),
                                                  axes_indices.c_str()};
      std::array<const char*, 1> gather_outputs = {gathered_dims.c_str()};
      std::array<ScopedOrtOpAttr, 1> gather_attributes = {
          model_editor_.CreateAttribute(kAttrAxis, int64_t{0})};
      model_editor_.AddNode(kOpTypeGather, gather_node_name, gather_inputs,
                            gather_outputs, gather_attributes);

      // 4. Use ScatterElements to update ones_shape at axes positions.
      const std::string compatible_shape_output = GenerateOperandName();
      const std::string scatter_node_name =
          GenerateNodeName(GenerateEmulatedOpLabel("ScatterElements",
                                                   layer_normalization.label));
      std::array<const char*, 3> scatter_inputs = {ones_shape_output.c_str(),
                                                   axes_indices.c_str(),
                                                   gathered_dims.c_str()};
      std::array<const char*, 1> scatter_outputs = {
          compatible_shape_output.c_str()};
      std::array<ScopedOrtOpAttr, 1> scatter_attributes = {
          model_editor_.CreateAttribute(kAttrAxis, int64_t{0})};
      model_editor_.AddNode(kOpTypeScatterElements, scatter_node_name,
                            scatter_inputs, scatter_outputs,
                            scatter_attributes);
      compatible_shape = compatible_shape_output;
    }

    // Handle scale operand using the compatible_shape variant.
    if (layer_normalization.scale_operand_id) {
      if (permutation.has_value()) {
        scale = CreateTransposeNode(scale, permutation.value());
      }
      if (std::holds_alternative<std::string>(compatible_shape)) {
        // Dynamic shape: reshape scale to compatible_shape.
        const std::string reshape_scale_node_name =
            GenerateNodeName(GenerateEmulatedOpLabel(
                "Reshape", layer_normalization.label, "scale"));
        std::array<const char*, 2> reshape_scale_inputs = {
            scale.c_str(), std::get<std::string>(compatible_shape).c_str()};
        const std::string reshaped_scale = GenerateOperandName();
        std::array<const char*, 1> reshape_scale_outputs = {
            reshaped_scale.c_str()};
        model_editor_.AddNode(kOpTypeReshape, reshape_scale_node_name,
                              reshape_scale_inputs, reshape_scale_outputs);
        scale = reshaped_scale;
      } else if (static_scale_shape.value().size() !=
                 static_input_shape.value().size()) {
        // Static shape: reshape if needed.
        scale = CreateReshapeNode(
            scale, std::get<std::vector<uint32_t>>(compatible_shape));
      }
    } else {
      if (std::holds_alternative<std::string>(compatible_shape)) {
        // Dynamic shape: create scale with axes, then reshape.
        std::string scale_with_axes =
            CreateInitializerWithInputShapeAndDataTypeForFloat(
                layer_normalization.input_operand_id, 1.0f, axes);
        const std::string reshape_scale_node_name =
            GenerateNodeName(GenerateEmulatedOpLabel(
                "Reshape", layer_normalization.label, "scale"));
        std::array<const char*, 2> reshape_scale_inputs = {
            scale_with_axes.c_str(),
            std::get<std::string>(compatible_shape).c_str()};
        const std::string reshaped_scale = GenerateOperandName();
        std::array<const char*, 1> reshape_scale_outputs = {
            reshaped_scale.c_str()};
        model_editor_.AddNode(kOpTypeReshape, reshape_scale_node_name,
                              reshape_scale_inputs, reshape_scale_outputs);
        scale = reshaped_scale;
      } else {
        // Static shape: create one initializer.
        scale = CreateOneInitializer(
            input_data_type, std::get<std::vector<uint32_t>>(compatible_shape));
      }
    }

    const std::string mul_label =
        GenerateEmulatedOpLabel(kOpTypeMul, layer_normalization.label);
    const std::string mul_node_name = GenerateNodeName(mul_label);
    std::array<const char*, 2> mul_inputs = {scale.c_str(), div_output.c_str()};
    if (layer_normalization.bias_operand_id) {
      const std::string mul_output = GenerateOperandName();
      std::array<const char*, 1> mul_outputs = {mul_output.c_str()};
      model_editor_.AddNode(kOpTypeMul, mul_node_name, mul_inputs, mul_outputs);
      if (permutation.has_value()) {
        bias = CreateTransposeNode(bias, permutation.value());
      }
      // Reshape bias to compatible_shape for broadcasting.
      if (std::holds_alternative<std::string>(compatible_shape)) {
        // Dynamic shape: reuse the dynamic compatible_shape.
        const std::string reshape_bias_node_name =
            GenerateNodeName(GenerateEmulatedOpLabel(
                "Reshape", layer_normalization.label, "bias"));
        std::array<const char*, 2> reshape_bias_inputs = {
            bias.c_str(), std::get<std::string>(compatible_shape).c_str()};
        const std::string reshaped_bias = GenerateOperandName();
        std::array<const char*, 1> reshape_bias_outputs = {
            reshaped_bias.c_str()};
        model_editor_.AddNode(kOpTypeReshape, reshape_bias_node_name,
                              reshape_bias_inputs, reshape_bias_outputs);
        bias = reshaped_bias;
      } else if (static_scale_shape.value().size() !=
                 static_input_shape.value().size()) {
        // Static shape: reuse the static compatible_shape.
        bias = CreateReshapeNode(
            bias, std::get<std::vector<uint32_t>>(compatible_shape));
      }

      const std::string add_2_label =
          GenerateEmulatedOpLabel(kOpTypeAdd, layer_normalization.label, "2");
      const std::string add_2_node_name = GenerateNodeName(add_2_label);
      std::array<const char*, 2> add_2_inputs = {mul_output.c_str(),
                                                 bias.c_str()};
      model_editor_.AddNode(kOpTypeAdd, add_2_node_name, add_2_inputs, outputs);
    } else {
      model_editor_.AddNode(kOpTypeMul, mul_node_name, mul_inputs, outputs);
    }
  }
}

void GraphBuilderOrt::AddLeakyReluOperation(
    const mojom::LeakyRelu& leaky_relu) {
  const std::string node_name = GenerateNodeName(leaky_relu.label);
  const std::string input = GetOperandNameById(leaky_relu.input_operand_id);
  const std::string output = GetOperandNameById(leaky_relu.output_operand_id);

  CHECK(context_properties_.data_type_limits.leaky_relu_input.Supports(
      GetOperand(leaky_relu.input_operand_id).descriptor));

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrAlpha, leaky_relu.alpha)};
  model_editor_.AddNode(kOpTypeLeakyRelu, node_name, inputs, outputs,
                        attributes);
}

void GraphBuilderOrt::AddLinearOperation(const mojom::Linear& linear) {
  const OperandDescriptor& input_descriptor =
      GetOperand(linear.input_operand_id).descriptor;
  CHECK(context_properties_.data_type_limits.linear_input.Supports(
      input_descriptor));

  // Emulate a linear operation using two ONNX nodes for expression `alpha * x +
  // beta`.
  const OperandDataType input_data_type = input_descriptor.data_type();
  std::string alpha = CreateScalarInitializer(
      input_data_type, MLNumber::FromFloat64(linear.alpha));
  std::string beta = CreateScalarInitializer(
      input_data_type, MLNumber::FromFloat64(linear.beta));

  // Step 1: Create 'Mul' node (alpha * x)
  const std::string mul_node_label =
      GenerateEmulatedOpLabel(kOpTypeMul, linear.label);
  const std::string mul_node_name = GenerateNodeName(mul_node_label);
  const std::string input = GetOperandNameById(linear.input_operand_id);
  std::array<const char*, 2> mul_inputs = {input.c_str(), alpha.c_str()};
  const std::string mul_output = GenerateOperandName();
  std::array<const char*, 1> mul_outputs = {mul_output.c_str()};
  model_editor_.AddNode(kOpTypeMul, mul_node_name, mul_inputs, mul_outputs);

  // Step 2: Create 'Add' node (mul_output + beta)
  const std::string add_node_label =
      GenerateEmulatedOpLabel(kOpTypeAdd, linear.label);
  const std::string add_node_name = GenerateNodeName(add_node_label);
  std::array<const char*, 2> add_inputs = {mul_output.c_str(), beta.c_str()};
  const std::string output = GetOperandNameById(linear.output_operand_id);
  std::array<const char*, 1> add_outputs = {output.c_str()};
  model_editor_.AddNode(kOpTypeAdd, add_node_name, add_inputs, add_outputs);
}

// `LstmType` must be `mojom::Lstm` or `mojom::LstmCell`.
template <typename LstmType>
  requires(std::is_same_v<LstmType, mojom::Lstm> ||
           std::is_same_v<LstmType, mojom::LstmCell>)
void GraphBuilderOrt::AddLstmOperation(const LstmType& lstm) {
  const std::string node_name = GenerateNodeName(lstm.label);
  std::string input = GetOperandNameById(lstm.input_operand_id);
  std::string weight = GetOperandNameById(lstm.weight_operand_id);
  std::string recurrent_weight =
      GetOperandNameById(lstm.recurrent_weight_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(lstm.input_operand_id).descriptor;
  const OperandDescriptor& weight_descriptor =
      GetOperand(lstm.weight_operand_id).descriptor;
  const OperandDescriptor& recurrent_weight_descriptor =
      GetOperand(lstm.recurrent_weight_operand_id).descriptor;

  uint32_t num_directions = 1;
  if constexpr (std::is_same_v<LstmType, mojom::Lstm>) {
    CHECK(context_properties_.data_type_limits.lstm_input.Supports(
        input_descriptor));
    CHECK(context_properties_.data_type_limits.lstm_input.Supports(
        weight_descriptor));
    CHECK(context_properties_.data_type_limits.lstm_input.Supports(
        recurrent_weight_descriptor));
    num_directions =
        lstm.direction == mojom::RecurrentNetworkDirection::kBoth ? 2 : 1;
  } else {
    CHECK(context_properties_.data_type_limits.lstm_cell_input.Supports(
        input_descriptor));
    CHECK(context_properties_.data_type_limits.lstm_cell_input.Supports(
        weight_descriptor));
    CHECK(context_properties_.data_type_limits.lstm_cell_input.Supports(
        recurrent_weight_descriptor));

    // Reshape the input into a 3-D tensor, since the LSTM of ONNX requires
    // the input shape to be [seq_length, batch_size, input_size]. For
    // lstmCell, `seq_length` is equal to 1.
    const std::vector<uint32_t> input_shape =
        input_descriptor.StaticShape().value();
    CHECK_EQ(input_shape.size(), 2u);
    input = CreateReshapeNode(input, {1, input_shape[0], input_shape[1]});

    // Reshape the weight into a 3-D tensor, since the LSTM of ONNX requires
    // the weight shape to be [num_directions, 4*hidden_size, input_size].
    // For lstmCell, `num_directions` is equal to 1.
    const std::vector<uint32_t> weight_shape =
        weight_descriptor.StaticShape().value();
    CHECK_EQ(weight_shape.size(), 2u);
    weight = CreateReshapeNode(weight, {1, weight_shape[0], weight_shape[1]});

    // Reshape the recurrent weight into a 3-D tensor, since the LSTM of ONNX
    // requires the recurrent weight shape to be [num_directions,
    // 4*hidden_size, hidden_size]. For lstmCell, `num_directions` is equal
    // to 1.
    const std::vector<uint32_t> recurrent_weight_shape =
        recurrent_weight_descriptor.StaticShape().value();
    CHECK_EQ(recurrent_weight_shape.size(), 2u);
    recurrent_weight = CreateReshapeNode(
        recurrent_weight,
        {1, recurrent_weight_shape[0], recurrent_weight_shape[1]});
  }

  constexpr std::array<uint32_t, 4> kIfgoToIofgPermutation = {0, 3, 1, 2};
  if (lstm.layout == mojom::LstmWeightLayout::kIfgo) {
    weight = TransposeRnnWeightOrBiasLayout(weight, kIfgoToIofgPermutation);
    recurrent_weight = TransposeRnnWeightOrBiasLayout(recurrent_weight,
                                                      kIfgoToIofgPermutation);
  }

  std::vector<const char*> inputs = {input.c_str(), weight.c_str(),
                                     recurrent_weight.c_str()};

  const uint32_t hidden_size = lstm.hidden_size;
  // Graph validation already checked that hidden_size * 4 would not overflow.
  std::array<uint32_t, 2> bias_dims = {num_directions, hidden_size * 4};
  std::string bias, recurrent_bias, concatenated_bias;
  if (!lstm.bias_operand_id.has_value() &&
      !lstm.recurrent_bias_operand_id.has_value()) {
    // When both bias and recurrentBias are not present, set ONNX LSTM input "B"
    // as not specified.
    inputs.push_back("");
  } else {
    if (lstm.bias_operand_id.has_value()) {
      bias = GetOperandNameById(*lstm.bias_operand_id);
      if constexpr (std::is_same_v<LstmType, mojom::Lstm>) {
        CHECK(context_properties_.data_type_limits.lstm_bias.Supports(
            GetOperand(*lstm.bias_operand_id).descriptor));
      } else {
        CHECK(context_properties_.data_type_limits.lstm_cell_bias.Supports(
            GetOperand(*lstm.bias_operand_id).descriptor));
        // Reshape the bias into a 2-D tensor, since the LSTM of ONNX requires
        // the bias shape to be [num_directions, 4*hidden_size]. For lstmCell,
        // `num_directions` is equal to 1.
        bias = CreateReshapeNode(bias, bias_dims);
      }
      if (lstm.layout == mojom::LstmWeightLayout::kIfgo) {
        bias = TransposeRnnWeightOrBiasLayout(bias, kIfgoToIofgPermutation);
      }
    } else {
      bias = CreateZeroInitializer(input_descriptor.data_type(), bias_dims);
    }

    if (lstm.recurrent_bias_operand_id.has_value()) {
      recurrent_bias = GetOperandNameById(*lstm.recurrent_bias_operand_id);
      if constexpr (std::is_same_v<LstmType, mojom::Lstm>) {
        CHECK(context_properties_.data_type_limits.lstm_bias.Supports(
            GetOperand(*lstm.recurrent_bias_operand_id).descriptor));
      } else {
        CHECK(context_properties_.data_type_limits.lstm_cell_bias.Supports(
            GetOperand(*lstm.recurrent_bias_operand_id).descriptor));
        // Reshape the recurrentBias into a 2-D tensor, since the LSTM of ONNX
        // requires the recurrentBias shape to be [num_directions,
        // 4*hidden_size]. For lstmCell, `num_directions` is equal to 1.
        recurrent_bias = CreateReshapeNode(recurrent_bias, bias_dims);
      }
      if (lstm.layout == mojom::LstmWeightLayout::kIfgo) {
        recurrent_bias = TransposeRnnWeightOrBiasLayout(recurrent_bias,
                                                        kIfgoToIofgPermutation);
      }
    } else {
      recurrent_bias =
          CreateZeroInitializer(input_descriptor.data_type(), bias_dims);
    }

    // Concatenate bias and recurrent_bias to create the ONNX LSTM input "B".
    concatenated_bias = GenerateOperandName();
    std::array<const char*, 2> bias_inputs = {bias.c_str(),
                                              recurrent_bias.c_str()};
    std::array<const char*, 1> bias_outputs = {concatenated_bias.c_str()};
    std::array<ScopedOrtOpAttr, 1> bias_attributes = {
        model_editor_.CreateAttribute(kAttrAxis, static_cast<int64_t>(1))};
    std::string concat_node_name = GenerateNodeName(
        base::JoinString({kInserted, kOpTypeConcat}, kUnderscore));
    model_editor_.AddNode(kOpTypeConcat, concat_node_name, bias_inputs,
                          bias_outputs, bias_attributes);
    inputs.push_back(concatenated_bias.c_str());
  }

  // "sequence_lens" is an optional tensor specifying lengths of the sequences
  // in a batch.
  inputs.push_back("");

  std::string hidden_state, cell_state;
  if constexpr (std::is_same_v<LstmType, mojom::Lstm>) {
    if (lstm.initial_hidden_state_operand_id.has_value()) {
      hidden_state = GetOperandNameById(*lstm.initial_hidden_state_operand_id);
      CHECK(context_properties_.data_type_limits.lstm_input.Supports(
          GetOperand(*lstm.initial_hidden_state_operand_id).descriptor));
    }
    if (lstm.initial_cell_state_operand_id.has_value()) {
      cell_state = GetOperandNameById(*lstm.initial_cell_state_operand_id);
      CHECK(context_properties_.data_type_limits.lstm_input.Supports(
          GetOperand(*lstm.initial_cell_state_operand_id).descriptor));
    }
  } else {
    hidden_state = GetOperandNameById(lstm.hidden_state_operand_id);
    const OperandDescriptor& hidden_state_descriptor =
        GetOperand(lstm.hidden_state_operand_id).descriptor;
    CHECK(context_properties_.data_type_limits.lstm_cell_input.Supports(
        hidden_state_descriptor));
    cell_state = GetOperandNameById(lstm.cell_state_operand_id);
    const OperandDescriptor& cell_state_descriptor =
        GetOperand(lstm.cell_state_operand_id).descriptor;
    CHECK(context_properties_.data_type_limits.lstm_cell_input.Supports(
        cell_state_descriptor));

    // Reshape the hidden/cell_state into a 3-D tensor, since the LSTM of ONNX
    // requires the "initial_h"/"initial_c" shape to be [num_directions,
    // batch_size, hidden_size]. For lstmCell, `num_directions` is equal to 1.
    const std::vector<uint32_t> hidden_state_shape =
        hidden_state_descriptor.StaticShape().value();
    const std::vector<uint32_t> cell_state_shape =
        cell_state_descriptor.StaticShape().value();
    hidden_state = CreateReshapeNode(
        hidden_state, {1, hidden_state_shape[0], hidden_state_shape[1]});
    cell_state = CreateReshapeNode(
        cell_state, {1, cell_state_shape[0], cell_state_shape[1]});
  }
  inputs.push_back(hidden_state.c_str());
  inputs.push_back(cell_state.c_str());

  std::string peephole_weight;
  if (lstm.peephole_weight_operand_id.has_value()) {
    peephole_weight = GetOperandNameById(*lstm.peephole_weight_operand_id);
    if constexpr (std::is_same_v<LstmType, mojom::Lstm>) {
      CHECK(context_properties_.data_type_limits.lstm_bias.Supports(
          GetOperand(*lstm.peephole_weight_operand_id).descriptor));
    } else {
      const OperandDescriptor& peephole_weight_descriptor =
          GetOperand(*lstm.peephole_weight_operand_id).descriptor;
      CHECK(context_properties_.data_type_limits.lstm_cell_bias.Supports(
          peephole_weight_descriptor));
      // Reshape the peephole_weight into a 2-D tensor, since the LSTM of ONNX
      // requires the peephole_weight shape to be [num_directions,
      // 3*hidden_size]. For lstmCell, `num_directions` is equal to 1.
      const std::vector<uint32_t> peephole_weight_shape =
          peephole_weight_descriptor.StaticShape().value();
      peephole_weight =
          CreateReshapeNode(peephole_weight, {1, peephole_weight_shape[0]});
    }
  }
  inputs.push_back(peephole_weight.c_str());

  std::vector<ScopedOrtOpAttr> attributes;
  attributes.reserve(3);
  base::cstring_view direction = "forward";
  if constexpr (std::is_same_v<LstmType, mojom::Lstm>) {
    direction = GetRecurrentNetworkDirection(lstm.direction);
  }
  attributes.push_back(
      model_editor_.CreateAttribute(kAttrDirection, direction));

  const std::vector<base::cstring_view> activations =
      GetRecurrentNetworkActivations(lstm.activations,
                                     direction == "bidirectional");
  std::vector<const char*> activations_c_str;
  for (const auto& activation : activations) {
    activations_c_str.push_back(activation.c_str());
  }
  attributes.push_back(
      model_editor_.CreateAttribute(kAttrActivations, activations_c_str));

  attributes.push_back(model_editor_.CreateAttribute(
      kAttrHiddenSize, base::checked_cast<int64_t>(hidden_size)));

  std::string output, output_hidden, output_cell;
  if constexpr (std::is_same_v<LstmType, mojom::Lstm>) {
    output_hidden = GetOperandNameById(lstm.output_operand_ids[0]);
    output_cell = GetOperandNameById(lstm.output_operand_ids[1]);
    if (lstm.return_sequence) {
      output = GetOperandNameById(lstm.output_operand_ids[2]);
    }
  } else {
    output_hidden = GenerateOperandName();
    output_cell = GenerateOperandName();
  }
  std::array<const char*, 3> outputs = {output.c_str(), output_hidden.c_str(),
                                        output_cell.c_str()};
  model_editor_.AddNode(kOpTypeLstm, node_name, inputs, outputs, attributes);

  if constexpr (std::is_same_v<LstmType, mojom::LstmCell>) {
    // Reshape the output_hidden and output_cell back to 2-D tensors, since the
    // LSTM of WebNN requires the "output_h"/"output_c" shape to be
    // [batch_size, hidden_size].
    const std::vector<uint32_t> output_shape =
        GetOperand(lstm.output_operand_ids[0]).descriptor.StaticShape().value();
    CHECK_EQ(output_shape.size(), 2u);
    InsertReshapeNode(output_hidden,
                      GetOperandNameById(lstm.output_operand_ids[0]),
                      output_shape);
    InsertReshapeNode(output_cell,
                      GetOperandNameById(lstm.output_operand_ids[1]),
                      output_shape);
  }
}

template void GraphBuilderOrt::AddLstmOperation(const mojom::Lstm& lstm);

template void GraphBuilderOrt::AddLstmOperation(
    const mojom::LstmCell& lstm_cell);

base::expected<void, mojom::ErrorPtr> GraphBuilderOrt::AddMatMulOperation(
    const mojom::Matmul& matmul) {
  const std::string node_name = GenerateNodeName(matmul.label);
  const std::string input_a = GetOperandNameById(matmul.a_operand_id);
  const std::string input_b = GetOperandNameById(matmul.b_operand_id);
  const std::string output = GetOperandNameById(matmul.output_operand_id);

  CHECK(context_properties_.data_type_limits.matmul_input.SupportsAll(
      {GetOperand(matmul.a_operand_id).descriptor,
       GetOperand(matmul.b_operand_id).descriptor}));

  if (batched_matmul_k_dimension_limit_.has_value()) {
    bool is_batched_matmul =
        GetOperand(matmul.output_operand_id).descriptor.Rank() > 2;
    webnn::Dimension k_dim =
        GetOperand(matmul.a_operand_id).descriptor.shape().back();
    if (is_batched_matmul && std::holds_alternative<uint32_t>(k_dim)) {
      uint32_t batched_matmul_k_dimension_size = std::get<uint32_t>(k_dim);
      // Limitation: Reject batched MatMul operations with excessively large K
      // dimension size to prevent the EP from becoming unresponsive during
      // model compilation on some NPU devices.
      // OpenVINO issue: https://github.com/microsoft/onnxruntime/issues/26643
      // The fix is expected to be available in NPU driver Feb '26 release.
      //
      // TODO(crbug.com/468812994): Check the version of OV EP or NPU driver
      // before applying the Limitation.
      // TODO(crbug.com/467468912): When the OpenVINO issue is fixed, remove
      // the limitation and increase the minimum required EP version.
      if (batched_matmul_k_dimension_size >
          batched_matmul_k_dimension_limit_.value()) {
        return base::unexpected(mojom::Error::New(
            mojom::Error::Code::kNotSupportedError,
            "The K dimension size of the batched MatMul operation is too "
            "large which is not supported on NPU."));
      }
    }
  }

  std::array<const char*, 2> inputs = {input_a.c_str(), input_b.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeMatMul, node_name, inputs, outputs);

  return base::ok();
}

void GraphBuilderOrt::AddPool2dOperation(const mojom::Pool2d& pool2d) {
  if (!pool2d.window_dimensions) {
    base::cstring_view op_type;
    std::vector<ScopedOrtOpAttr> attributes;
    const OperandDescriptor& input_descriptor =
        GetOperand(pool2d.input_operand_id).descriptor;
    const DataTypeLimits& data_type_limits =
        context_properties_.data_type_limits;
    switch (pool2d.kind) {
      case mojom::Pool2d::Kind::kAveragePool2d:
        CHECK(data_type_limits.average_pool2d_input.Supports(input_descriptor));
        op_type = "GlobalAveragePool";
        break;
      case mojom::Pool2d::Kind::kL2Pool2d:
        CHECK(data_type_limits.l2_pool2d_input.Supports(input_descriptor));
        op_type = "GlobalLpPool";
        attributes.push_back(
            model_editor_.CreateAttribute(kAttrP, static_cast<int64_t>(2)));
        break;
      case mojom::Pool2d::Kind::kMaxPool2d:
        CHECK(data_type_limits.max_pool2d_input.Supports(input_descriptor));
        op_type = "GlobalMaxPool";
        break;
    }
    const std::string node_name = GenerateNodeName(pool2d.label);
    const std::string input = GetOperandNameById(pool2d.input_operand_id);
    const std::string output = GetOperandNameById(pool2d.output_operand_id);
    std::array<const char*, 1> inputs = {input.c_str()};
    std::array<const char*, 1> outputs = {output.c_str()};
    model_editor_.AddNode(op_type, node_name, inputs, outputs, attributes);
    return;
  }

  std::vector<ScopedOrtOpAttr> attributes;

  std::array<int64_t, 2> dilations = {
      base::checked_cast<int64_t>(pool2d.dilations->height),
      base::checked_cast<int64_t>(pool2d.dilations->width)};
  attributes.push_back(
      model_editor_.CreateAttribute(kAttrDilations, dilations));

  std::array<int64_t, 2> strides = {
      base::checked_cast<int64_t>(pool2d.strides->height),
      base::checked_cast<int64_t>(pool2d.strides->width)};
  attributes.push_back(model_editor_.CreateAttribute(kAttrStrides, strides));

  std::array<int64_t, 2> window_dimensions = {
      base::checked_cast<int64_t>(pool2d.window_dimensions->height),
      base::checked_cast<int64_t>(pool2d.window_dimensions->width)};
  attributes.push_back(
      model_editor_.CreateAttribute(kAttrKernelShape, window_dimensions));

  // ONNX's pads are [beginning_height, beginning_width, ending_height,
  // ending_width].
  std::array<int64_t, 4> pads = {
      base::checked_cast<int64_t>(pool2d.padding->beginning->height),
      base::checked_cast<int64_t>(pool2d.padding->beginning->width),
      base::checked_cast<int64_t>(pool2d.padding->ending->height),
      base::checked_cast<int64_t>(pool2d.padding->ending->width)};
  attributes.push_back(model_editor_.CreateAttribute(kAttrPads, pads));

  // Calculate the ceil_mode.
  const OperandDescriptor& input_descriptor =
      GetOperand(pool2d.input_operand_id).descriptor;
  const std::vector<webnn::Dimension>& input_shape = input_descriptor.shape();
  const std::vector<webnn::Dimension>& output_shape =
      GetOperand(pool2d.output_operand_id).descriptor.shape();

  CHECK_EQ(context_properties_.input_operand_layout, InputOperandLayout::kNchw);
  const uint32_t* input_height_opt = std::get_if<uint32_t>(&input_shape[2]);
  const uint32_t* output_height_opt = std::get_if<uint32_t>(&output_shape[2]);
  // For ceil_mode calculation, if spatial dims are dynamic, default to floor
  // mode (ceil_mode=0).
  int64_t ceil_mode = 0;
  if (input_height_opt && output_height_opt) {
    uint32_t input_height = *input_height_opt;
    uint32_t output_height = *output_height_opt;
    const auto float_output_height = CalculateConv2dOutputSize(
        input_height, pool2d.window_dimensions->height,
        pool2d.padding->beginning->height, pool2d.padding->ending->height,
        pool2d.strides->height, pool2d.dilations->height, pool2d.label);
    CHECK(float_output_height.has_value());
    ceil_mode = float_output_height.value() < output_height ? 1 : 0;
  }
  attributes.push_back(model_editor_.CreateAttribute(kAttrCeilMode, ceil_mode));

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  base::cstring_view op_type;
  switch (pool2d.kind) {
    case mojom::Pool2d::Kind::kAveragePool2d: {
      CHECK(data_type_limits.average_pool2d_input.Supports(input_descriptor));
      op_type = kOpTypeAveragePool2d;
      break;
    }
    case mojom::Pool2d::Kind::kMaxPool2d: {
      CHECK(data_type_limits.max_pool2d_input.Supports(input_descriptor));
      op_type = kOpTypeMaxPool2d;
      break;
    }
    case mojom::Pool2d::Kind::kL2Pool2d: {
      CHECK(data_type_limits.l2_pool2d_input.Supports(input_descriptor));
      op_type = kOpTypeLpPool2d;
      attributes.push_back(
          model_editor_.CreateAttribute(kAttrP, static_cast<int64_t>(2)));
      break;
    }
  }

  const std::string node_name = GenerateNodeName(pool2d.label);
  const std::string input = GetOperandNameById(pool2d.input_operand_id);
  const std::string output = GetOperandNameById(pool2d.output_operand_id);
  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(op_type, node_name, inputs, outputs, attributes);
}

void GraphBuilderOrt::AddReduceOperation(const mojom::Reduce& reduce) {
  const std::string input = GetOperandNameById(reduce.input_operand_id);
  const std::string output = GetOperandNameById(reduce.output_operand_id);
  std::vector<const char*> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  CheckReduceInputSupported(context_properties_.data_type_limits, reduce.kind,
                            GetOperand(reduce.input_operand_id).descriptor);

  // According to
  // https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-reduce,
  // if axes is empty, WebNN applies reduction function to each value in the
  // tensor individually with no dimensions reduced, but the ONNX reduction
  // operations either reduce all dimensions or act as a no-op. So we need to
  // emulate the behavior of reducing each value individually:
  // 1. Element-wise log for reduceLogSum
  // 2. Element-wise pow of 2 for reduceSumSquare
  // 3. Element-wise abs for reduceL1 and reduceL2
  // 4. No-op for other reduction operations e.g. reduceMin and reduceSum
  //
  // TODO(crbug.com/429272269): Remove the workaround for reduction operations
  // when ORT issue is fixed.
  // https://github.com/onnx/onnx/issues/6103
  if (reduce.axes.empty()) {
    switch (reduce.kind) {
      case mojom::Reduce::Kind::kLogSum: {
        const std::string node_name =
            GenerateNodeName(GenerateEmulatedOpLabel(kOpTypeLog, reduce.label));
        model_editor_.AddNode(kOpTypeLog, node_name, inputs, outputs);
        return;
      }
      case mojom::Reduce::Kind::kSumSquare: {
        const std::string node_name =
            GenerateNodeName(GenerateEmulatedOpLabel(kOpTypePow, reduce.label));
        const std::string pow = CreateScalarInitializer<int64_t>(2);
        inputs.push_back(pow.c_str());
        model_editor_.AddNode(kOpTypePow, node_name, inputs, outputs);
        return;
      }
      case mojom::Reduce::Kind::kL1:
      case mojom::Reduce::Kind::kL2: {
        const std::string node_name =
            GenerateNodeName(GenerateEmulatedOpLabel(kOpTypeAbs, reduce.label));
        model_editor_.AddNode(kOpTypeAbs, node_name, inputs, outputs);
        return;
      }
      case mojom::Reduce::Kind::kLogSumExp:
      case mojom::Reduce::Kind::kMax:
      case mojom::Reduce::Kind::kMean:
      case mojom::Reduce::Kind::kMin:
      case mojom::Reduce::Kind::kProduct:
      case mojom::Reduce::Kind::kSum:
        // Setting the `noop_with_empty_axes` attribute to 1 will make them act
        // as a no-op.
        break;
    }
  }

  const std::string axes = CreateInt64InitializerForUint32Array(reduce.axes);
  inputs.push_back(axes.c_str());

  int64_t keepdims = reduce.keep_dimensions ? 1 : 0;
  int64_t noop_with_empty_axes = 1;
  std::array<ScopedOrtOpAttr, 2> attributes = {
      model_editor_.CreateAttribute(kAttrKeepDims, keepdims),
      model_editor_.CreateAttribute(kAttrNoopWithEmptyAxes,
                                    noop_with_empty_axes),
  };

  const std::string node_name = GenerateNodeName(reduce.label);
  base::cstring_view reduce_op_type = MapReduceKindToOrtOpType(reduce.kind);
  model_editor_.AddNode(reduce_op_type, node_name, inputs, outputs, attributes);
}

void GraphBuilderOrt::AddResample2dOperation(
    const mojom::Resample2d& resample2d) {
  const std::string node_name = GenerateNodeName(resample2d.label);
  const std::string input = GetOperandNameById(resample2d.input_operand_id);
  const std::string output = GetOperandNameById(resample2d.output_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(resample2d.input_operand_id).descriptor;
  CHECK(context_properties_.data_type_limits.resample2d_input.Supports(
      input_descriptor));

  std::string scales;
  std::string sizes;
  if (resample2d.scales) {
    // Each element of scales applies to a dimension of the input.
    CHECK_EQ(input_descriptor.Rank(), 4u);
    std::array<float, 4> scales_data = {1.f, 1.f, 1.f, 1.f};
    CHECK_EQ(resample2d.axes.size(), 2u);
    CHECK_EQ(resample2d.scales->size(), 2u);
    scales_data.at(resample2d.axes[0]) = resample2d.scales->at(0);
    scales_data.at(resample2d.axes[1]) = resample2d.scales->at(1);
    scales = Create1DInitializer<float>(scales_data);
  } else {
    sizes = CreateInt64InitializerForUint32Array(
        GetOperand(resample2d.output_operand_id)
            .descriptor.StaticShape()
            .value());
  }

  std::string mode;
  switch (resample2d.mode) {
    case mojom::Resample2d::InterpolationMode::kLinear:
      mode = "linear";
      break;
    case mojom::Resample2d::InterpolationMode::kNearestNeighbor:
      mode = "nearest";
      break;
  }

  AddResizeNode(node_name, input, scales, sizes, mode, output);
}

void GraphBuilderOrt::AddReshapeOperation(const mojom::Reshape& reshape) {
  const std::string node_name = GenerateNodeName(reshape.label);
  const std::string input = GetOperandNameById(reshape.input_operand_id);
  const std::string output = GetOperandNameById(reshape.output_operand_id);

  CHECK(context_properties_.data_type_limits.reshape_input.Supports(
      GetOperand(reshape.input_operand_id).descriptor));

  const OperandDescriptor& input_descriptor =
      GetOperand(reshape.input_operand_id).descriptor;
  const OperandDescriptor& output_descriptor =
      GetOperand(reshape.output_operand_id).descriptor;

  const auto static_output_shape = output_descriptor.StaticShape();
  if (static_output_shape.has_value()) {
    // All dimensions are static, use the simple path.
    AddReshapeNode(node_name, input, output, *static_output_shape);
    return;
  }

  // Output shape has dynamic dimensions. Build the shape tensor at runtime
  // using Shape and Gather operators.

  // Step 1: Get the input shape at runtime using Shape operator.
  const std::string input_shape_name = GenerateOperandName();
  {
    std::array<const char*, 1> shape_inputs = {input.c_str()};
    std::array<const char*, 1> shape_outputs = {input_shape_name.c_str()};
    const std::string shape_node_name = GenerateNodeName(base::JoinString(
        {kInserted, "Shape", kToEmulate, reshape.label}, kUnderscore));
    model_editor_.AddNode("Shape", shape_node_name, shape_inputs,
                          shape_outputs);
  }

  // Step 2: For each dimension in output shape, either use a constant (for
  // static dims) or gather from input shape (for dynamic dims).
  const std::vector<webnn::Dimension>& output_shape = output_descriptor.shape();
  std::vector<std::string> dimension_names;
  dimension_names.reserve(output_shape.size());

  // Build a mapping from input shape to find dynamic dimensions.
  const std::vector<webnn::Dimension>& input_shape = input_descriptor.shape();

  // Track the index of the inferred dimension (if any) for post-loop
  // computation.
  size_t inferred_dim_index = output_shape.size();

  for (size_t i = 0; i < output_shape.size(); ++i) {
    const auto& dim = output_shape[i];
    if (std::holds_alternative<uint32_t>(dim)) {
      // Static dimension: create a 1-D constant with shape [1].
      uint32_t static_value = std::get<uint32_t>(dim);
      std::array<int64_t, 1> value_array = {static_cast<int64_t>(static_value)};
      std::string const_name = Create1DInitializer<int64_t>(value_array);
      dimension_names.push_back(std::move(const_name));
    } else {
      // Dynamic dimension: first try to find it in the input shape.
      CHECK(std::holds_alternative<DynamicDimension>(dim));

      auto it = std::ranges::find(input_shape, dim);
      if (it != input_shape.end()) {
        // Directly matching dynamic dimension: gather from input shape.
        int64_t input_axis = std::distance(input_shape.begin(), it);

        const std::string gather_output = GenerateOperandName();
        std::array<int64_t, 1> indices_array = {input_axis};
        const std::string indices_const =
            Create1DInitializer<int64_t>(indices_array);

        std::array<const char*, 2> gather_inputs = {input_shape_name.c_str(),
                                                    indices_const.c_str()};
        std::array<const char*, 1> gather_outputs = {gather_output.c_str()};
        std::array<ScopedOrtOpAttr, 1> gather_attributes = {
            model_editor_.CreateAttribute(kAttrAxis, static_cast<int64_t>(0))};
        const std::string gather_node_name = GenerateNodeName(
            base::JoinString({kInserted, kOpTypeGather, kToEmulate,
                              reshape.label, base::NumberToString(input_axis)},
                             kUnderscore));
        model_editor_.AddNode(kOpTypeGather, gather_node_name, gather_inputs,
                              gather_outputs, gather_attributes);

        dimension_names.push_back(std::move(gather_output));
      } else {
        // Derived dynamic dimension (not in input shape): will be computed
        // after the loop from the input shape. At most one inferred dimension
        // is allowed per reshape.
        CHECK_EQ(inferred_dim_index, output_shape.size())
            << "Only one inferred dimension is allowed per reshape";
        inferred_dim_index = i;
        dimension_names.push_back(std::string());  // placeholder
      }
    }
  }

  // If there is an inferred dimension (a derived dynamic dim not found in the
  // input shape), use -1 for the reshape shape tensor so that ONNX Runtime
  // infers it at runtime. The dynamic dim will be registered against the
  // reshape output operand by the pre-pass in BuildModel(), so downstream
  // operations that reference it will emit Shape(reshape_output) + Gather.
  if (inferred_dim_index < output_shape.size()) {
    std::array<int64_t, 1> minus_one = {-1};
    dimension_names[inferred_dim_index] =
        Create1DInitializer<int64_t>(minus_one);
  }

  // Step 3: Concatenate all dimension values to create the final shape tensor.
  std::string final_shape_name;
  if (dimension_names.size() == 1) {
    // Single dimension, no need to concatenate.
    final_shape_name = dimension_names[0];
  } else {
    // Multiple dimensions, concatenate them.
    final_shape_name = GenerateOperandName();
    std::vector<const char*> concat_inputs;
    concat_inputs.reserve(dimension_names.size());
    for (const auto& name : dimension_names) {
      concat_inputs.push_back(name.c_str());
    }
    std::array<const char*, 1> concat_outputs = {final_shape_name.c_str()};
    std::array<ScopedOrtOpAttr, 1> concat_attributes = {
        model_editor_.CreateAttribute(kAttrAxis, static_cast<int64_t>(0))};
    const std::string concat_node_name = GenerateNodeName(base::JoinString(
        {kInserted, kOpTypeConcat, kToEmulate, reshape.label}, kUnderscore));
    model_editor_.AddNode(kOpTypeConcat, concat_node_name, concat_inputs,
                          concat_outputs, concat_attributes);
  }

  // Step 4: Use the constructed shape tensor for the Reshape operation.
  std::array<const char*, 2> reshape_inputs = {input.c_str(),
                                               final_shape_name.c_str()};
  std::array<const char*, 1> reshape_outputs = {output.c_str()};
  model_editor_.AddNode(kOpTypeReshape, node_name, reshape_inputs,
                        reshape_outputs);
}

void GraphBuilderOrt::AddReverseOperation(const mojom::Reverse& reverse) {
  const std::string input = GetOperandNameById(reverse.input_operand_id);
  const std::string output = GetOperandNameById(reverse.output_operand_id);

  CHECK(context_properties_.data_type_limits.reverse_input.Supports(
      GetOperand(reverse.input_operand_id).descriptor));

  // Workaround: explicitly empty axes for a reverse operation should result in
  // a no-op per spec. But we map this to an Identity node to prevent ORT
  // EPs from mishandling empty arrays.
  // TODO(crbug.com/500385615): Remove the workaround when the issue is fixed.
  if (reverse.axes.empty()) {
    EmulateWithIdentityNode(reverse.label, input, output);
    return;
  }

  const std::string node_name = GenerateNodeName(reverse.label);

  // Axes can be empty, which means no dimensions are reversed.
  base::FixedArray<int64_t> axes(reverse.axes.begin(), reverse.axes.end());
  size_t axes_size = axes.size();

  // Emulate reverse operation using backward slice with negative steps.
  // For each axis to be reversed:
  // - start = -1 (last element)
  // - end = min_int64 (goes to the beginning)
  // - step = -1 (backward direction)
  base::FixedArray<int64_t> starts(axes_size, -1);
  base::FixedArray<int64_t> ends(axes_size,
                                 std::numeric_limits<int64_t>::min());
  base::FixedArray<int64_t> steps(axes_size, -1);

  return AddSliceNode(node_name, input, output, axes, starts, ends, steps);
}

void GraphBuilderOrt::AddShapeOperation(const mojom::Shape& shape) {
  const std::string input = GetOperandNameById(shape.input_operand_id);
  const std::string output = GetOperandNameById(shape.output_operand_id);

  CHECK(context_properties_.data_type_limits.shape_input.Supports(
      GetOperand(shape.input_operand_id).descriptor));

  AddUnaryOperation(shape, "Shape");
}

void GraphBuilderOrt::AddScatterElementsOperation(
    const mojom::ScatterElements& scatter_elements) {
  const std::string node_name = GenerateNodeName(scatter_elements.label);
  const std::string input =
      GetOperandNameById(scatter_elements.input_operand_id);
  const std::string indices =
      GetOperandNameById(scatter_elements.indices_operand_id);
  const std::string updates =
      GetOperandNameById(scatter_elements.updates_operand_id);
  const std::string output =
      GetOperandNameById(scatter_elements.output_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(scatter_elements.input_operand_id).descriptor;
  const OperandDescriptor& updates_descriptor =
      GetOperand(scatter_elements.updates_operand_id).descriptor;
  const OperandDescriptor& indices_descriptor =
      GetOperand(scatter_elements.indices_operand_id).descriptor;
  CHECK(context_properties_.data_type_limits.scatter_elements_input.SupportsAll(
      {input_descriptor, updates_descriptor}));
  CHECK(context_properties_.data_type_limits.scatter_elements_indices.Supports(
      indices_descriptor));

  // Clamp the indices operand to prevent out-of-bounds writing which will cause
  // ORT CPU EP to throw a runtime error.
  const Dimension& axis_dimension =
      input_descriptor.shape()[scatter_elements.axis];
  std::string clamped_indices;
  if (std::holds_alternative<uint32_t>(axis_dimension)) {
    clamped_indices = ClampIndices(indices, indices_descriptor.data_type(),
                                   std::get<uint32_t>(axis_dimension));
  } else {
    // For dynamic dimension, clamp indices at runtime.
    clamped_indices = ClampIndicesForDynamicShape(
        node_name, input, indices, scatter_elements.axis,
        indices_descriptor.data_type());
  }

  std::array<const char*, 3> inputs = {input.c_str(), clamped_indices.c_str(),
                                       updates.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  std::array<ScopedOrtOpAttr, 1> attributes = {model_editor_.CreateAttribute(
      kAttrAxis, static_cast<int64_t>(scatter_elements.axis))};

  model_editor_.AddNode(kOpTypeScatterElements, node_name, inputs, outputs,
                        attributes);
}

void GraphBuilderOrt::AddScatterNDOperation(
    const mojom::ScatterND& scatter_nd) {
  const std::string node_name = GenerateNodeName(scatter_nd.label);
  const std::string input = GetOperandNameById(scatter_nd.input_operand_id);
  const std::string indices = GetOperandNameById(scatter_nd.indices_operand_id);
  const std::string updates = GetOperandNameById(scatter_nd.updates_operand_id);
  const std::string output = GetOperandNameById(scatter_nd.output_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(scatter_nd.input_operand_id).descriptor;
  const OperandDescriptor& updates_descriptor =
      GetOperand(scatter_nd.updates_operand_id).descriptor;
  const OperandDescriptor& indices_descriptor =
      GetOperand(scatter_nd.indices_operand_id).descriptor;
  CHECK(context_properties_.data_type_limits.scatter_nd_input.Supports(
      input_descriptor));
  CHECK(context_properties_.data_type_limits.scatter_nd_updates.Supports(
      updates_descriptor));
  CHECK(context_properties_.data_type_limits.scatter_nd_indices.Supports(
      indices_descriptor));

  // ONNX ScatterND only supports int64 indices.
  std::string int64_indices =
      indices_descriptor.data_type() == OperandDataType::kInt64
          ? indices
          : CreateCastNode(indices, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

  // Clamp the indices operand to prevent out-of-bounds writing which will cause
  // ORT CPU EP to throw a runtime error.
  std::string clamped_indices = ClampGatherNDIndices(
      input, int64_indices, input_descriptor, indices_descriptor);

  std::array<const char*, 3> inputs = {input.c_str(), clamped_indices.c_str(),
                                       updates.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeScatterND, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddSliceOperation(const mojom::Slice& slice) {
  const std::string node_name = GenerateNodeName(slice.label);
  const std::string input = GetOperandNameById(slice.input_operand_id);
  const std::string output = GetOperandNameById(slice.output_operand_id);

  CHECK(context_properties_.data_type_limits.slice_input.Supports(
      GetOperand(slice.input_operand_id).descriptor));

  base::FixedArray<int64_t> starts(slice.ranges.size());
  base::FixedArray<int64_t> ends(slice.ranges.size());
  base::FixedArray<int64_t> steps(slice.ranges.size());
  for (size_t i = 0; i < slice.ranges.size(); ++i) {
    starts[i] = base::checked_cast<int64_t>(slice.ranges[i].start);
    ends[i] = base::checked_cast<int64_t>(slice.ranges[i].start +
                                          slice.ranges[i].size);
    steps[i] = base::checked_cast<int64_t>(slice.ranges[i].stride);
  }

  // Explicitly provide axes to avoid validation failure of DirectML EP.
  // https://github.com/microsoft/onnxruntime/issues/25252
  base::FixedArray<int64_t> axes(slice.ranges.size());
  std::iota(axes.begin(), axes.end(), 0);

  AddSliceNode(node_name, input, output, axes, starts, ends, steps);
}

void GraphBuilderOrt::AddSoftmaxOperation(const mojom::Softmax& softmax) {
  const std::string node_name = GenerateNodeName(softmax.label);
  const std::string input = GetOperandNameById(softmax.input_operand_id);
  const std::string output = GetOperandNameById(softmax.output_operand_id);

  CHECK(context_properties_.data_type_limits.softmax_input.Supports(
      GetOperand(softmax.input_operand_id).descriptor));

  std::array<const char*, 1> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  std::array<ScopedOrtOpAttr, 1> attributes = {model_editor_.CreateAttribute(
      kAttrAxis, static_cast<int64_t>(softmax.axis))};

  model_editor_.AddNode(kOpTypeSoftmax, node_name, inputs, outputs, attributes);
}

void GraphBuilderOrt::AddPadOperation(const mojom::Pad& pad) {
  const std::string node_name = GenerateNodeName(pad.label);
  const std::string input = GetOperandNameById(pad.input_operand_id);
  const std::string output = GetOperandNameById(pad.output_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(pad.input_operand_id).descriptor;
  CHECK(context_properties_.data_type_limits.pad_input.Supports(
      input_descriptor));

  std::vector<const char*> inputs = {input.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  size_t paddings_size =
      pad.beginning_padding.size() + pad.ending_padding.size();
  CHECK_EQ(paddings_size, input_descriptor.Rank() * 2);
  std::vector<uint32_t> paddings_value;
  paddings_value.reserve(paddings_size);
  std::ranges::copy(pad.beginning_padding, std::back_inserter(paddings_value));
  std::ranges::copy(pad.ending_padding, std::back_inserter(paddings_value));
  const std::string paddings =
      CreateInt64InitializerForUint32Array(paddings_value);
  inputs.push_back(paddings.c_str());

  std::string mode;
  std::string constant;
  switch (pad.mode->which()) {
    case mojom::PaddingMode::Tag::kConstant: {
      mode = "constant";
      constant = CreateScalarInitializer(input_descriptor.data_type(),
                                         pad.mode->get_constant()->value);
      inputs.push_back(constant.c_str());
      break;
    }
    case mojom::PaddingMode::Tag::kEdge:
      mode = "edge";
      break;
    case mojom::PaddingMode::Tag::kReflection:
      mode = "reflect";
      break;
  }

  std::array<ScopedOrtOpAttr, 1> attributes = {
      model_editor_.CreateAttribute(kAttrMode, mode)};
  model_editor_.AddNode(kOpTypePad, node_name, inputs, outputs, attributes);
}

void GraphBuilderOrt::AddPreluOperation(const mojom::Prelu& prelu) {
  const std::string node_name = GenerateNodeName(prelu.label);
  const std::string input = GetOperandNameById(prelu.input_operand_id);
  const std::string slope = GetOperandNameById(prelu.slope_operand_id);
  const std::string output = GetOperandNameById(prelu.output_operand_id);

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  const OperandDescriptor& input_descriptor =
      GetOperand(prelu.input_operand_id).descriptor;
  CHECK(data_type_limits.prelu_input.Supports(input_descriptor));
  const OperandDescriptor& slope_descriptor =
      GetOperand(prelu.slope_operand_id).descriptor;
  CHECK(data_type_limits.prelu_input.Supports(slope_descriptor));

  // Validation guarantees slope is unidirectionally broadcastable to input,
  // which matches ONNX PRelu's broadcasting semantics. No manual expand needed.
  std::array<const char*, 2> inputs = {input.c_str(), slope.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypePRelu, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddSplitOperation(const mojom::Split& split) {
  const std::string node_name = GenerateNodeName(split.label);
  const std::string input = GetOperandNameById(split.input_operand_id);

  CHECK(context_properties_.data_type_limits.split_input.Supports(
      GetOperand(split.input_operand_id).descriptor));

  const auto output_count = split.output_operand_ids.size();

  base::FixedArray<std::string> output_names(output_count);
  base::FixedArray<const char*> outputs(output_count);
  for (size_t i = 0; i < output_count; i++) {
    output_names[i] = GetOperandNameById(split.output_operand_ids[i]);
    outputs[i] = output_names[i].c_str();
  }

  // Decide which of ONNX Split's two mutually exclusive forms to emit by
  // inspecting the outputs' size along `axis`. A dynamic output axis can only
  // arise from an equal split of a dynamic input axis: WebNN's `splits`
  // sequence is build-time constant, so explicit per-part sizes are always
  // static, whereas an equal split of a dynamic axis derives a symbolic
  // per-part size. The two cases map onto:
  //   * static axis  -> the `split` input tensor of concrete per-part sizes.
  //   * dynamic axis -> the `num_outputs` attribute, letting ORT divide the
  //     runtime axis size into `output_count` equal parts at dispatch.
  bool dynamic_split_axis = false;
  for (size_t i = 0; i < output_count; i++) {
    const std::vector<Dimension>& output_shape =
        GetOperand(split.output_operand_ids[i]).descriptor.shape();
    CHECK_LT(split.axis, output_shape.size());
    if (!std::holds_alternative<uint32_t>(output_shape[split.axis])) {
      dynamic_split_axis = true;
      break;
    }
  }

  if (dynamic_split_axis) {
    std::array<const char*, 1> inputs = {input.c_str()};
    std::array<ScopedOrtOpAttr, 2> attributes = {
        model_editor_.CreateAttribute(kAttrAxis,
                                      base::checked_cast<int64_t>(split.axis)),
        model_editor_.CreateAttribute(
            kAttrNumOutputs, base::checked_cast<int64_t>(output_count))};
    model_editor_.AddNode(kOpTypeSplit, node_name, inputs, outputs, attributes);
    return;
  }

  // 'split' is a 1-D tensor which specifies the length of each output. The sum
  // of the values must be equal to the input size along 'axis'.
  // https://onnx.ai/onnx/operators/onnx__Split.html#inputs
  base::FixedArray<uint32_t> split_sizes(output_count);
  for (size_t i = 0; i < output_count; i++) {
    const std::vector<Dimension>& output_shape =
        GetOperand(split.output_operand_ids[i]).descriptor.shape();
    split_sizes[i] = std::get<uint32_t>(output_shape[split.axis]);
  }
  const std::string split_input =
      CreateInt64InitializerForUint32Array(split_sizes);
  std::array<const char*, 2> inputs = {input.c_str(), split_input.c_str()};

  std::array<ScopedOrtOpAttr, 1> attributes = {model_editor_.CreateAttribute(
      kAttrAxis, base::checked_cast<int64_t>(split.axis))};

  model_editor_.AddNode(kOpTypeSplit, node_name, inputs, outputs, attributes);
}

void GraphBuilderOrt::AddTileOperation(const mojom::Tile& tile) {
  const std::string input = GetOperandNameById(tile.input_operand_id);
  const std::string output = GetOperandNameById(tile.output_operand_id);

  const OperandDescriptor& input_descriptor =
      GetOperand(tile.input_operand_id).descriptor;
  CHECK(context_properties_.data_type_limits.tile_input.Supports(
      input_descriptor));

  // Workaround: emulate the tile operation with identity operation for
  // unsupported scalar input.
  // TODO(crbug.com/500385615): Remove the workaround when the issue is fixed.
  if (input_descriptor.Rank() == 0) {
    EmulateWithIdentityNode(tile.label, input, output);
    return;
  }

  const std::string repeats =
      CreateInt64InitializerForUint32Array(tile.repetitions);

  std::array<const char*, 2> inputs = {input.c_str(), repeats.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  const std::string node_name = GenerateNodeName(tile.label);
  model_editor_.AddNode(kOpTypeTile, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddTransposeOperation(const mojom::Transpose& transpose) {
  const std::string node_name = GenerateNodeName(transpose.label);
  const std::string input = GetOperandNameById(transpose.input_operand_id);
  const std::string output = GetOperandNameById(transpose.output_operand_id);

  CHECK(context_properties_.data_type_limits.transpose_input.Supports(
      GetOperand(transpose.input_operand_id).descriptor));

  AddTransposeNode(node_name, input, output, transpose.permutation);
}

void GraphBuilderOrt::AddTriangularOperation(
    const mojom::Triangular& triangular) {
  const std::string node_name = GenerateNodeName(triangular.label);
  const std::string input = GetOperandNameById(triangular.input_operand_id);
  const std::string output = GetOperandNameById(triangular.output_operand_id);

  CHECK(context_properties_.data_type_limits.triangular_input.Supports(
      GetOperand(triangular.input_operand_id).descriptor));

  const std::string diagonal =
      CreateScalarInitializer(static_cast<int64_t>(triangular.diagonal));
  std::array<const char*, 2> inputs = {input.c_str(), diagonal.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  std::array<ScopedOrtOpAttr, 1> attributes = {model_editor_.CreateAttribute(
      kAttrUpper, static_cast<int64_t>(triangular.upper))};

  model_editor_.AddNode(kOpTypeTriangular, node_name, inputs, outputs,
                        attributes);
}

void GraphBuilderOrt::AddWhereOperation(const mojom::Where& where) {
  const std::string node_name = GenerateNodeName(where.label);
  std::string condition = GetOperandNameById(where.condition_operand_id);
  const std::string true_value =
      GetOperandNameById(where.true_value_operand_id);
  const std::string false_value =
      GetOperandNameById(where.false_value_operand_id);
  const std::string output = GetOperandNameById(where.output_operand_id);

  const OperandDescriptor& condition_descriptor =
      GetOperand(where.condition_operand_id).descriptor;
  const OperandDescriptor& true_value_descriptor =
      GetOperand(where.true_value_operand_id).descriptor;
  const OperandDescriptor& false_value_descriptor =
      GetOperand(where.false_value_operand_id).descriptor;

  const DataTypeLimits& data_type_limits = context_properties_.data_type_limits;
  CHECK(data_type_limits.where_condition.Supports(condition_descriptor));
  CHECK(data_type_limits.where_value.Supports(true_value_descriptor));
  CHECK(data_type_limits.where_value.Supports(false_value_descriptor));

  // ONNX where operation only supports bool condition input.
  CHECK_EQ(condition_descriptor.data_type(), OperandDataType::kUint8);
  condition = CreateCastNode(condition, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);

  std::array<const char*, 3> inputs = {condition.c_str(), true_value.c_str(),
                                       false_value.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode(kOpTypeWhere, node_name, inputs, outputs);
}

void GraphBuilderOrt::AddRangeOperation(const mojom::RangeOp& range) {
  const std::string node_name = GenerateNodeName(range.label);
  const std::string start = GetOperandNameById(range.start_operand_id);
  const std::string limit = GetOperandNameById(range.limit_operand_id);
  const std::string delta = GetOperandNameById(range.delta_operand_id);
  const std::string output = GetOperandNameById(range.output_operand_id);

  std::array<const char*, 3> inputs = {start.c_str(), limit.c_str(),
                                       delta.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};

  model_editor_.AddNode("Range", node_name, inputs, outputs);
}

void GraphBuilderOrt::AddDynamicReshapeOperation(
    const mojom::DynamicReshape& op) {
  const std::string node_name = GenerateNodeName(op.label);
  const std::string input = GetOperandNameById(op.input_operand_id);
  const std::string new_shape = GetOperandNameById(op.new_shape_operand_id);
  const std::string output = GetOperandNameById(op.output_operand_id);

  std::array<const char*, 2> inputs = {input.c_str(), new_shape.c_str()};
  std::array<const char*, 1> outputs_arr = {output.c_str()};

  model_editor_.AddNode(kOpTypeReshape, node_name, inputs, outputs_arr);
}

void GraphBuilderOrt::AddDynamicExpandOperation(
    const mojom::DynamicExpand& op) {
  const std::string node_name = GenerateNodeName(op.label);
  const std::string input = GetOperandNameById(op.input_operand_id);
  const std::string new_shape = GetOperandNameById(op.new_shape_operand_id);
  const std::string output = GetOperandNameById(op.output_operand_id);

  std::array<const char*, 2> inputs = {input.c_str(), new_shape.c_str()};
  std::array<const char*, 1> outputs_arr = {output.c_str()};

  model_editor_.AddNode(kOpTypeExpand, node_name, inputs, outputs_arr);
}

void GraphBuilderOrt::AddDynamicSliceOperation(
    const mojom::DynamicSlice& op) {
  const std::string node_name = GenerateNodeName(op.label);
  const std::string input = GetOperandNameById(op.input_operand_id);
  const std::string starts = GetOperandNameById(op.starts_operand_id);
  const std::string ends = GetOperandNameById(op.ends_operand_id);

  // ONNX Slice: inputs are data, starts, ends, [axes], [steps]
  std::vector<const char*> slice_inputs = {input.c_str(), starts.c_str(),
                                           ends.c_str()};

  std::string axes, strides;
  if (op.axes_operand_id.has_value()) {
    axes = GetOperandNameById(*op.axes_operand_id);
    slice_inputs.push_back(axes.c_str());
  }
  if (op.strides_operand_id.has_value()) {
    if (!op.axes_operand_id.has_value()) {
      // Need an empty string for axes if strides are provided but axes are not.
      slice_inputs.push_back("");
    }
    strides = GetOperandNameById(*op.strides_operand_id);
    slice_inputs.push_back(strides.c_str());
  }

  const std::string output = GetOperandNameById(op.output_operand_id);
  std::array<const char*, 1> outputs_arr = {output.c_str()};

  model_editor_.AddNode(kOpTypeSlice, node_name, slice_inputs, outputs_arr);
}

void GraphBuilderOrt::AddDynamicPadOperation(const mojom::DynamicPad& op) {
  const std::string node_name = GenerateNodeName(op.label);
  const std::string input = GetOperandNameById(op.input_operand_id);
  const std::string beginning_padding =
      GetOperandNameById(op.beginning_padding_operand_id);
  const std::string ending_padding =
      GetOperandNameById(op.ending_padding_operand_id);

  // ONNX Pad's `pads` input must be int64, but the WebNN operands may be int32
  // (preferred for backends like CoreML that lack int64). Cast each operand to
  // int64 if needed before concatenating, since ONNX Concat requires its
  // inputs to share a data type and the two operands may differ.
  std::string beginning_int64 = beginning_padding;
  if (GetOperand(op.beginning_padding_operand_id).descriptor.data_type() !=
      OperandDataType::kInt64) {
    beginning_int64 =
        CreateCastNode(beginning_padding, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  }
  std::string ending_int64 = ending_padding;
  if (GetOperand(op.ending_padding_operand_id).descriptor.data_type() !=
      OperandDataType::kInt64) {
    ending_int64 =
        CreateCastNode(ending_padding, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  }

  // WebNN provides padding as two separate 1-D tensors of length `rank`
  // (beginning and ending). ONNX Pad takes a single 1-D `pads` tensor of
  // length 2*rank laid out as [begin_0..begin_{r-1}, end_0..end_{r-1}], so
  // concatenate the two operands along axis 0.
  const std::string pads_int64 = GenerateOperandName();
  {
    std::array<const char*, 2> concat_inputs = {beginning_int64.c_str(),
                                                ending_int64.c_str()};
    std::array<const char*, 1> concat_outputs = {pads_int64.c_str()};
    std::array<ScopedOrtOpAttr, 1> concat_attrs = {
        model_editor_.CreateAttribute(kAttrAxis, static_cast<int64_t>(0))};
    const std::string concat_node_name = GenerateNodeName(base::JoinString(
        {kInserted, kOpTypeConcat, kToEmulate, node_name}, kUnderscore));
    model_editor_.AddNode(kOpTypeConcat, concat_node_name, concat_inputs,
                          concat_outputs, concat_attrs);
  }

  // ONNX Pad: inputs are data, pads, [constant_value]
  std::vector<const char*> pad_inputs = {input.c_str(), pads_int64.c_str()};

  std::string constant_value;
  if (op.constant_value_operand_id.has_value()) {
    constant_value = GetOperandNameById(*op.constant_value_operand_id);
    pad_inputs.push_back(constant_value.c_str());
  }

  const std::string output = GetOperandNameById(op.output_operand_id);
  std::array<const char*, 1> outputs_arr = {output.c_str()};

  model_editor_.AddNode(kOpTypePad, node_name, pad_inputs, outputs_arr);
}

void GraphBuilderOrt::AddDynamicSplitOperation(
    const mojom::DynamicSplit& op) {
  const std::string node_name = GenerateNodeName(op.label);
  const std::string input = GetOperandNameById(op.input_operand_id);
  const std::string splits = GetOperandNameById(op.splits_operand_id);

  std::array<const char*, 2> inputs = {input.c_str(), splits.c_str()};

  std::vector<std::string> output_names;
  output_names.reserve(op.output_operand_ids.size());
  for (const auto& output_id : op.output_operand_ids) {
    output_names.push_back(GetOperandNameById(output_id));
  }
  std::vector<const char*> outputs_arr;
  outputs_arr.reserve(output_names.size());
  for (const auto& name : output_names) {
    outputs_arr.push_back(name.c_str());
  }

  std::array<ScopedOrtOpAttr, 1> attributes = {model_editor_.CreateAttribute(
      kAttrAxis, base::checked_cast<int64_t>(op.axis))};

  model_editor_.AddNode(kOpTypeSplit, node_name, inputs, outputs_arr,
                        attributes);
}

void GraphBuilderOrt::AddDynamicResample2dOperation(
    const mojom::DynamicResample2d& op) {
  const std::string node_name = GenerateNodeName(op.label);
  const std::string input = GetOperandNameById(op.input_operand_id);
  const std::string user_sizes = GetOperandNameById(op.sizes_operand_id);
  const std::string output = GetOperandNameById(op.output_operand_id);

  // Step 1: Get the input shape at runtime, as an int64[4] tensor.
  const std::string input_shape = GenerateOperandName();
  {
    const std::string shape_node_name = GenerateNodeName(base::JoinString(
        {kInserted, kOpTypeShape, kToEmulate, node_name}, kUnderscore));
    std::array<const char*, 1> shape_inputs = {input.c_str()};
    std::array<const char*, 1> shape_outputs = {input_shape.c_str()};
    model_editor_.AddNode(kOpTypeShape, shape_node_name, shape_inputs,
                          shape_outputs);
  }

  // Step 2: Cast the user-provided sizes to int64 if needed. ONNX Resize's
  // `sizes` input must be int64.
  const OperandDataType sizes_data_type =
      GetOperand(op.sizes_operand_id).descriptor.data_type();
  std::string sizes_int64 = user_sizes;
  if (sizes_data_type != OperandDataType::kInt64) {
    sizes_int64 =
        CreateCastNode(user_sizes, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  }

  // Step 3: Scatter the 2 user-provided sizes into the 4-element input_shape
  // tensor at the positions given by `axes`, producing the 4-element `sizes`
  // tensor that ONNX Resize expects.
  //   indices = int64[axes[0], axes[1]]  (shape [2])
  //   sizes_4 = ScatterElements(data=input_shape, indices, updates=sizes_int64,
  //                             axis=0)
  const std::array<int64_t, 2> indices_values = {
      static_cast<int64_t>(op.axes[0]), static_cast<int64_t>(op.axes[1])};
  const std::string indices = Create1DInitializer<int64_t>(indices_values);

  const std::string sizes_4 = GenerateOperandName();
  {
    const std::string scatter_node_name = GenerateNodeName(base::JoinString(
        {kInserted, kOpTypeScatterElements, kToEmulate, node_name},
        kUnderscore));
    std::array<const char*, 3> scatter_inputs = {
        input_shape.c_str(), indices.c_str(), sizes_int64.c_str()};
    std::array<const char*, 1> scatter_outputs = {sizes_4.c_str()};
    std::array<ScopedOrtOpAttr, 1> scatter_attributes = {
        model_editor_.CreateAttribute(kAttrAxis, int64_t{0})};
    model_editor_.AddNode(kOpTypeScatterElements, scatter_node_name,
                          scatter_inputs, scatter_outputs, scatter_attributes);
  }

  // Step 4: Emit the Resize node.
  std::string mode;
  switch (op.mode) {
    case mojom::Resample2d::InterpolationMode::kLinear:
      mode = "linear";
      break;
    case mojom::Resample2d::InterpolationMode::kNearestNeighbor:
      mode = "nearest";
      break;
  }

  AddResizeNode(node_name, input, /*scales=*/"", sizes_4, mode, output);
}

void GraphBuilderOrt::AddDynamicTileOperation(const mojom::DynamicTile& op) {
  const std::string node_name = GenerateNodeName(op.label);
  const std::string input = GetOperandNameById(op.input_operand_id);
  const std::string repetitions = GetOperandNameById(op.repetitions_operand_id);
  const std::string output = GetOperandNameById(op.output_operand_id);

  // ONNX Tile's `repeats` input must be int64, but the WebNN operand may be
  // int32 (preferred for backends like CoreML that lack int64). Cast if needed.
  std::string repeats_int64 = repetitions;
  if (GetOperand(op.repetitions_operand_id).descriptor.data_type() !=
      OperandDataType::kInt64) {
    repeats_int64 =
        CreateCastNode(repetitions, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  }

  std::array<const char*, 2> inputs = {input.c_str(), repeats_int64.c_str()};
  std::array<const char*, 1> outputs = {output.c_str()};
  model_editor_.AddNode(kOpTypeTile, node_name, inputs, outputs);
}

// Returns the output operand IDs for `operation`.
// Expand outputs are excluded: expand derives its output shape *from*
// known_dynamic_dims_, so registering the expand output as a shape source
// would require inserting a Shape node before the expand itself, creating a
// cycle in the ORT graph.
std::vector<OperandId> GetOperationOutputOperandIds(
    const mojom::Operation& operation) {
  switch (operation.which()) {
    case mojom::Operation::Tag::kExpand:
      return {};
    // Operations with multiple output operand IDs.
    case mojom::Operation::Tag::kGru: {
      const auto& ids = operation.get_gru()->output_operand_ids;
      return {ids.begin(), ids.end()};
    }
    case mojom::Operation::Tag::kLstm: {
      const auto& ids = operation.get_lstm()->output_operand_ids;
      return {ids.begin(), ids.end()};
    }
    case mojom::Operation::Tag::kLstmCell: {
      const auto& ids = operation.get_lstm_cell()->output_operand_ids;
      return {ids.begin(), ids.end()};
    }
    case mojom::Operation::Tag::kSplit: {
      const auto& ids = operation.get_split()->output_operand_ids;
      return {ids.begin(), ids.end()};
    }
    case mojom::Operation::Tag::kDynamicSplit: {
      const auto& ids = operation.get_dynamic_split()->output_operand_ids;
      return {ids.begin(), ids.end()};
    }
    // All remaining operations have a single output_operand_id.
    case mojom::Operation::Tag::kArgMinMax:
      return {operation.get_arg_min_max()->output_operand_id};
    case mojom::Operation::Tag::kBatchNormalization:
      return {operation.get_batch_normalization()->output_operand_id};
    case mojom::Operation::Tag::kClamp:
      return {operation.get_clamp()->output_operand_id};
    case mojom::Operation::Tag::kConcat:
      return {operation.get_concat()->output_operand_id};
    case mojom::Operation::Tag::kConv2d:
      return {operation.get_conv2d()->output_operand_id};
    case mojom::Operation::Tag::kCumulativeSum:
      return {operation.get_cumulative_sum()->output_operand_id};
    case mojom::Operation::Tag::kDequantizeLinear:
      return {operation.get_dequantize_linear()->output_operand_id};
    case mojom::Operation::Tag::kElu:
      return {operation.get_elu()->output_operand_id};
    case mojom::Operation::Tag::kElementWiseBinary:
      return {operation.get_element_wise_binary()->output_operand_id};
    case mojom::Operation::Tag::kElementWiseUnary:
      return {operation.get_element_wise_unary()->output_operand_id};
    case mojom::Operation::Tag::kGather:
      return {operation.get_gather()->output_operand_id};
    case mojom::Operation::Tag::kGatherElements:
      return {operation.get_gather_elements()->output_operand_id};
    case mojom::Operation::Tag::kGatherNd:
      return {operation.get_gather_nd()->output_operand_id};
    case mojom::Operation::Tag::kGelu:
      return {operation.get_gelu()->output_operand_id};
    case mojom::Operation::Tag::kGemm:
      return {operation.get_gemm()->output_operand_id};
    case mojom::Operation::Tag::kGruCell:
      return {operation.get_gru_cell()->output_operand_id};
    case mojom::Operation::Tag::kHardSigmoid:
      return {operation.get_hard_sigmoid()->output_operand_id};
    case mojom::Operation::Tag::kHardSwish:
      return {operation.get_hard_swish()->output_operand_id};
    case mojom::Operation::Tag::kInstanceNormalization:
      return {operation.get_instance_normalization()->output_operand_id};
    case mojom::Operation::Tag::kLayerNormalization:
      return {operation.get_layer_normalization()->output_operand_id};
    case mojom::Operation::Tag::kLeakyRelu:
      return {operation.get_leaky_relu()->output_operand_id};
    case mojom::Operation::Tag::kLinear:
      return {operation.get_linear()->output_operand_id};
    case mojom::Operation::Tag::kMatmul:
      return {operation.get_matmul()->output_operand_id};
    case mojom::Operation::Tag::kPad:
      return {operation.get_pad()->output_operand_id};
    case mojom::Operation::Tag::kPool2d:
      return {operation.get_pool2d()->output_operand_id};
    case mojom::Operation::Tag::kPrelu:
      return {operation.get_prelu()->output_operand_id};
    case mojom::Operation::Tag::kQuantizeLinear:
      return {operation.get_quantize_linear()->output_operand_id};
    case mojom::Operation::Tag::kRelu:
      return {operation.get_relu()->output_operand_id};
    case mojom::Operation::Tag::kReduce:
      return {operation.get_reduce()->output_operand_id};
    case mojom::Operation::Tag::kResample2d:
      return {operation.get_resample2d()->output_operand_id};
    case mojom::Operation::Tag::kReshape:
      return {operation.get_reshape()->output_operand_id};
    case mojom::Operation::Tag::kReverse:
      return {operation.get_reverse()->output_operand_id};
    case mojom::Operation::Tag::kScatterElements:
      return {operation.get_scatter_elements()->output_operand_id};
    case mojom::Operation::Tag::kScatterNd:
      return {operation.get_scatter_nd()->output_operand_id};
    case mojom::Operation::Tag::kShape:
      return {operation.get_shape()->output_operand_id};
    case mojom::Operation::Tag::kSlice:
      return {operation.get_slice()->output_operand_id};
    case mojom::Operation::Tag::kSigmoid:
      return {operation.get_sigmoid()->output_operand_id};
    case mojom::Operation::Tag::kSoftmax:
      return {operation.get_softmax()->output_operand_id};
    case mojom::Operation::Tag::kSoftplus:
      return {operation.get_softplus()->output_operand_id};
    case mojom::Operation::Tag::kSoftsign:
      return {operation.get_softsign()->output_operand_id};
    case mojom::Operation::Tag::kTanh:
      return {operation.get_tanh()->output_operand_id};
    case mojom::Operation::Tag::kTile:
      return {operation.get_tile()->output_operand_id};
    case mojom::Operation::Tag::kTranspose:
      return {operation.get_transpose()->output_operand_id};
    case mojom::Operation::Tag::kTriangular:
      return {operation.get_triangular()->output_operand_id};
    case mojom::Operation::Tag::kWhere:
      return {operation.get_where()->output_operand_id};
    case mojom::Operation::Tag::kRange:
      return {operation.get_range()->output_operand_id};
    case mojom::Operation::Tag::kDynamicReshape:
      return {operation.get_dynamic_reshape()->output_operand_id};
    case mojom::Operation::Tag::kDynamicExpand:
      return {operation.get_dynamic_expand()->output_operand_id};
    case mojom::Operation::Tag::kDynamicSlice:
      return {operation.get_dynamic_slice()->output_operand_id};
    case mojom::Operation::Tag::kDynamicPad:
      return {operation.get_dynamic_pad()->output_operand_id};
    case mojom::Operation::Tag::kDynamicResample2d:
      return {operation.get_dynamic_resample_2d()->output_operand_id};
    case mojom::Operation::Tag::kDynamicTile:
      return {operation.get_dynamic_tile()->output_operand_id};
  }
}

void GraphBuilderOrt::RegisterOperandDynamicDims(OperandId operand_id) {
  const mojom::Operand& operand = GetOperand(operand_id);
  const std::string operand_name = GetOperandNameById(operand_id);
  const std::vector<Dimension>& shape = operand.descriptor.shape();
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (std::holds_alternative<DynamicDimension>(shape[axis])) {
      const DynamicDimension& dynamic_dim =
          std::get<DynamicDimension>(shape[axis]);
      // Only named dynamic dims are trackable by name. First occurrence wins;
      // input operands (registered in the constructor) are always preferred as
      // sources.
      if (dynamic_dim.name.has_value() &&
          !known_dynamic_dims_.contains(*dynamic_dim.name)) {
        known_dynamic_dims_[*dynamic_dim.name] =
            DynamicDimensionInfo(operand_name, static_cast<uint32_t>(axis));
      }
    }
  }
}

base::expected<std::unique_ptr<ModelEditor::ModelInfo>, mojom::ErrorPtr>
GraphBuilderOrt::BuildModel() {
  for (OperandId input_id : graph_info_->input_operands) {
    model_editor_.AddInput(GetOperandNameById(input_id), GetOperand(input_id));
  }

  for (auto& [constant_id, constant_operand] : constant_operands_) {
    model_editor_.AddInitializer(GetOperandNameById(constant_id),
                                 std::move(constant_operand));
  }
  constant_operands_.clear();

  // Pre-pass: register dynamic dimensions produced by operations so that ops
  // processed in topological order can always resolve a dim they only
  // reference via shape metadata (e.g. Expand).
  //
  // Done in two phases:
  //
  //  Phase 1 — Reshape-inferred dims. For every Reshape that introduces a
  //  brand-new dynamic dim via ONNX "-1", pin the dim to that Reshape's
  //  own output operand. The Reshape's output is the semantic definition
  //  of the dim, so it is the natural source for downstream Shape(...)
  //  emulation, and its tensor-level dependencies can never form a cycle
  //  through an op that references the dim only via shape metadata.
  //
  //  Phase 2 — naive registration of all other output dynamic dims (first
  //  occurrence wins). Expand outputs are excluded (see
  //  GetOperationOutputOperandIds) to prevent Shape-before-Expand cycles.
  //
  // ONNX Runtime's ModelEditor API is insertion-order independent: nodes
  // can reference operand names produced by nodes added later, as long as
  // the full graph is valid at session creation time.
  for (const mojom::OperationPtr& operation : graph_info_->operations) {
    if (!operation->is_reshape()) {
      continue;
    }
    const auto& reshape = *operation->get_reshape();
    const std::vector<Dimension>& in_shape =
        GetOperand(reshape.input_operand_id).descriptor.shape();
    const std::vector<Dimension>& out_shape =
        GetOperand(reshape.output_operand_id).descriptor.shape();

    for (size_t i = 0; i < out_shape.size(); ++i) {
      if (!std::holds_alternative<DynamicDimension>(out_shape[i])) {
        continue;
      }
      const auto& inferred = std::get<DynamicDimension>(out_shape[i]);
      // Only named dynamic dims are trackable by name.
      if (!inferred.name.has_value() ||
          known_dynamic_dims_.contains(*inferred.name)) {
        continue;
      }
      if (std::ranges::find(in_shape, out_shape[i]) != in_shape.end()) {
        continue;  // dim comes from input shape directly; default path works
      }

      // Brand-new dynamic dim introduced by this Reshape's "-1". The
      // Reshape's own output operand is always a safe & canonical source:
      // its tensor-level dependencies are the Reshape's inputs (and their
      // transitive producers), which — absent an ML-level semantic cycle
      // — do not depend on any downstream op that references this dim
      // only via shape metadata (e.g. Expand). Registering here, before
      // the naive pass below, ensures that operand wins over any later
      // op whose output happens to carry the same dim at a lower mojom
      // index (e.g. a CumSum sitting between the Expand producer and
      // this Reshape in topological order).
      known_dynamic_dims_[*inferred.name] =
          DynamicDimensionInfo(GetOperandNameById(reshape.output_operand_id),
                               static_cast<uint32_t>(i));
    }
  }

  for (const mojom::OperationPtr& operation : graph_info_->operations) {
    for (OperandId output_id : GetOperationOutputOperandIds(*operation)) {
      RegisterOperandDynamicDims(output_id);
    }
  }

  for (const mojom::OperationPtr& operation : graph_info_->operations) {
    const DataTypeLimits& data_type_limits =
        context_properties_.data_type_limits;
    switch (operation->which()) {
      case mojom::Operation::Tag::kArgMinMax: {
        AddArgMinMaxOperation(*operation->get_arg_min_max());
        break;
      }
      case mojom::Operation::Tag::kBatchNormalization: {
        AddBatchNormalizationOperation(*operation->get_batch_normalization());
        break;
      }
      case mojom::Operation::Tag::kClamp: {
        AddClampOperation(*operation->get_clamp());
        break;
      }
      case mojom::Operation::Tag::kConcat: {
        AddConcatOperation(*operation->get_concat());
        break;
      }
      case mojom::Operation::Tag::kConv2d: {
        AddConv2dOperation(*operation->get_conv2d());
        break;
      }
      case mojom::Operation::Tag::kCumulativeSum: {
        AddCumulativeSumOperation(*operation->get_cumulative_sum());
        break;
      }
      case mojom::Operation::Tag::kDequantizeLinear: {
        CHECK(data_type_limits.dequantize_linear_input.SupportsAll(
            {GetOperand(operation->get_dequantize_linear()->input_operand_id)
                 .descriptor,
             GetOperand(
                 operation->get_dequantize_linear()->zero_point_operand_id)
                 .descriptor}));
        CHECK(data_type_limits.dequantize_linear_scale.Supports(
            GetOperand(operation->get_dequantize_linear()->scale_operand_id)
                .descriptor));
        AddDequantizeOrQuantizeLinearOperation(
            *operation->get_dequantize_linear(), kOpTypeDequantizeLinear);
        break;
      }
      case mojom::Operation::Tag::kElu: {
        AddEluOperation(*operation->get_elu());
        break;
      }
      case mojom::Operation::Tag::kElementWiseBinary: {
        AddElementWiseBinaryOperation(*operation->get_element_wise_binary());
        break;
      }
      case mojom::Operation::Tag::kElementWiseUnary: {
        AddElementWiseUnaryOperation(*operation->get_element_wise_unary());
        break;
      }
      case mojom::Operation::Tag::kExpand: {
        AddExpandOperation(*operation->get_expand());
        break;
      }
      case mojom::Operation::Tag::kGather: {
        CHECK(data_type_limits.gather_input.Supports(
            GetOperand(operation->get_gather()->input_operand_id).descriptor));
        CHECK(data_type_limits.gather_indices.Supports(
            GetOperand(operation->get_gather()->indices_operand_id)
                .descriptor));
        AddGatherOperation(*operation->get_gather(), kOpTypeGather);
        break;
      }
      case mojom::Operation::Tag::kGatherElements: {
        CHECK(data_type_limits.gather_elements_input.Supports(
            GetOperand(operation->get_gather_elements()->input_operand_id)
                .descriptor));
        CHECK(data_type_limits.gather_elements_indices.Supports(
            GetOperand(operation->get_gather_elements()->indices_operand_id)
                .descriptor));
        AddGatherOperation(*operation->get_gather_elements(),
                           kOpTypeGatherElements);
        break;
      }
      case mojom::Operation::Tag::kGatherNd: {
        AddGatherNDOperation(*operation->get_gather_nd());
        break;
      }
      case mojom::Operation::Tag::kGelu: {
        CHECK(data_type_limits.gelu_input.Supports(
            GetOperand(operation->get_gelu()->input_operand_id).descriptor));
        AddUnaryOperation(*operation->get_gelu(), kOpTypeGelu);
        break;
      }
      case mojom::Operation::Tag::kGemm: {
        AddGemmOperation(*operation->get_gemm());
        break;
      }
      case mojom::Operation::Tag::kGru: {
        AddGruOperation(*operation->get_gru());
        break;
      }
      case mojom::Operation::Tag::kGruCell: {
        AddGruOperation(*operation->get_gru_cell());
        break;
      }
      case mojom::Operation::Tag::kHardSigmoid: {
        AddHardSigmoidOperation(*operation->get_hard_sigmoid());
        break;
      }
      case mojom::Operation::Tag::kHardSwish: {
        CHECK(data_type_limits.hard_swish_input.Supports(
            GetOperand(operation->get_hard_swish()->input_operand_id)
                .descriptor));
        AddUnaryOperation(*operation->get_hard_swish(), kOpTypeHardSwish);
        break;
      }
      case mojom::Operation::Tag::kInstanceNormalization: {
        AddInstanceNormalizationOperation(
            *operation->get_instance_normalization());
        break;
      }
      case mojom::Operation::Tag::kLayerNormalization: {
        AddLayerNormalizationOperation(*operation->get_layer_normalization());
        break;
      }
      case mojom::Operation::Tag::kLeakyRelu: {
        AddLeakyReluOperation(*operation->get_leaky_relu());
        break;
      }
      case mojom::Operation::Tag::kLinear: {
        AddLinearOperation(*operation->get_linear());
        break;
      }
      case mojom::Operation::Tag::kLstm: {
        AddLstmOperation(*operation->get_lstm());
        break;
      }
      case mojom::Operation::Tag::kLstmCell: {
        AddLstmOperation(*operation->get_lstm_cell());
        break;
      }
      case mojom::Operation::Tag::kMatmul: {
        auto result = AddMatMulOperation(*operation->get_matmul());
        if (!result.has_value()) {
          return base::unexpected(std::move(result.error()));
        }
        break;
      }
      case mojom::Operation::Tag::kPad: {
        AddPadOperation(*operation->get_pad());
        break;
      }
      case mojom::Operation::Tag::kPool2d: {
        AddPool2dOperation(*operation->get_pool2d());
        break;
      }
      case mojom::Operation::Tag::kPrelu: {
        AddPreluOperation(*operation->get_prelu());
        break;
      }
      case mojom::Operation::Tag::kQuantizeLinear: {
        CHECK(data_type_limits.quantize_linear_input.SupportsAll(
            {GetOperand(operation->get_quantize_linear()->input_operand_id)
                 .descriptor,
             GetOperand(operation->get_quantize_linear()->scale_operand_id)
                 .descriptor}));
        CHECK(data_type_limits.quantize_linear_zero_point.Supports(
            GetOperand(operation->get_quantize_linear()->zero_point_operand_id)
                .descriptor));
        AddDequantizeOrQuantizeLinearOperation(
            *operation->get_quantize_linear(), kOpTypeQuantizeLinear);
        break;
      }
      case mojom::Operation::Tag::kRelu: {
        CHECK(data_type_limits.relu_input.Supports(
            GetOperand(operation->get_relu()->input_operand_id).descriptor));
        AddUnaryOperation(*operation->get_relu(), kOpTypeRelu);
        break;
      }
      case mojom::Operation::Tag::kReduce: {
        AddReduceOperation(*operation->get_reduce());
        break;
      }
      case mojom::Operation::Tag::kResample2d: {
        AddResample2dOperation(*operation->get_resample2d());
        break;
      }
      case mojom::Operation::Tag::kReshape: {
        AddReshapeOperation(*operation->get_reshape());
        break;
      }
      case mojom::Operation::Tag::kReverse: {
        AddReverseOperation(*operation->get_reverse());
        break;
      }
      case mojom::Operation::Tag::kShape: {
        AddShapeOperation(*operation->get_shape());
        break;
      }
      case mojom::Operation::Tag::kScatterElements: {
        AddScatterElementsOperation(*operation->get_scatter_elements());
        break;
      }
      case mojom::Operation::Tag::kScatterNd: {
        AddScatterNDOperation(*operation->get_scatter_nd());
        break;
      }
      case mojom::Operation::Tag::kSlice: {
        AddSliceOperation(*operation->get_slice());
        break;
      }
      case mojom::Operation::Tag::kSigmoid: {
        CHECK(data_type_limits.sigmoid_input.Supports(
            GetOperand(operation->get_sigmoid()->input_operand_id).descriptor));
        AddUnaryOperation(*operation->get_sigmoid(), kOpTypeSigmoid);
        break;
      }
      case mojom::Operation::Tag::kSoftmax: {
        AddSoftmaxOperation(*operation->get_softmax());
        break;
      }
      case mojom::Operation::Tag::kSoftplus: {
        CHECK(data_type_limits.softplus_input.Supports(
            GetOperand(operation->get_softplus()->input_operand_id)
                .descriptor));
        AddUnaryOperation(*operation->get_softplus(), kOpTypeSoftplus);
        break;
      }
      case mojom::Operation::Tag::kSoftsign: {
        CHECK(data_type_limits.softsign_input.Supports(
            GetOperand(operation->get_softsign()->input_operand_id)
                .descriptor));
        AddUnaryOperation(*operation->get_softsign(), kOpTypeSoftsign);
        break;
      }
      case mojom::Operation::Tag::kSplit: {
        AddSplitOperation(*operation->get_split());
        break;
      }
      case mojom::Operation::Tag::kTanh: {
        CHECK(data_type_limits.tanh_input.Supports(
            GetOperand(operation->get_tanh()->input_operand_id).descriptor));
        AddUnaryOperation(*operation->get_tanh(), kOpTypeTanh);
        break;
      }
      case mojom::Operation::Tag::kTile: {
        AddTileOperation(*operation->get_tile());
        break;
      }
      case mojom::Operation::Tag::kTranspose: {
        AddTransposeOperation(*operation->get_transpose());
        break;
      }
      case mojom::Operation::Tag::kTriangular: {
        AddTriangularOperation(*operation->get_triangular());
        break;
      }
      case mojom::Operation::Tag::kWhere: {
        AddWhereOperation(*operation->get_where());
        break;
      }
      case mojom::Operation::Tag::kRange: {
        AddRangeOperation(*operation->get_range());
        break;
      }
      case mojom::Operation::Tag::kDynamicReshape: {
        AddDynamicReshapeOperation(*operation->get_dynamic_reshape());
        break;
      }
      case mojom::Operation::Tag::kDynamicExpand: {
        AddDynamicExpandOperation(*operation->get_dynamic_expand());
        break;
      }
      case mojom::Operation::Tag::kDynamicSlice: {
        AddDynamicSliceOperation(*operation->get_dynamic_slice());
        break;
      }
      case mojom::Operation::Tag::kDynamicPad: {
        AddDynamicPadOperation(*operation->get_dynamic_pad());
        break;
      }
      case mojom::Operation::Tag::kDynamicSplit: {
        AddDynamicSplitOperation(*operation->get_dynamic_split());
        break;
      }
      case mojom::Operation::Tag::kDynamicResample2d: {
        AddDynamicResample2dOperation(
            *operation->get_dynamic_resample_2d());
        break;
      }
      case mojom::Operation::Tag::kDynamicTile: {
        AddDynamicTileOperation(*operation->get_dynamic_tile());
        break;
      }
    }
    // Register any dynamic dimensions introduced by this operation's output
    // operands so that subsequent operations (e.g. expand) can use them as
    // Shape sources. Expand outputs are excluded by
    // GetOperationOutputOperandIds to prevent Shape-before-Expand cycles in the
    // ORT graph.
    for (OperandId output_id : GetOperationOutputOperandIds(*operation)) {
      RegisterOperandDynamicDims(output_id);
    }
  }

  for (OperandId output_id : graph_info_->output_operands) {
    model_editor_.AddOutput(GetOperandNameById(output_id),
                            GetOperand(output_id));
  }

  return model_editor_.BuildAndTakeModelInfo();
}

}  // namespace webnn::ort
