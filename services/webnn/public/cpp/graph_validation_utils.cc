// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/public/cpp/graph_validation_utils.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <set>
#include <variant>
#include <vector>

#include "base/check.h"
#include "base/check_op.h"
#include "base/notreached.h"
#include "base/numerics/checked_math.h"
#include "base/numerics/safe_conversions.h"
#include "base/strings/strcat.h"
#include "base/strings/string_util.h"
#include "base/strings/stringprintf.h"
#include "base/types/expected.h"
#include "base/types/expected_macros.h"
#include "base/types/fixed_array.h"
#include "services/webnn/public/cpp/context_properties.h"
#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/public/cpp/supported_data_types.h"
#include "services/webnn/public/cpp/supported_tensors.h"
#include "services/webnn/public/cpp/webnn_errors.h"

namespace webnn {

namespace {

// Used by concat/split to validate operand count limit.
constexpr uint32_t kMaxValidTensorCount = 8192;

// The error message labels for corresponding operands.
static constexpr char kBiasParam[] = "bias";
static constexpr char kCellStateParam[] = "cellState";
static constexpr char kConditionParam[] = "condition";
static constexpr char kFalseValueParam[] = "falseValue";
static constexpr char kFilterParam[] = "filter";
static constexpr char kGemmAParam[] = "gemmA";
static constexpr char kGemmBParam[] = "gemmB";
static constexpr char kGemmCParam[] = "gemmC";
static constexpr char kHiddenStateParam[] = "hiddenState";
static constexpr char kIndicesParam[] = "indices";
static constexpr char kInitialCellStateParam[] = "initialCellState";
static constexpr char kInitialHiddenStateParam[] = "initialHiddenState";
static constexpr char kMeanParam[] = "mean";
static constexpr char kPeepholeWeightParam[] = "peepholeWeight";
static constexpr char kRecurrentBiasParam[] = "recurrentBias";
static constexpr char kRecurrentWeightParam[] = "recurrentWeight";
static constexpr char kScaleParam[] = "scale";
static constexpr char kSlopeParam[] = "slope";
static constexpr char kTrueValueParam[] = "trueValue";
static constexpr char kUpdatesParam[] = "updates";
static constexpr char kVarianceParam[] = "variance";
static constexpr char kWeightParam[] = "weight";
static constexpr char kZeroPointParam[] = "zeroPoint";

// Channels may be dynamic; divisibility/equality checks against the filter
// are skipped at build time in that case and enforced at dispatch time.
struct Conv2dInputOutputInfo {
  Dimension batches;
  Dimension channels;
  Dimension height;
  Dimension width;
};

// Validate that the intermediate padded shape is within the limits of
// OperandDescriptor. This is useful for convolution and pooling operations
// that may be implemented by padding the input tensor first.
base::expected<void, std::string> ValidateIntermediatePaddedDescriptor(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const Padding2d& padding,
    const InputOperandLayout& input_layout,
    const Conv2dInputOutputInfo& input_info,
    std::string_view label) {
  // If any dimension is dynamic, defer this validation to dispatch time when
  // concrete values are known.
  if (!std::holds_alternative<uint32_t>(input_info.batches) ||
      !std::holds_alternative<uint32_t>(input_info.channels) ||
      !std::holds_alternative<uint32_t>(input_info.height) ||
      !std::holds_alternative<uint32_t>(input_info.width)) {
    return base::ok();
  }
  const uint32_t batches = std::get<uint32_t>(input_info.batches);
  const uint32_t channels = std::get<uint32_t>(input_info.channels);
  const uint32_t height = std::get<uint32_t>(input_info.height);
  const uint32_t width = std::get<uint32_t>(input_info.width);
  uint32_t padded_height;
  uint32_t padded_width;
  if (!(base::CheckedNumeric<uint32_t>(height) +
        padding.beginning.height + padding.ending.height)
           .AssignIfValid(&padded_height) ||
      !(base::CheckedNumeric<uint32_t>(width) +
        padding.beginning.width + padding.ending.width)
           .AssignIfValid(&padded_width)) {
    return base::unexpected(
        ErrorWithLabel(label, "The padded intermediate shape is too large."));
  }

  std::array<uint32_t, 4> padded_shape;
  switch (input_layout) {
    case InputOperandLayout::kNchw:
      padded_shape = {batches, channels, padded_height, padded_width};
      break;
    case InputOperandLayout::kNhwc:
      padded_shape = {batches, padded_height, padded_width, channels};
      break;
  }

  auto padded_descriptor = OperandDescriptor::Create(
      context_properties, input.data_type(), padded_shape, label);
  if (!padded_descriptor.has_value()) {
    return base::unexpected(ErrorWithLabel(
        label, base::StrCat({"The padded intermediate operand is invalid: ",
                             padded_descriptor.error()})));
  }

  return base::ok();
}

// Validate and calculate the output spatial dimensions of convTranspose2d given
// input sizes, filter sizes, padding, strides, dilations and output padding.
base::expected<Size2d<uint32_t>, std::string>
ValidateAndCalculateConvTranspose2dOutputSizes(
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t filter_height,
    const uint32_t filter_width,
    const Padding2d& padding,
    const Size2d<uint32_t>& strides,
    const Size2d<uint32_t>& dilations,
    const Size2d<uint32_t>& output_padding,
    std::string_view label) {
  if (strides.height == 0 || strides.width == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "All strides should be greater than 0."));
  }
  if (dilations.height == 0 || dilations.width == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "All dilations should be greater than 0."));
  }
  if (output_padding.height >= strides.height ||
      output_padding.width >= strides.width) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The output padding must be smaller than the stride along the same "
        "dimension."));
  }

  const auto output_height = CalculateConvTranspose2dOutputSize(
      input_height, filter_height, padding.beginning.height,
      padding.ending.height, strides.height, dilations.height,
      output_padding.height);
  if (!output_height.has_value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        "Failed to calculate the output height: " + output_height.error()));
  }

  const auto output_width = CalculateConvTranspose2dOutputSize(
      input_width, filter_width, padding.beginning.width, padding.ending.width,
      strides.width, dilations.width, output_padding.width);
  if (!output_width.has_value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        "Failed to calculate the output width: " + output_width.error()));
  }

  return Size2d<uint32_t>{.height = output_height.value(),
                          .width = output_width.value()};
}

// Get the input info of 2-D direct and transposed convolution
// operation given input operand and attributes.
base::expected<Conv2dInputOutputInfo, std::string> GetConv2dInputInfo(
    const std::string& label,
    const OperandDescriptor& input,
    const Conv2dAttributesBase& attributes) {
  const std::vector<Dimension>& input_shape = input.shape();
  // The input layout option specifies the layout format of the input tensor.
  Dimension batches, channels, height, width;
  switch (attributes.input_layout) {
    case InputOperandLayout::kNchw:
      // "nchw": [batches, input_channels, height, width]
      batches = input_shape[0];
      channels = input_shape[1];
      height = input_shape[2];
      width = input_shape[3];
      break;
    case InputOperandLayout::kNhwc:
      // "nhwc": [batches, height, width, input_channels]
      batches = input_shape[0];
      height = input_shape[1];
      width = input_shape[2];
      channels = input_shape[3];
      break;
  }

  return Conv2dInputOutputInfo{.batches = batches,
                               .channels = channels,
                               .height = height,
                               .width = width};
}

// Validate the bias of 2-D direct and transposed convolution operation and
// create output operand given input operand, attributes and output info.
base::expected<OperandDescriptor, std::string>
ValidateConv2dBiasAndCreateOutputOperand(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const Conv2dAttributesBase& attributes,
    const Conv2dInputOutputInfo& output_info) {
  const std::string& label = attributes.label;
  // Validate bias operand if it is present.
  if (attributes.bias_operand) {
    CHECK_EQ(attributes.bias_operand->shape().size(), 1u);
    auto bias_shape = attributes.bias_operand->StaticShape();
    if (!bias_shape.has_value()) {
      return base::unexpected(
          ErrorWithLabel(label, "Dynamic bias shape is not supported."));
    }
    // output_info.channels is set by the caller from the filter's static
    // output_channels, so it is always a uint32_t here.
    CHECK(std::holds_alternative<uint32_t>(output_info.channels));
    const uint32_t output_channels = std::get<uint32_t>(output_info.channels);
    if ((*bias_shape)[0] != output_channels) {
      return base::unexpected(ErrorWithLabel(
          label, base::StringPrintf("The bias shape should be [%u].",
                                    output_channels)));
    }
    if (attributes.bias_operand->data_type() != input.data_type()) {
      return base::unexpected(ErrorWithLabel(
          label, "The bias data type doesn't match input data type."));
    }
  }

  // The input layout option specifies the layout format of the output tensor.
  std::vector<Dimension> output_shape;
  switch (attributes.input_layout) {
    case InputOperandLayout::kNchw:
      // "nchw": [batches, output_channels, height, width]
      output_shape = {output_info.batches, output_info.channels,
                      output_info.height, output_info.width};
      break;
    case InputOperandLayout::kNhwc:
      // "nhwc": [batches, height, width, output_channels]
      output_shape = {output_info.batches, output_info.height,
                      output_info.width, output_info.channels};
      break;
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

// Validate the axes and infer output for reduce operations.
base::expected<std::vector<Dimension>, std::string>
ValidateReduceAxesAndInferOutput(base::span<const Dimension> input_dimensions,
                                 base::span<const uint32_t> axes,
                                 bool keep_dimensions,
                                 std::string_view label) {
  auto input_rank = static_cast<uint32_t>(input_dimensions.size());
  RETURN_IF_ERROR(ValidateAxes(axes, input_rank, label));

  std::vector<Dimension> output_shape;
  if (keep_dimensions) {
    output_shape.assign(input_dimensions.begin(), input_dimensions.end());
    for (auto axis : axes) {
      output_shape[axis] = uint32_t{1};
    }
  } else {
    for (size_t i = 0; i < input_rank; i++) {
      if (!std::ranges::contains(axes, i)) {
        output_shape.push_back(input_dimensions[i]);
      }
    }
  }
  return output_shape;
}

// Validate the operand of recurrent network.
base::expected<void, std::string> ValidateRecurrentNetworkOperand(
    const OperandDescriptor& operand,
    const char* operand_name,
    base::span<const uint32_t> expected_shape,
    OperandDataType input_data_type,
    std::string_view label) {
  const auto shape = operand.StaticShape();
  if (!shape.has_value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf("For %s: dynamic shape is not supported for input.",
                           operand_name)));
  }
  if (!std::ranges::equal(*shape, expected_shape)) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf("The %s operand shape is invalid.", operand_name)));
  }
  if (operand.data_type() != input_data_type) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf(
            "The %s operand data type doesn't match the input data type.",
            operand_name)));
  }
  return base::ok();
}

// This helper method is intended to validate mean, variance, scale and bias
// operands of batchNormalization and instanceNormalization against the input
// operand. These operands share the same constraint.
base::expected<void, std::string>
ValidateNormalizationOperandIsCompatibleWithInput(
    const OperandDescriptor& operand,
    const OperandDataType input_data_type,
    Dimension input_size_on_axis,
    std::string_view label,
    std::string_view argument_name) {
  if (operand.data_type() != input_data_type) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StrCat(
            {"For ", argument_name,
             " operand: the data type doesn't match the input data type."})));
  }

  // An unranked operand (rank only known at dispatch) has no shape() to check;
  // skip this size-compatibility check and defer it to dispatch, when the
  // operand is concrete. The operand's role is otherwise unaffected.
  if (!operand.HasRank()) {
    return base::ok();
  }

  if (DimensionsAreDefinitelyUnequal(operand.shape()[0], input_size_on_axis)) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StrCat({"For ", argument_name,
                      " operand: the size of operand must be equal to the size "
                      "of the feature dimension of the input."})));
  }

  return base::ok();
}

// Error returned by operators that cannot validate or infer an output from an
// unranked (dynamic-rank) input because their semantics are defined in terms of
// a concrete rank: fixed-rank ops (conv2d/pool2d/gemm and the recurrent
// networks expect a specific rank) and resample2d. Operators whose output rank
// follows the input instead propagate the unranked state and defer the
// rank-dependent checks to computeShapes/dispatch. A renderer is untrusted, so
// returning an error here (rather than dereferencing an absent rank) keeps the
// service robust against a malformed "unranked where a rank is required"
// message.
constexpr char kUnrankedInputNotSupportedError[] =
    "The operator does not support an input operand with unknown rank.";

}  // namespace

bool DimensionsAreDefinitelyUnequal(const Dimension& a, const Dimension& b) {
  // Only a pair of static dimensions carrying different uint32_t values
  // constitutes a proof of inequality at build time. Any dynamic side makes
  // the relationship unknown until dispatch.
  const uint32_t* a_static = std::get_if<uint32_t>(&a);
  const uint32_t* b_static = std::get_if<uint32_t>(&b);
  return a_static && b_static && *a_static != *b_static;
}

std::optional<uint64_t> TryGetStaticProduct(base::span<const Dimension> dims) {
  base::CheckedNumeric<uint64_t> product = 1u;
  for (const Dimension& dim : dims) {
    const uint32_t* size = std::get_if<uint32_t>(&dim);
    if (!size) {
      // A dynamic dimension makes the product unknown at build time.
      return std::nullopt;
    }
    product *= *size;
  }
  uint64_t result;
  return product.AssignIfValid(&result) ? std::make_optional(result)
                                        : std::nullopt;
}

base::expected<void, std::string> ValidateReshapeDynamicDimsCompatible(
    base::span<const Dimension> input_shape,
    base::span<const Dimension> output_shape,
    std::string_view label) {
  // Cancel NAMED dynamic dimensions that appear identically on both sides: a
  // shared name denotes the same runtime value, so it contributes the same
  // factor to both element counts and can be removed from each before
  // comparing. Unnamed dynamic dims are independent unknowns and must not be
  // cancelled (even though two of them compare structurally equal).
  std::vector<Dimension> remaining_input(input_shape.begin(),
                                         input_shape.end());
  std::vector<Dimension> remaining_output(output_shape.begin(),
                                          output_shape.end());
  for (auto out_it = remaining_output.begin();
       out_it != remaining_output.end();) {
    const auto* out_dyn = std::get_if<DynamicDimension>(&*out_it);
    if (out_dyn && out_dyn->name.has_value()) {
      auto in_it =
          std::find(remaining_input.begin(), remaining_input.end(), *out_it);
      if (in_it != remaining_input.end()) {
        remaining_input.erase(in_it);
        out_it = remaining_output.erase(out_it);
        continue;
      }
    }
    ++out_it;
  }

  // The reshape is only refutable at build time when both remainders are fully
  // static and their products disagree. Any surviving dynamic dimension makes
  // the element count unknown, so the check is deferred to dispatch.
  std::optional<uint64_t> input_elements = TryGetStaticProduct(remaining_input);
  std::optional<uint64_t> output_elements =
      TryGetStaticProduct(remaining_output);
  if (input_elements.has_value() && output_elements.has_value() &&
      *input_elements != *output_elements) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The number of elements implied by the new shape doesn't match the "
        "number of elements in the input tensor."));
  }
  return base::ok();
}

base::expected<double, std::string> CalculateConv2dOutputSize(
    uint32_t input_size,
    uint32_t filter_size,
    uint32_t beginning_padding,
    uint32_t ending_padding,
    uint32_t stride,
    uint32_t dilation,
    std::string_view label) {
  // Calculate the dilated filter sizes.
  auto checked_effective_filter_size =
      (base::CheckedNumeric<uint32_t>(filter_size) - 1) * dilation + 1;
  if (!checked_effective_filter_size.IsValid()) {
    return base::unexpected(
        ErrorWithLabel(label, "The effective filter size is too large."));
  }

  // Calculate the output size in double precision floating point number that
  // ensures all dimension values of type uint32_t can be exactly represented.
  // https://en.wikipedia.org/wiki/Double-precision_floating-point_format#Precision_limitations_on_integer_values
  // The max value of checked_output_size should be 3 * UINT_MAX + 1,
  // which is smaller than the max safe integer value for double type.
  auto checked_output_size =
      (base::CheckedNumeric<double>(input_size) -
       checked_effective_filter_size + beginning_padding + ending_padding) /
          stride +
      1;

  if (checked_output_size.ValueOrDie() <= 0) {
    return base::unexpected(ErrorWithLabel(
        label, "The input size is too small to fill the window."));
  }

  // Check if the value is valid for rounding to uint32_t type.
  if (!checked_output_size.IsValid<uint32_t>()) {
    return base::unexpected(
        ErrorWithLabel(label, "The output size is too large."));
  }

  return checked_output_size.ValueOrDie();
}

BatchNormalizationAttributes::BatchNormalizationAttributes() = default;
BatchNormalizationAttributes::~BatchNormalizationAttributes() = default;
BatchNormalizationAttributes::BatchNormalizationAttributes(
    BatchNormalizationAttributes&& other) = default;
BatchNormalizationAttributes& BatchNormalizationAttributes::operator=(
    BatchNormalizationAttributes&& other) = default;

Conv2dAttributesBase::Conv2dAttributesBase() = default;
Conv2dAttributesBase::~Conv2dAttributesBase() = default;
Conv2dAttributesBase::Conv2dAttributesBase(Conv2dAttributesBase&& other) =
    default;
Conv2dAttributesBase& Conv2dAttributesBase::operator=(
    Conv2dAttributesBase&& other) = default;

Conv2dAttributes::Conv2dAttributes() = default;
Conv2dAttributes::~Conv2dAttributes() = default;
Conv2dAttributes::Conv2dAttributes(Conv2dAttributes&& other) = default;
Conv2dAttributes& Conv2dAttributes::operator=(Conv2dAttributes&& other) =
    default;

ConvTranspose2dAttributes::ConvTranspose2dAttributes() = default;
ConvTranspose2dAttributes::~ConvTranspose2dAttributes() = default;
ConvTranspose2dAttributes::ConvTranspose2dAttributes(
    ConvTranspose2dAttributes&& other) = default;
ConvTranspose2dAttributes& ConvTranspose2dAttributes::operator=(
    ConvTranspose2dAttributes&& other) = default;

GemmAttributes::GemmAttributes() = default;
GemmAttributes::~GemmAttributes() = default;
GemmAttributes::GemmAttributes(GemmAttributes&& other) = default;
GemmAttributes& GemmAttributes::operator=(GemmAttributes&& other) = default;

GruAttributes::GruAttributes() = default;
GruAttributes::~GruAttributes() = default;
GruAttributes::GruAttributes(GruAttributes&& other) = default;
GruAttributes& GruAttributes::operator=(GruAttributes&& other) = default;

GruCellAttributes::GruCellAttributes() = default;
GruCellAttributes::~GruCellAttributes() = default;
GruCellAttributes::GruCellAttributes(GruCellAttributes&& other) = default;
GruCellAttributes& GruCellAttributes::operator=(GruCellAttributes&& other) =
    default;

InstanceNormalizationAttributes::InstanceNormalizationAttributes() = default;
InstanceNormalizationAttributes::~InstanceNormalizationAttributes() = default;
InstanceNormalizationAttributes::InstanceNormalizationAttributes(
    InstanceNormalizationAttributes&& other) = default;
InstanceNormalizationAttributes& InstanceNormalizationAttributes::operator=(
    InstanceNormalizationAttributes&& other) = default;

LayerNormalizationAttributes::LayerNormalizationAttributes() = default;
LayerNormalizationAttributes::~LayerNormalizationAttributes() = default;
LayerNormalizationAttributes::LayerNormalizationAttributes(
    LayerNormalizationAttributes&& other) = default;
LayerNormalizationAttributes& LayerNormalizationAttributes::operator=(
    LayerNormalizationAttributes&& other) = default;

LstmAttributes::LstmAttributes() = default;
LstmAttributes::~LstmAttributes() = default;
LstmAttributes::LstmAttributes(LstmAttributes&& other) = default;
LstmAttributes& LstmAttributes::operator=(LstmAttributes&& other) = default;

LstmCellAttributes::LstmCellAttributes() = default;
LstmCellAttributes::~LstmCellAttributes() = default;
LstmCellAttributes::LstmCellAttributes(LstmCellAttributes&& other) = default;
LstmCellAttributes& LstmCellAttributes::operator=(LstmCellAttributes&& other) =
    default;

Pool2dAttributes::Pool2dAttributes() = default;
Pool2dAttributes::~Pool2dAttributes() = default;
Pool2dAttributes::Pool2dAttributes(Pool2dAttributes&& other) = default;
Pool2dAttributes& Pool2dAttributes::operator=(Pool2dAttributes&& other) =
    default;

SliceAttributes::SliceAttributes() = default;
SliceAttributes::~SliceAttributes() = default;
SliceAttributes::SliceAttributes(SliceAttributes&& other) = default;
SliceAttributes& SliceAttributes::operator=(SliceAttributes&& other) = default;

base::expected<OperandDescriptor, std::string> ValidateArgMinMaxAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    std::string_view label,
    uint32_t axis,
    OperandDataType output_data_type,
    bool keep_dimensions) {
  if (!context_properties.data_type_limits.arg_min_max_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.arg_min_max_input)));
  }

  if (!context_properties.data_type_limits.arg_min_max_output.data_types.Has(
          output_data_type)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedOpOutputTypeError(
                   output_data_type, context_properties.data_type_limits
                                         .arg_min_max_output.data_types)));
  }

  // An unranked input cannot have its axis validated against a rank; defer the
  // output inference to dispatch by propagating the unranked state.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(output_data_type);
  }

  ASSIGN_OR_RETURN(std::vector<Dimension> output_shape,
                   ValidateReduceAxesAndInferOutput(
                       input.shape(), std::array<uint32_t, 1>{axis},
                       keep_dimensions, label));

  return OperandDescriptor::Create(context_properties, output_data_type,
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string>
ValidateBatchNormalizationAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& mean,
    const OperandDescriptor& variance,
    const BatchNormalizationAttributes& attributes) {
  // Validate input operand.
  const std::string& label = attributes.label;
  if (!context_properties.data_type_limits.batch_normalization_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.batch_normalization_input)));
  }

  // An unranked input cannot have its axis validated against a rank; the output
  // is the same shape as the input, so propagate the unranked state and defer
  // to dispatch.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  if (attributes.axis >= input.Rank().value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The value of axis must be in the range [0, N-1] where N is the rank "
        "of the input tensor."));
  }

  Dimension input_size_on_axis = input.shape()[attributes.axis];
  // Validate mean operand.
  if (!context_properties.data_type_limits.batch_normalization_mean.Supports(
          mean)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kMeanParam, mean,
            context_properties.data_type_limits.batch_normalization_mean)));
  }
  RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
      mean, input.data_type(), input_size_on_axis, label, kMeanParam));

  // Validate variance operand.
  if (!context_properties.data_type_limits.batch_normalization_mean.Supports(
          variance)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kVarianceParam, variance,
            context_properties.data_type_limits.batch_normalization_mean)));
  }
  RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
      variance, input.data_type(), input_size_on_axis, label, kVarianceParam));

  // Validate scale operand.
  if (attributes.scale) {
    if (!context_properties.data_type_limits.batch_normalization_mean.Supports(
            attributes.scale.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kScaleParam, attributes.scale.value(),
              context_properties.data_type_limits.batch_normalization_mean)));
    }
    RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
        attributes.scale.value(), input.data_type(), input_size_on_axis, label,
        kScaleParam));
  }

  // Validate bias operand.
  if (attributes.bias) {
    if (!context_properties.data_type_limits.batch_normalization_mean.Supports(
            attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kBiasParam, attributes.bias.value(),
              context_properties.data_type_limits.batch_normalization_mean)));
    }
    RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
        attributes.bias.value(), input.data_type(), input_size_on_axis, label,
        kBiasParam));
  }

  // The output tensor of batchNormalization is the same shape as the input
  // tensor.
  return input;
}

base::expected<OperandDescriptor, std::string> ValidateCastAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    OperandDataType output_data_type,
    std::string_view label) {
  // Validate input operand.
  if (!context_properties.data_type_limits.cast_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.cast_input)));
  }

  // Validate output data type.
  if (!context_properties.data_type_limits.cast_input.data_types.Has(
          output_data_type)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedOpOutputTypeError(
                   output_data_type,
                   context_properties.data_type_limits.cast_input.data_types)));
  }

  // Cast preserves the input shape (only the data type changes); propagate an
  // unranked input to an unranked output.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(output_data_type);
  }

  return OperandDescriptor::Create(context_properties, output_data_type,
                                   input.shape(), label);
}

base::expected<OperandDescriptor, std::string> ValidateConcatAndInferOutput(
    const ContextProperties& context_properties,
    const std::vector<OperandDescriptor>& inputs,
    const uint32_t axis,
    std::string_view label) {
  if (inputs.empty()) {
    return base::unexpected(
        ErrorWithLabel(label, "The inputs should not be empty."));
  }

  if (inputs.size() > kMaxValidTensorCount) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf(
                   "The number of inputs must be less than or equal to %u.",
                   kMaxValidTensorCount)));
  }

  bool any_input_unranked = false;
  for (const auto& input : inputs) {
    if (!context_properties.data_type_limits.concat_inputs.Supports(input)) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedInputArgumentError(
              input, context_properties.data_type_limits.concat_inputs)));
    }
    any_input_unranked |= !input.HasRank();
  }

  // When any input has unknown rank, the axis cannot be range-checked and the
  // per-dimension compatibility cannot be inferred; defer to dispatch by
  // propagating the unranked state.
  if (any_input_unranked) {
    return OperandDescriptor::CreateUnranked(inputs[0].data_type());
  }

  const auto first_input_rank = inputs[0].Rank().value();
  // According to WebNN spec:
  // https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-concat-inputs-axis-axis,
  // the axis that the inputs concatenate along, with the value in the interval
  // [0, N-1] where N is the rank of input tensors. We just check the first
  // input rank here because we will check all inputs have same rank in the
  // following loop.
  if (axis >= first_input_rank) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The axis must be in the range [0, N-1] where N is the rank of input "
        "tensor."));
  }

  const auto first_input_shape = inputs[0].shape();
  const auto output_type = inputs[0].data_type();
  // The loop skips the first input to avoid repeated checks.
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i].data_type() != output_type) {
      return base::unexpected(
          ErrorWithLabel(label, "The input data types don't match."));
    }
    // According to WebNN spec:
    // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-concat, all input tensors
    // must have the same dimension.
    if (inputs[i].Rank().value() != first_input_rank) {
      return base::unexpected(ErrorWithLabel(
          label, "All input tensors must have the same dimension."));
    }
    // According to WebNN spec:
    // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-concat, all input tensors
    // must have the same shape, except for the size of the dimension to
    // concatenate on.
    for (size_t dim = 0; dim < first_input_rank; ++dim) {
      if (dim == axis) {
        continue;
      }
      const Dimension& dim_i = inputs[i].shape()[dim];
      const Dimension& dim_first = first_input_shape[dim];
      // When either dimension is dynamic, skip the equality check at build
      // time - compatibility will be validated at dispatch time.
      if (!DimensionsAreDefinitelyUnequal(dim_i, dim_first)) {
        continue;
      }
      return base::unexpected(ErrorWithLabel(
          label,
          "All input tensors must have the same shape, except for the size of "
          "the dimension to concatenate on."));
    }
  }
  // Calculate the output shape according to WebNN spec:
  // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-concat, the output tensor
  // has the same shape except on the dimension that all the inputs concatenated
  // along. The size of that dimension is computed as the sum of all the input
  // sizes of the same dimension.
  auto axis_size = base::CheckedNumeric<uint32_t>(0);
  bool has_dynamic_size = false;
  for (auto& input : inputs) {
    const Dimension& dim = input.shape()[axis];
    const uint32_t* size = std::get_if<uint32_t>(&dim);
    if (!size) {
      // A dynamic dim along the concat axis makes the sum unknown until
      // dispatch.
      has_dynamic_size = true;
      break;
    }
    axis_size += *size;
  }
  std::vector<Dimension> output_shape = first_input_shape;
  if (has_dynamic_size) {
    // The summed size is unknown at build time; it resolves at dispatch. Emit
    // an unnamed dynamic dimension.
    output_shape[axis] = DynamicDimension{};
  } else {
    if (!axis_size.IsValid()) {
      return base::unexpected(ErrorWithLabel(
          label, "The concatenated dimension size is too large."));
    }
    output_shape[axis] = axis_size.ValueOrDie();
  }

  return OperandDescriptor::Create(context_properties, output_type,
                                   output_shape, label);
}

base::expected<Size2d<double>, std::string>
ValidateAndCalculateConv2dOutputSizes(uint32_t input_height,
                                      uint32_t input_width,
                                      uint32_t filter_height,
                                      uint32_t filter_width,
                                      const Padding2d& padding,
                                      const Size2d<uint32_t>& strides,
                                      const Size2d<uint32_t>& dilations,
                                      std::string_view label) {
  if (strides.height == 0 || strides.width == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "All strides should be greater than 0."));
  }
  if (dilations.height == 0 || dilations.width == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "All dilations should be greater than 0."));
  }

  const auto float_output_height = CalculateConv2dOutputSize(
      input_height, filter_height, padding.beginning.height,
      padding.ending.height, strides.height, dilations.height, label);
  if (!float_output_height.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Failed to calculate the output height: " +
                                  float_output_height.error()));
  }

  const auto float_output_width = CalculateConv2dOutputSize(
      input_width, filter_width, padding.beginning.width, padding.ending.width,
      strides.width, dilations.width, label);
  if (!float_output_width.has_value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        "Failed to calculate the output width: " + float_output_width.error()));
  }

  return Size2d<double>{.height = float_output_height.value(),
                        .width = float_output_width.value()};
}

base::expected<OperandDescriptor, std::string> ValidateConv2dAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& filter,
    const Conv2dAttributes& attributes) {
  const std::string& label = attributes.label;
  // Validate input operand.
  if (!context_properties.data_type_limits.conv2d_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.conv2d_input)));
  }
  // conv2d is defined on a fixed 4-D input; it cannot operate on an operand of
  // unknown rank.
  if (!input.HasRank()) {
    return base::unexpected(
        ErrorWithLabel(label, kUnrankedInputNotSupportedError));
  }
  ASSIGN_OR_RETURN(Conv2dInputOutputInfo input_info,
                   GetConv2dInputInfo(label, input, attributes));

  // Validate filter operand.
  if (!context_properties.data_type_limits.conv2d_input.Supports(filter)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kFilterParam, filter,
                   context_properties.data_type_limits.conv2d_input)));
  }
  if (filter.data_type() != input.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The filter data type doesn't match the input data type."));
  }

  // Validate bias operand if it is present.
  if (attributes.bias_operand) {
    if (!context_properties.data_type_limits.conv2d_bias.Supports(
            attributes.bias_operand.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kBiasParam, attributes.bias_operand.value(),
                     context_properties.data_type_limits.conv2d_bias)));
    }
  }

  const auto filter_shape = filter.StaticShape();
  if (!filter_shape.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Dynamic shape is not supported for filter."));
  }

  uint32_t filter_height, filter_width, filter_input_channels, output_channels;
  // The conv2d filter layout specifies the filter layout format.
  switch (attributes.filter_layout) {
    case Conv2dFilterOperandLayout::kHwio:
      // "hwio": [height, width, input_channels/groups, output_channels]
      filter_height = (*filter_shape)[0];
      filter_width = (*filter_shape)[1];
      filter_input_channels = (*filter_shape)[2];
      output_channels = (*filter_shape)[3];
      break;
    case Conv2dFilterOperandLayout::kOhwi:
      // "ohwi": [output_channels, height, width, input_channels/groups]
      output_channels = (*filter_shape)[0];
      filter_height = (*filter_shape)[1];
      filter_width = (*filter_shape)[2];
      filter_input_channels = (*filter_shape)[3];
      break;
    case Conv2dFilterOperandLayout::kIhwo:
      // "ihwo": [input_channels/groups, height, width, output_channels]
      filter_input_channels = (*filter_shape)[0];
      filter_height = (*filter_shape)[1];
      filter_width = (*filter_shape)[2];
      output_channels = (*filter_shape)[3];
      break;
    case Conv2dFilterOperandLayout::kOihw:
      // "oihw": [output_channels, input_channels/groups, height, width]
      output_channels = (*filter_shape)[0];
      filter_input_channels = (*filter_shape)[1];
      filter_height = (*filter_shape)[2];
      filter_width = (*filter_shape)[3];
      break;
  }

  // Validate groups and input channels.
  if (attributes.groups == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The groups should be greater than 0."));
  }
  // When input channels is dynamic, skip the divisibility/equality check here;
  // it will be enforced at dispatch time once the concrete shape is known.
  if (std::holds_alternative<uint32_t>(input_info.channels)) {
    const uint32_t input_channels = std::get<uint32_t>(input_info.channels);
    if (input_channels % attributes.groups != 0 ||
        filter_input_channels != input_channels / attributes.groups) {
      return base::unexpected(ErrorWithLabel(
          label,
          "The groups must evenly divide the input channels to filter input "
          "channels."));
    }
  }
  if (output_channels % attributes.groups != 0) {
    return base::unexpected(ErrorWithLabel(
        label, "The groups must evenly divide the output channels."));
  }

  RETURN_IF_ERROR(ValidateIntermediatePaddedDescriptor(
      context_properties, input, attributes.padding, attributes.input_layout,
      input_info, label));

  // Validate and calculate output sizes. A static spatial dimension is reduced
  // to a concrete output size; a dynamic one yields an unnamed dynamic output
  // dimension (resolved at dispatch).
  Dimension output_height_dim, output_width_dim;

  if (const uint32_t* input_height =
          std::get_if<uint32_t>(&input_info.height)) {
    ASSIGN_OR_RETURN(
        double float_output_height,
        CalculateConv2dOutputSize(
            *input_height, filter_height, attributes.padding.beginning.height,
            attributes.padding.ending.height, attributes.strides.height,
            attributes.dilations.height, label));
    output_height_dim = base::ClampFloor<uint32_t>(float_output_height);
  } else {
    output_height_dim = DynamicDimension{};
  }

  if (const uint32_t* input_width = std::get_if<uint32_t>(&input_info.width)) {
    ASSIGN_OR_RETURN(
        double float_output_width,
        CalculateConv2dOutputSize(
            *input_width, filter_width, attributes.padding.beginning.width,
            attributes.padding.ending.width, attributes.strides.width,
            attributes.dilations.width, label));
    output_width_dim = base::ClampFloor<uint32_t>(float_output_width);
  } else {
    output_width_dim = DynamicDimension{};
  }

  Conv2dInputOutputInfo output_info{.batches = input_info.batches,
                                    .channels = output_channels,
                                    .height = output_height_dim,
                                    .width = output_width_dim};
  return ValidateConv2dBiasAndCreateOutputOperand(context_properties, input,
                                                  attributes, output_info);
}

base::expected<OperandDescriptor, std::string>
ValidateConvTranspose2dAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& filter,
    const ConvTranspose2dAttributes& attributes) {
  // Validate input operand.
  const std::string& label = attributes.label;
  if (!context_properties.data_type_limits.conv_transpose2d_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.conv_transpose2d_input)));
  }
  // convTranspose2d is defined on a fixed 4-D input; it cannot operate on an
  // operand of unknown rank.
  if (!input.HasRank()) {
    return base::unexpected(
        ErrorWithLabel(label, kUnrankedInputNotSupportedError));
  }
  ASSIGN_OR_RETURN(Conv2dInputOutputInfo input_info,
                   GetConv2dInputInfo(label, input, attributes));

  // Validate filter operand.
  if (!context_properties.data_type_limits.conv_transpose2d_input.Supports(
          filter)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kFilterParam, filter,
            context_properties.data_type_limits.conv_transpose2d_input)));
  }
  if (filter.data_type() != input.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The filter data type doesn't match the input data type."));
  }

  // Validate bias operand if it is present.
  if (attributes.bias_operand) {
    if (!context_properties.data_type_limits.conv_transpose2d_bias.Supports(
            attributes.bias_operand.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kBiasParam, attributes.bias_operand.value(),
              context_properties.data_type_limits.conv_transpose2d_bias)));
    }
  }

  const auto filter_shape = filter.StaticShape();
  if (!filter_shape.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Dynamic shape is not supported for filter."));
  }
  uint32_t input_channels, filter_height, filter_width, filter_output_channels;
  // The conv2d filter layout specifies the filter layout format.
  switch (attributes.filter_layout) {
    case ConvTranspose2dFilterOperandLayout::kIohw:
      // "iohw": [input_channels, output_channels/groups, height, width]
      input_channels = (*filter_shape)[0];
      filter_output_channels = (*filter_shape)[1];
      filter_height = (*filter_shape)[2];
      filter_width = (*filter_shape)[3];
      break;
    case ConvTranspose2dFilterOperandLayout::kHwoi:
      // "hwoi": [height, width, output_channels/groups, input_channels]
      filter_height = (*filter_shape)[0];
      filter_width = (*filter_shape)[1];
      filter_output_channels = (*filter_shape)[2];
      input_channels = (*filter_shape)[3];
      break;
    case ConvTranspose2dFilterOperandLayout::kOhwi:
      // "ohwi": [output_channels/groups, height, width, input_channels]
      filter_output_channels = (*filter_shape)[0];
      filter_height = (*filter_shape)[1];
      filter_width = (*filter_shape)[2];
      input_channels = (*filter_shape)[3];
      break;
  }
  // Validate groups, input channels and calculate output channels.
  if (attributes.groups == 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The groups should be greater than 0."));
  }
  // When input channels is dynamic, skip the equality check here; it will be
  // enforced at dispatch time once the concrete shape is known.
  if (DimensionsAreDefinitelyUnequal(input_info.channels, input_channels)) {
    return base::unexpected(ErrorWithLabel(
        label, "The input channels should equal to filter input channels."));
  }
  const auto checked_output_channels =
      base::CheckedNumeric<uint32_t>(filter_output_channels) *
      attributes.groups;
  if (!checked_output_channels.IsValid()) {
    return base::unexpected(
        ErrorWithLabel(label, "The output channels is too large."));
  }
  const uint32_t output_channels = checked_output_channels.ValueOrDie();

  RETURN_IF_ERROR(ValidateIntermediatePaddedDescriptor(
      context_properties, input, attributes.padding, attributes.input_layout,
      input_info, label));

  // Validate and calculate output sizes.
  uint32_t output_height = 0, output_width = 0;
  if (attributes.output_sizes) {
    if (std::holds_alternative<DynamicDimension>(input_info.height) ||
        std::holds_alternative<DynamicDimension>(input_info.width)) {
      return base::unexpected(
          ErrorWithLabel(label,
                         "Dynamic input sizes are not supported when "
                         "output_sizes are present."));
    }
    const auto& output_sizes = attributes.output_sizes;
    output_height = output_sizes->height;
    output_width = output_sizes->width;
    if (output_height <= 0 || output_width <= 0) {
      return base::unexpected(
          ErrorWithLabel(label, "All output sizes should be greater than 0."));
    }
    const auto strides = attributes.strides;
    ASSIGN_OR_RETURN(
        Size2d<uint32_t> calculated_output_sizes,
        ValidateAndCalculateConvTranspose2dOutputSizes(
            std::get<uint32_t>(input_info.height),
            std::get<uint32_t>(input_info.width), filter_height, filter_width,
            attributes.padding, strides, attributes.dilations,
            // According to WebNN spec:
            // https://webmachinelearning.github.io/webnn/#dom-mlconvtranspose2doptions-outputsizes
            // When the output sizes are explicitly specified, the output
            // padding values in outputPadding are ignored.
            {0, 0}, label));
    const auto calculated_output_height = calculated_output_sizes.height;
    const auto max_output_height =
        base::CheckedNumeric<uint32_t>(calculated_output_height) +
        strides.height;
    if (!max_output_height.IsValid()) {
      return base::unexpected(ErrorWithLabel(
          label, "The checked maximum output height is too large"));
    }
    if (output_height < calculated_output_height ||
        output_height >= max_output_height.ValueOrDie()) {
      return base::unexpected(
          ErrorWithLabel(label, "The height of output sizes is invalid."));
    }
    const auto calculated_output_width = calculated_output_sizes.width;
    const auto max_output_width =
        base::CheckedNumeric<uint32_t>(calculated_output_width) + strides.width;
    if (!max_output_width.IsValid()) {
      return base::unexpected(ErrorWithLabel(
          label, "The checked maximum output width is too large"));
    }
    if (output_width < calculated_output_width ||
        output_width >= max_output_width.ValueOrDie()) {
      return base::unexpected(
          ErrorWithLabel(label, "The width of output sizes is invalid."));
    }
  } else {
    // Validate strides, dilations, and output_padding unconditionally.
    if (attributes.strides.height == 0 || attributes.strides.width == 0) {
      return base::unexpected(
          ErrorWithLabel(label, "All strides should be greater than 0."));
    }
    if (attributes.dilations.height == 0 || attributes.dilations.width == 0) {
      return base::unexpected(
          ErrorWithLabel(label, "All dilations should be greater than 0."));
    }
    if (attributes.output_padding.height >= attributes.strides.height ||
        attributes.output_padding.width >= attributes.strides.width) {
      return base::unexpected(ErrorWithLabel(
          label,
          "The output padding must be smaller than the stride along the same "
          "dimension."));
    }

    // Compute each spatial dimension independently so that a static dimension
    // is correctly computed even when the other spatial dimension is dynamic.
    if (const uint32_t* input_height =
            std::get_if<uint32_t>(&input_info.height)) {
      ASSIGN_OR_RETURN(
          output_height,
          CalculateConvTranspose2dOutputSize(
              *input_height, filter_height, attributes.padding.beginning.height,
              attributes.padding.ending.height, attributes.strides.height,
              attributes.dilations.height, attributes.output_padding.height));
    }

    if (const uint32_t* input_width =
            std::get_if<uint32_t>(&input_info.width)) {
      ASSIGN_OR_RETURN(
          output_width,
          CalculateConvTranspose2dOutputSize(
              *input_width, filter_width, attributes.padding.beginning.width,
              attributes.padding.ending.width, attributes.strides.width,
              attributes.dilations.width, attributes.output_padding.width));
    }
  }

  Dimension output_height_dim, output_width_dim;
  if (std::holds_alternative<uint32_t>(input_info.height)) {
    output_height_dim = output_height;
  } else {
    output_height_dim = DynamicDimension{};
  }

  if (std::holds_alternative<uint32_t>(input_info.width)) {
    output_width_dim = output_width;
  } else {
    output_width_dim = DynamicDimension{};
  }

  Conv2dInputOutputInfo output_info{.batches = input_info.batches,
                                    .channels = output_channels,
                                    .height = output_height_dim,
                                    .width = output_width_dim};
  return ValidateConv2dBiasAndCreateOutputOperand(context_properties, input,
                                                  attributes, output_info);
}

base::expected<OperandDescriptor, std::string>
ValidateCumulativeSumAndInferOutput(const ContextProperties& context_properties,
                                    const OperandDescriptor& input,
                                    const uint32_t axis,
                                    std::string_view label) {
  if (input.Rank().value() <= axis) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf("The axis (%u) must be in the range [0, N-1] "
                                  "where N (%u) is the rank of input "
                                  "tensor.",
                                  axis, input.Rank().value())));
  }

  if (!context_properties.data_type_limits.cumulative_sum_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.cumulative_sum_input)));
  }

  // The data type and shape of input determine the output.
  return input;
}

// This helper method is intended to validate scale and zero_point
// operands of quantizeLinear and dequantizeLinear against the input
// operand.
// TODO(crbug.com/396176047): Make scale and zero_point's rank match with
// input.
base::expected<void, std::string>
ValidateScaleZeroPointOperandShapeIsCompatibleWithInput(
    base::span<const uint32_t> input_shape,
    base::span<const uint32_t> scale_shape,
    base::span<const uint32_t> zero_point_shape,
    std::string_view label) {
  // Check whether `scale_shape` is a subsample of `input_shape`.
  if (scale_shape.size() != input_shape.size()) {
    return base::unexpected(ErrorWithLabel(
        label, "The rank of scale is not equal to the rank of input."));
  }

  for (size_t i = 0; i < scale_shape.size(); ++i) {
    auto scale_dim = scale_shape[i];
    auto input_dim = input_shape[i];
    // The block_size should be an integer where block_size = dim_input /
    // dim_scale along the axis.
    if (input_dim % scale_dim != 0) {
      return base::unexpected(ErrorWithLabel(
          label,
          "The shape of scale is not a subsample of the shape of input."));
    }
  }

  if (!std::ranges::equal(scale_shape, zero_point_shape)) {
    return base::unexpected(ErrorWithLabel(
        label, "The shape of scale and zero point must be the same."));
  }
  return base::ok();
}

base::expected<OperandDescriptor, std::string>
ValidateDequantizeLinearAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& scale,
    const OperandDescriptor& zero_point,
    std::string_view label) {
  // Validate scale and zero_point operands.
  const auto input_shape = input.StaticShape();
  const auto scale_shape = scale.StaticShape();
  const auto zero_point_shape = zero_point.StaticShape();
  if (!input_shape.has_value() || !scale_shape.has_value() ||
      !zero_point_shape.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Dynamic input shape is not supported."));
  }
  RETURN_IF_ERROR(ValidateScaleZeroPointOperandShapeIsCompatibleWithInput(
      *input_shape, *scale_shape, *zero_point_shape, label));

  if (!context_properties.data_type_limits.dequantize_linear_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.dequantize_linear_input)));
  }

  if (!context_properties.data_type_limits.dequantize_linear_input.Supports(
          zero_point)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kZeroPointParam, zero_point,
            context_properties.data_type_limits.dequantize_linear_input)));
  }

  if (input.data_type() != zero_point.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The data type of input and zero point must be the same."));
  }

  if (!context_properties.data_type_limits.dequantize_linear_scale.Supports(
          scale)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kScaleParam, scale,
            context_properties.data_type_limits.dequantize_linear_scale)));
  }

  // The data type of scale determines the output type.
  return OperandDescriptor::Create(context_properties, scale.data_type(),
                                   input.shape(), label);
}

base::expected<OperandDescriptor, std::string>
ValidateQuantizeLinearAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& scale,
    const OperandDescriptor& zero_point,
    std::string_view label) {
  const auto input_shape = input.StaticShape();
  const auto scale_shape = scale.StaticShape();
  const auto zero_point_shape = zero_point.StaticShape();
  if (!input_shape.has_value() || !scale_shape.has_value() ||
      !zero_point_shape.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Dynamic input shape is not supported."));
  }
  // Validate scale and zero_point operands.
  RETURN_IF_ERROR(ValidateScaleZeroPointOperandShapeIsCompatibleWithInput(
      *input_shape, *scale_shape, *zero_point_shape, label));

  if (!context_properties.data_type_limits.quantize_linear_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.quantize_linear_input)));
  }

  if (!context_properties.data_type_limits.quantize_linear_input.Supports(
          scale)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kScaleParam, scale,
                   context_properties.data_type_limits.quantize_linear_input)));
  }

  if (input.data_type() != scale.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The data type of input and scale must be the same."));
  }

  if (!context_properties.data_type_limits.quantize_linear_zero_point.Supports(
          zero_point)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kZeroPointParam, zero_point,
            context_properties.data_type_limits.quantize_linear_zero_point)));
  }
  // The data type of zero_point determines the output type.
  return OperandDescriptor::Create(context_properties, zero_point.data_type(),
                                   input.shape(), label);
}

base::expected<OperandDescriptor, std::string> ValidateExpandAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    base::span<const Dimension> new_shape,
    std::string_view label,
    base::span<const DynamicDimension> known_dynamic_dims) {
  // Validate input operand.
  if (!context_properties.data_type_limits.expand_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.expand_input)));
  }

  uint32_t new_rank = base::checked_cast<uint32_t>(new_shape.size());
  if (!context_properties.data_type_limits.expand_input.ranks.Supports(
          new_rank)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedOpOutputRankError(
            new_rank, context_properties.data_type_limits.expand_input.ranks)));
  }

  // The output shape is fully determined by `new_shape` regardless of the input
  // rank. When the input is unranked the broadcast compatibility cannot be
  // checked at build time; defer it to dispatch and return the (ranked) output
  // described by `new_shape`.
  if (!input.HasRank()) {
    return OperandDescriptor::Create(context_properties, input.data_type(),
                                     new_shape, label);
  }

  std::optional<std::vector<Dimension>> output_shape =
      ExpandShape(input.shape(), new_shape, known_dynamic_dims);
  if (!output_shape) {
    return base::unexpected(ErrorWithLabel(
        label, "The input shape is not broadcastable to the new shape."));
  }
  CHECK(new_shape == *output_shape);

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   *output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidateGatherAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& indices,
    const uint32_t axis,
    std::string_view label) {
  // Validate input operand.
  if (!context_properties.data_type_limits.gather_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.gather_input)));
  }

  // The output rank depends on the input rank (input.rank - 1 + indices.rank);
  // an unranked input or indices makes it unknown, so defer to dispatch.
  if (!input.HasRank() || !indices.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  if (input.Rank().value() <= axis) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf("The axis (%u) must be in the range [0, N-1] "
                                  "where N=%u is the rank of input tensor.",
                                  axis, input.Rank().value())));
  }

  // Validate indices operand.
  if (!context_properties.data_type_limits.gather_indices.Supports(indices)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kIndicesParam, indices,
                   context_properties.data_type_limits.gather_indices)));
  }

  auto checked_output_rank =
      base::CheckedNumeric<uint32_t>(input.Rank().value()) - 1 +
      indices.Rank().value();
  if (!checked_output_rank.IsValid()) {
    return base::unexpected(
        ErrorWithLabel(label, "The output rank is too large."));
  }

  std::vector<Dimension> output_shape;
  output_shape.reserve(checked_output_rank.ValueOrDie());
  for (uint32_t i = 0; i < input.Rank().value(); ++i) {
    if (i == axis) {
      std::ranges::copy(indices.shape(), std::back_inserter(output_shape));
    } else {
      output_shape.push_back(input.shape()[i]);
    }
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string>
ValidateGatherElementsAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& indices,
    const uint32_t axis,
    std::string_view label) {
  // Validate input operand.
  if (!context_properties.data_type_limits.gather_elements_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.gather_elements_input)));
  }

  // The output has the same shape as indices; when either operand is unranked
  // the rank-dependent checks cannot run, so defer to dispatch.
  if (!input.HasRank() || !indices.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  if (input.Rank().value() <= axis) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf("The axis (%u) must be in the range [0, N-1] "
                                  "where N=%u is the rank of input "
                                  "tensor.",
                                  axis, input.Rank().value())));
  }

  // Validate indices operand.
  if (!context_properties.data_type_limits.gather_elements_indices.Supports(
          indices)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kIndicesParam, indices,
            context_properties.data_type_limits.gather_elements_indices)));
  }

  if (input.Rank().value() != indices.Rank().value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf(
            "The input rank (%u) must be equal to the indices rank (%u).",
            input.Rank().value(), indices.Rank().value())));
  }

  for (uint32_t i = 0; i < input.Rank().value(); ++i) {
    if (i == axis) {
      continue;
    }

    if (DimensionsAreDefinitelyUnequal(input.shape()[i], indices.shape()[i])) {
      return base::unexpected(
          ErrorWithLabel(label,
                         "Except on the axis dimension, the input and indices "
                         "tensor must have the same dimension size."));
    }
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   indices.shape(), label);
}

base::expected<OperandDescriptor, std::string> ValidateGatherNDAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& indices,
    std::string_view label) {
  // Validate input operand.
  if (!context_properties.data_type_limits.gather_nd_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.gather_nd_input)));
  }

  // Validate indices operand.
  if (!context_properties.data_type_limits.gather_nd_indices.Supports(
          indices)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kIndicesParam, indices,
                   context_properties.data_type_limits.gather_nd_indices)));
  }

  // The output rank depends on both the input rank and the indices shape; an
  // unranked input or indices makes it unknown, so defer to dispatch.
  if (!input.HasRank() || !indices.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  // The last dimension of indices must be static because the output rank
  // depends on it.
  const Dimension& indices_last_dim =
      indices.shape()[indices.Rank().value() - 1];
  if (!std::holds_alternative<uint32_t>(indices_last_dim)) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The last dimension of indices must be static because the output rank "
        "depends on it."));
  }

  uint32_t indices_last_dimension_size = std::get<uint32_t>(indices_last_dim);
  if (indices_last_dimension_size > input.Rank().value()) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf(
                   "The last dimension size of indices (%u) must be less than "
                   "or equal to the input rank (%u).",
                   indices_last_dimension_size, input.Rank().value())));
  }

  auto checked_output_rank = base::CheckedNumeric(indices.Rank().value()) - 1 +
                             input.Rank().value() - indices_last_dimension_size;
  if (!checked_output_rank.IsValid()) {
    return base::unexpected(
        ErrorWithLabel(label, "The output rank is too large."));
  }

  std::vector<Dimension> output_shape;
  output_shape.reserve(checked_output_rank.ValueOrDie());

  const std::vector<Dimension>& indices_shape = indices.shape();
  std::ranges::copy(indices_shape.begin(), indices_shape.end() - 1,
                    std::back_inserter(output_shape));
  const std::vector<Dimension>& input_shape = input.shape();
  std::ranges::copy(input_shape.begin() + indices_last_dimension_size,
                    input_shape.end(),
                    std::back_inserter(output_shape));

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidateGemmAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& a,
    const OperandDescriptor& b,
    const GemmAttributes& attributes) {
  const std::string& label = attributes.label;
  // Validate a and b operand.
  if (!context_properties.data_type_limits.gemm_a.Supports(a)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(kGemmAParam, a,
                                  context_properties.data_type_limits.gemm_a)));
  }

  if (!context_properties.data_type_limits.gemm_a.Supports(b)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(kGemmBParam, b,
                                  context_properties.data_type_limits.gemm_a)));
  }

  if (a.data_type() != b.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The data types of first two inputs don't match."));
  }

  // gemm operates on fixed 2-D matrices; it cannot operate on an operand of
  // unknown rank.
  if (!a.HasRank() || !b.HasRank()) {
    return base::unexpected(
        ErrorWithLabel(label, kUnrankedInputNotSupportedError));
  }

  auto shape_a = a.shape();
  if (attributes.a_transpose) {
    std::ranges::reverse(shape_a);
  }
  // The second input 2-D tensor with shape [K, N] if bTranspose is false, or
  // [N, K] if bTranspose is true.
  auto shape_b = b.shape();
  if (attributes.b_transpose) {
    std::ranges::reverse(shape_b);
  }
  // The number of columns in the first matrix must be equal to the number of
  // rows in the second matrix. When either dimension is dynamic, skip this
  // check at build time - it will be validated at dispatch time.
  if (DimensionsAreDefinitelyUnequal(shape_a[1], shape_b[0])) {
    auto to_string = [](const webnn::Dimension& dim) -> std::string {
      if (std::holds_alternative<uint32_t>(dim)) {
        return base::StringPrintf("%u", std::get<uint32_t>(dim));
      }
      return std::get<webnn::DynamicDimension>(dim).name.value_or("?");
    };
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf(
            "The number of columns (%s) in the %sfirst matrix isn't equal to "
            "the number of rows (%s) in the %ssecond matrix.",
            to_string(shape_a[1]).c_str(),
            attributes.a_transpose ? "transposed " : "",
            to_string(shape_b[0]).c_str(),
            attributes.b_transpose ? "transposed " : "")));
  };
  // The output is 2-D tensor of shape [M, N].
  std::vector<Dimension> output_shape = {shape_a[0], shape_b[1]};
  // The third input tensor c is either a scalar, or of the shape that is
  // unidirectionally broadcastable to the output shape [M, N].
  if (attributes.c_operand) {
    if (!context_properties.data_type_limits.gemm_c.Supports(
            attributes.c_operand.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kGemmCParam, attributes.c_operand.value(),
                     context_properties.data_type_limits.gemm_c)));
    }

    if (attributes.c_operand->data_type() != a.data_type()) {
      return base::unexpected(ErrorWithLabel(
          label,
          "The third input data type doesn't match other inputs' data type."));
    }

    // An unranked bias (rank only known at dispatch) has no shape() to
    // broadcast-check here; defer the check to dispatch, when c is concrete.
    // The 2-D output shape [M, N] is determined by a and b regardless.
    if (attributes.c_operand->HasRank() &&
        !BroadcastShapes(attributes.c_operand->shape(), output_shape,
                         /*bidirectional=*/false)) {
      return base::unexpected(ErrorWithLabel(
          label,
          "The third input tensor isn't unidirectionally broadcastable to the "
          "output tensor."));
    }
  }

  return OperandDescriptor::Create(context_properties, a.data_type(),
                                   output_shape, label);
}

base::expected<std::vector<OperandDescriptor>, std::string>
ValidateGruAndInferOutput(const ContextProperties& context_properties,
                          const OperandDescriptor& input,
                          const OperandDescriptor& weight,
                          const OperandDescriptor& recurrent_weight,
                          uint32_t steps,
                          uint32_t hidden_size,
                          const GruAttributes& attributes) {
  const std::string& label = attributes.label;
  if (steps <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The steps must be greater than 0."));
  }
  if (hidden_size <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size must be greater than 0."));
  }

  // Validate the input operand.
  if (!context_properties.data_type_limits.gru_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.gru_input)));
  }

  // TODO(crbug.com/329482489): Support dynamic shapes.
  const auto input_dimensions = input.StaticShape();
  if (!input_dimensions.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Dynamic input is not supported."));
  }
  if ((*input_dimensions)[0] != steps) {
    return base::unexpected(ErrorWithLabel(
        label, "The input dimension[0] must be equal to the steps."));
  }
  const auto batch_size = (*input_dimensions)[1];
  const auto input_size = (*input_dimensions)[2];
  auto checked_three_times_hidden_size = base::CheckedNumeric(hidden_size) * 3;
  uint32_t three_times_hidden_size;
  if (!checked_three_times_hidden_size.AssignIfValid(
          &three_times_hidden_size)) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size is too large."));
  }
  const uint32_t num_directions =
      attributes.direction == RecurrentNetworkDirection::kBoth ? 2 : 1;

  // Validate the weight operand.
  if (!context_properties.data_type_limits.gru_input.Supports(weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kWeightParam, weight,
                   context_properties.data_type_limits.gru_input)));
  }
  std::array<uint32_t, 3> expected_weight_shape = {
      num_directions, three_times_hidden_size, input_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      weight, kWeightParam, expected_weight_shape, input.data_type(), label));

  // Validate the recurrent weight operand.
  if (!context_properties.data_type_limits.gru_input.Supports(
          recurrent_weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kRecurrentWeightParam, recurrent_weight,
                   context_properties.data_type_limits.gru_input)));
  }
  std::array<uint32_t, 3> expected_recurrent_weight_shape = {
      num_directions, three_times_hidden_size, hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      recurrent_weight, kRecurrentWeightParam, expected_recurrent_weight_shape,
      input.data_type(), label));

  // Validate the bias operand.
  if (attributes.bias) {
    if (!context_properties.data_type_limits.gru_bias.Supports(
            attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kBiasParam, attributes.bias.value(),
                     context_properties.data_type_limits.gru_bias)));
    }
    std::array<uint32_t, 2> expected_bias_shape = {num_directions,
                                                   three_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        *attributes.bias, kBiasParam, expected_bias_shape, input.data_type(),
        label));
  }

  // Validate the recurrent bias operand.
  if (attributes.recurrent_bias) {
    if (!context_properties.data_type_limits.gru_bias.Supports(
            attributes.recurrent_bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kRecurrentBiasParam, attributes.recurrent_bias.value(),
                     context_properties.data_type_limits.gru_bias)));
    }
    std::array<uint32_t, 2> expected_recurrent_bias_shape = {
        num_directions, three_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        *attributes.recurrent_bias, kRecurrentBiasParam,
        expected_recurrent_bias_shape, input.data_type(), label));
  }

  // Validate the initial hidden state operand.
  if (attributes.initial_hidden_state) {
    if (!context_properties.data_type_limits.gru_input.Supports(
            attributes.initial_hidden_state.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kInitialHiddenStateParam, attributes.initial_hidden_state.value(),
              context_properties.data_type_limits.gru_input)));
    }
    std::array<uint32_t, 3> expected_initial_hidden_state_shape = {
        num_directions, batch_size, hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        *attributes.initial_hidden_state, kInitialHiddenStateParam,
        expected_initial_hidden_state_shape, input.data_type(), label));
  }

  if (attributes.activation_count != 2) {
    return base::unexpected(
        ErrorWithLabel(label, "The number of activations must be 2."));
  }

  std::vector<OperandDescriptor> outputs;
  ASSIGN_OR_RETURN(
      OperandDescriptor output,
      OperandDescriptor::Create(
          context_properties, input.data_type(),
          std::array{num_directions, batch_size, hidden_size}, label));
  outputs.push_back(std::move(output));
  if (attributes.return_sequence) {
    ASSIGN_OR_RETURN(
        OperandDescriptor return_sequence_output,
        OperandDescriptor::Create(
            context_properties, input.data_type(),
            std::array{steps, num_directions, batch_size, hidden_size}, label));
    outputs.push_back(std::move(return_sequence_output));
  }

  return outputs;
}

base::expected<OperandDescriptor, std::string> ValidateGruCellAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& weight,
    const OperandDescriptor& recurrent_weight,
    const OperandDescriptor& hidden_state,
    uint32_t hidden_size,
    const GruCellAttributes& attributes) {
  const std::string& label = attributes.label;
  if (hidden_size <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size must be greater than 0."));
  }

  // Validate the input operand.
  if (!context_properties.data_type_limits.gru_cell_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.gru_cell_input)));
  }

  // TODO(crbug.com/329482489): Support dynamic shapes.
  const auto input_shape = input.StaticShape();
  if (!input_shape.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Dynamic input is not supported."));
  }
  const uint32_t batch_size = (*input_shape)[0];
  const uint32_t input_size = (*input_shape)[1];
  auto checked_three_times_hidden_size = base::CheckedNumeric(hidden_size) * 3;
  uint32_t three_times_hidden_size;
  if (!checked_three_times_hidden_size.AssignIfValid(
          &three_times_hidden_size)) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size is too large."));
  }

  // Validate the weight operand.
  if (!context_properties.data_type_limits.gru_cell_input.Supports(weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kWeightParam, weight,
                   context_properties.data_type_limits.gru_cell_input)));
  }
  std::array<uint32_t, 2> expected_weight_shape = {three_times_hidden_size,
                                                   input_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      weight, kWeightParam, expected_weight_shape, input.data_type(), label));

  // Validate the recurrent weight operand.
  if (!context_properties.data_type_limits.gru_cell_input.Supports(
          recurrent_weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kRecurrentWeightParam, recurrent_weight,
                   context_properties.data_type_limits.gru_cell_input)));
  }
  std::array<uint32_t, 2> expected_recurrent_weight_shape = {
      three_times_hidden_size, hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      recurrent_weight, kRecurrentWeightParam, expected_recurrent_weight_shape,
      input.data_type(), label));

  // Validate the hidden state operand.
  if (!context_properties.data_type_limits.gru_cell_input.Supports(
          hidden_state)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kHiddenStateParam, hidden_state,
                   context_properties.data_type_limits.gru_cell_input)));
  }
  std::array<uint32_t, 2> expected_hidden_state_shape = {batch_size,
                                                         hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      hidden_state, kHiddenStateParam, expected_hidden_state_shape,
      input.data_type(), label));

  // Validate the bias operand.
  if (attributes.bias) {
    if (!context_properties.data_type_limits.gru_cell_bias.Supports(
            attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kBiasParam, attributes.bias.value(),
                     context_properties.data_type_limits.gru_cell_bias)));
    }
    std::array<uint32_t, 1> expected_bias_shape = {three_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        *attributes.bias, kBiasParam, expected_bias_shape, input.data_type(),
        label));
  }

  // Validate the recurrent bias operand.
  if (attributes.recurrent_bias) {
    if (!context_properties.data_type_limits.gru_cell_bias.Supports(
            attributes.recurrent_bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kRecurrentBiasParam, attributes.recurrent_bias.value(),
                     context_properties.data_type_limits.gru_cell_bias)));
    }
    std::array<uint32_t, 1> expected_recurrent_bias_shape = {
        three_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        *attributes.recurrent_bias, kRecurrentBiasParam,
        expected_recurrent_bias_shape, input.data_type(), label));
  }

  if (attributes.activation_count != 2) {
    return base::unexpected(
        ErrorWithLabel(label, "The number of activations must be 2."));
  }

  std::array<uint32_t, 2> output_shape{batch_size, hidden_size};
  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string>
ValidateInstanceNormalizationAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const InstanceNormalizationAttributes& attributes) {
  const std::string& label = attributes.label;
  // Validate the input operand.
  if (!context_properties.data_type_limits.instance_normalization_input
           .Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.instance_normalization_input)));
  }

  // instanceNormalization is defined on a fixed 4-D input (it normalizes over
  // the spatial dimensions); it cannot operate on an operand of unknown rank.
  if (!input.HasRank()) {
    return base::unexpected(
        ErrorWithLabel(label, kUnrankedInputNotSupportedError));
  }

  uint32_t axis;
  switch (attributes.layout) {
    case InputOperandLayout::kNchw:
      axis = 1;
      break;
    case InputOperandLayout::kNhwc:
      axis = 3;
      break;
  }

  // Validate scale operand.
  if (attributes.scale.has_value()) {
    if (!context_properties.data_type_limits.instance_normalization_scale
             .Supports(attributes.scale.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(kScaleParam, attributes.scale.value(),
                                    context_properties.data_type_limits
                                        .instance_normalization_scale)));
    }
    RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
        attributes.scale.value(), input.data_type(), input.shape()[axis], label,
        kScaleParam));
  }

  // Validate the bias operand.
  if (attributes.bias.has_value()) {
    if (!context_properties.data_type_limits.instance_normalization_scale
             .Supports(attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(kBiasParam, attributes.bias.value(),
                                           context_properties.data_type_limits
                                               .instance_normalization_scale)));
    }
    RETURN_IF_ERROR(ValidateNormalizationOperandIsCompatibleWithInput(
        attributes.bias.value(), input.data_type(), input.shape()[axis], label,
        kBiasParam));
  }

  return input;
}

base::expected<OperandDescriptor, std::string>
ValidateLayerNormalizationAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    base::span<const uint32_t> axes,
    const LayerNormalizationAttributes& attributes) {
  const std::string& label = attributes.label;
  // Validate the input operand.
  if (!context_properties.data_type_limits.layer_normalization_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.layer_normalization_input)));
  }

  // The output has the same shape as the input; when the input is unranked the
  // reduction axes cannot be validated, so defer to dispatch.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  // Ensure that the axes are all less than the input rank and have no
  // duplication.
  RETURN_IF_ERROR(ValidateAxes(axes, input.Rank().value(), label));

  const auto input_dimensions = input.shape();
  // The dimensions for layerNormalization to reduce along.
  std::vector<Dimension> reduction_dimensions;
  reduction_dimensions.reserve(axes.size());
  std::ranges::transform(
      axes, std::back_inserter(reduction_dimensions),
      [&input_dimensions](uint32_t axis) { return input_dimensions[axis]; });

  // scale/bias must match the reduction dimensions. Reject only on a definite
  // mismatch (a different rank, or a dimension pair that is statically
  // unequal); a dynamic dimension on either side is deferred to dispatch.
  auto shape_definitely_mismatches = [&reduction_dimensions](
                                         base::span<const Dimension> shape) {
    if (shape.size() != reduction_dimensions.size()) {
      return true;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
      if (DimensionsAreDefinitelyUnequal(shape[i], reduction_dimensions[i])) {
        return true;
      }
    }
    return false;
  };

  // Validate the scale operand.
  if (attributes.scale.has_value()) {
    if (attributes.scale->data_type() != input.data_type()) {
      return base::unexpected(ErrorWithLabel(
          label,
          "For scale operand: the data type doesn't match the input data "
          "type."));
    }
    // An unranked scale (rank only known at dispatch) has no shape() to check;
    // defer to dispatch, when it is concrete.
    if (attributes.scale->HasRank() &&
        shape_definitely_mismatches(attributes.scale->shape())) {
      return base::unexpected(ErrorWithLabel(
          label,
          "For scale operand: the shape doesn't match the axis dimensions of "
          "the input."));
    }
  }

  // Validate the bias operand.
  if (attributes.bias.has_value()) {
    if (attributes.bias->data_type() != input.data_type()) {
      return base::unexpected(
          ErrorWithLabel(label,
                         "For bias operand: the data type doesn't match the "
                         "input data type."));
    }
    // An unranked bias (rank only known at dispatch) has no shape() to check;
    // defer to dispatch, when it is concrete.
    if (attributes.bias->HasRank() &&
        shape_definitely_mismatches(attributes.bias->shape())) {
      return base::unexpected(ErrorWithLabel(
          label,
          "For bias operand: the shape doesn't match the axis dimensions of "
          "the input."));
    }
  }

  return input;
}

base::expected<std::vector<OperandDescriptor>, std::string>
ValidateLstmAndInferOutput(const ContextProperties& context_properties,
                           const OperandDescriptor& input,
                           const OperandDescriptor& weight,
                           const OperandDescriptor& recurrent_weight,
                           const uint32_t steps,
                           const uint32_t hidden_size,
                           const LstmAttributes& attributes) {
  const std::string& label = attributes.label;
  if (steps <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The steps must be greater than 0."));
  }
  if (hidden_size <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size must be greater than 0."));
  }

  uint32_t four_times_hidden_size;
  auto checked_four_times_hidden_size = base::CheckedNumeric(hidden_size) * 4;
  if (!checked_four_times_hidden_size.AssignIfValid(&four_times_hidden_size)) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size is too large."));
  }

  // Validate the input operand.
  if (!context_properties.data_type_limits.lstm_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.lstm_input)));
  }

  // TODO(crbug.com/329482489): Support dynamic shapes.
  const auto input_dimensions = input.StaticShape();
  if (!input_dimensions.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Dynamic input is not supported."));
  }
  if ((*input_dimensions)[0] != steps) {
    return base::unexpected(ErrorWithLabel(
        label, "The input dimensions[0] must be equal to the steps."));
  }

  const uint32_t batch_size = (*input_dimensions)[1];
  const uint32_t input_size = (*input_dimensions)[2];
  const uint32_t direction_count =
      attributes.direction == RecurrentNetworkDirection::kBoth ? 2 : 1;

  // Validate the weight operand.
  if (!context_properties.data_type_limits.lstm_input.Supports(weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kWeightParam, weight,
                   context_properties.data_type_limits.lstm_input)));
  }
  uint32_t expected_weight_shape[3] = {direction_count, four_times_hidden_size,
                                       input_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      weight, kWeightParam, expected_weight_shape, input.data_type(), label));

  // Validate the recurrent weight operand.
  if (!context_properties.data_type_limits.lstm_input.Supports(
          recurrent_weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kRecurrentWeightParam, recurrent_weight,
                   context_properties.data_type_limits.lstm_input)));
  }
  uint32_t expected_recurrent_weight_shape[3] = {
      direction_count, four_times_hidden_size, hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      recurrent_weight, kRecurrentWeightParam, expected_recurrent_weight_shape,
      input.data_type(), label));

  // Validate the bias operand.
  if (attributes.bias) {
    if (!context_properties.data_type_limits.lstm_bias.Supports(
            attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kBiasParam, attributes.bias.value(),
                     context_properties.data_type_limits.lstm_bias)));
    }
    uint32_t expected_bias_shape[2] = {direction_count, four_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.bias.value(), kBiasParam, expected_bias_shape,
        input.data_type(), label));
  }

  // Validate the recurrent bias operand.
  if (attributes.recurrent_bias) {
    if (!context_properties.data_type_limits.lstm_bias.Supports(
            attributes.recurrent_bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kRecurrentBiasParam, attributes.recurrent_bias.value(),
                     context_properties.data_type_limits.lstm_bias)));
    }
    uint32_t expected_recurrent_bias_shape[2] = {direction_count,
                                                 four_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.recurrent_bias.value(), kRecurrentBiasParam,
        expected_recurrent_bias_shape, input.data_type(), label));
  }

  // Validate the peephole weight operand.
  if (attributes.peephole_weight) {
    if (!context_properties.data_type_limits.lstm_bias.Supports(
            attributes.peephole_weight.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kPeepholeWeightParam, attributes.peephole_weight.value(),
                     context_properties.data_type_limits.lstm_bias)));
    }
    // Here `3 * hidden_size` will not overflow because `4 * hidden_size` has
    // already been checked.
    uint32_t expected_peephole_weight_shape[2] = {direction_count,
                                                  3 * hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.peephole_weight.value(), kPeepholeWeightParam,
        expected_peephole_weight_shape, input.data_type(), label));
  }

  // Validate the initial hidden state operand.
  if (attributes.initial_hidden_state) {
    if (!context_properties.data_type_limits.lstm_input.Supports(
            attributes.initial_hidden_state.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kInitialHiddenStateParam, attributes.initial_hidden_state.value(),
              context_properties.data_type_limits.lstm_input)));
    }
    uint32_t expected_initial_hidden_state_shape[3] = {direction_count,
                                                       batch_size, hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.initial_hidden_state.value(), kInitialHiddenStateParam,
        expected_initial_hidden_state_shape, input.data_type(), label));
  }

  // Validate the initial cell state operand.
  if (attributes.initial_cell_state) {
    if (!context_properties.data_type_limits.lstm_input.Supports(
            attributes.initial_cell_state.value())) {
      return base::unexpected(ErrorWithLabel(
          label,
          NotSupportedArgumentError(
              kInitialCellStateParam, attributes.initial_cell_state.value(),
              context_properties.data_type_limits.lstm_input)));
    }
    uint32_t expected_initial_cell_state_shape[3] = {direction_count,
                                                     batch_size, hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.initial_cell_state.value(), kInitialCellStateParam,
        expected_initial_cell_state_shape, input.data_type(), label));
  }

  if (attributes.activation_count != 3) {
    return base::unexpected(ErrorWithLabel(
        label, "The activations should be a sequence of length 3."));
  }

  std::vector<OperandDescriptor> outputs;
  ASSIGN_OR_RETURN(
      OperandDescriptor output,
      OperandDescriptor::Create(
          context_properties, input.data_type(),
          std::array{direction_count, batch_size, hidden_size}, label));
  outputs.push_back(output);
  outputs.push_back(std::move(output));
  if (attributes.return_sequence) {
    ASSIGN_OR_RETURN(
        OperandDescriptor return_sequence_output,
        OperandDescriptor::Create(
            context_properties, input.data_type(),
            std::array{steps, direction_count, batch_size, hidden_size},
            label));
    outputs.push_back(std::move(return_sequence_output));
  }

  return outputs;
}

base::expected<std::vector<OperandDescriptor>, std::string>
ValidateLstmCellAndInferOutput(const ContextProperties& context_properties,
                               const OperandDescriptor& input,
                               const OperandDescriptor& weight,
                               const OperandDescriptor& recurrent_weight,
                               const OperandDescriptor& hidden_state,
                               const OperandDescriptor& cell_state,
                               const uint32_t hidden_size,
                               const LstmCellAttributes& attributes) {
  const std::string& label = attributes.label;
  if (hidden_size <= 0) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size must be greater than 0."));
  }

  uint32_t four_times_hidden_size;
  auto checked_four_times_hidden_size = base::CheckedNumeric(hidden_size) * 4;
  if (!checked_four_times_hidden_size.AssignIfValid(&four_times_hidden_size)) {
    return base::unexpected(
        ErrorWithLabel(label, "The hidden size is too large."));
  }

  // Validate the input operand.
  if (!context_properties.data_type_limits.lstm_cell_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.lstm_cell_input)));
  }

  // TODO(crbug.com/329482489): Support dynamic shapes.
  const auto input_shape = input.StaticShape();
  if (!input_shape.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Dyanmic input is not supported."));
  }
  const uint32_t batch_size = (*input_shape)[0];
  const uint32_t input_size = (*input_shape)[1];

  // Validate the weight operand.
  if (!context_properties.data_type_limits.lstm_cell_input.Supports(weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kWeightParam, weight,
                   context_properties.data_type_limits.lstm_cell_input)));
  }
  std::array<uint32_t, 2> expected_weight_shape = {four_times_hidden_size,
                                                   input_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      weight, kWeightParam, expected_weight_shape, input.data_type(), label));

  // Validate the hidden state operand.
  if (!context_properties.data_type_limits.lstm_cell_input.Supports(
          hidden_state)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kHiddenStateParam, hidden_state,
                   context_properties.data_type_limits.lstm_cell_input)));
  }
  std::array<uint32_t, 2> expected_hidden_state_shape = {batch_size,
                                                         hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      hidden_state, kHiddenStateParam, expected_hidden_state_shape,
      input.data_type(), label));

  // Validate the cell state operand.
  if (!context_properties.data_type_limits.lstm_cell_input.Supports(
          cell_state)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kCellStateParam, cell_state,
                   context_properties.data_type_limits.lstm_cell_input)));
  }
  std::array<uint32_t, 2> expected_cell_state_shape = {batch_size, hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(cell_state, kCellStateParam,
                                                  expected_cell_state_shape,
                                                  input.data_type(), label));

  // Validate the recurrent weight operand.
  if (!context_properties.data_type_limits.lstm_cell_input.Supports(
          recurrent_weight)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kRecurrentWeightParam, recurrent_weight,
                   context_properties.data_type_limits.lstm_cell_input)));
  }
  std::array<uint32_t, 2> expected_recurrent_weight_shape = {
      four_times_hidden_size, hidden_size};
  RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
      recurrent_weight, kRecurrentWeightParam, expected_recurrent_weight_shape,
      input.data_type(), label));

  // Validate the bias operand.
  if (attributes.bias) {
    if (!context_properties.data_type_limits.lstm_cell_bias.Supports(
            attributes.bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kBiasParam, attributes.bias.value(),
                     context_properties.data_type_limits.lstm_cell_bias)));
    }
    std::array<uint32_t, 1> expected_bias_shape = {four_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.bias.value(), kBiasParam, expected_bias_shape,
        input.data_type(), label));
  }

  // Validate the recurrent bias operand.
  if (attributes.recurrent_bias) {
    if (!context_properties.data_type_limits.lstm_cell_bias.Supports(
            attributes.recurrent_bias.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kRecurrentBiasParam, attributes.recurrent_bias.value(),
                     context_properties.data_type_limits.lstm_cell_bias)));
    }
    std::array<uint32_t, 1> expected_recurrent_bias_shape = {
        four_times_hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.recurrent_bias.value(), kRecurrentBiasParam,
        expected_recurrent_bias_shape, input.data_type(), label));
  }

  // Validate the peephole weight operand.
  if (attributes.peephole_weight) {
    if (!context_properties.data_type_limits.lstm_cell_bias.Supports(
            attributes.peephole_weight.value())) {
      return base::unexpected(ErrorWithLabel(
          label, NotSupportedArgumentError(
                     kPeepholeWeightParam, attributes.peephole_weight.value(),
                     context_properties.data_type_limits.lstm_cell_bias)));
    }
    // Here `3 * hidden_size` will not overflow because `4 * hidden_size` has
    // already been checked.
    std::array<uint32_t, 1> expected_peephole_weight_shape = {3 * hidden_size};
    RETURN_IF_ERROR(ValidateRecurrentNetworkOperand(
        attributes.peephole_weight.value(), kPeepholeWeightParam,
        expected_peephole_weight_shape, input.data_type(), label));
  }

  if (attributes.activation_count != 3) {
    return base::unexpected(ErrorWithLabel(
        label, "The activations should be a sequence of length 3."));
  }

  std::vector<OperandDescriptor> outputs;
  outputs.reserve(2);

  ASSIGN_OR_RETURN(
      OperandDescriptor output,
      OperandDescriptor::Create(context_properties, input.data_type(),
                                std::array{batch_size, hidden_size}, label));
  outputs.push_back(output);
  outputs.push_back(std::move(output));

  return outputs;
}

base::expected<OperandDescriptor, std::string> ValidateMatmulAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& a,
    const OperandDescriptor& b,
    std::string_view label) {
  if (!context_properties.data_type_limits.matmul_input.Supports(a)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   "a", a, context_properties.data_type_limits.matmul_input)));
  }

  if (!context_properties.data_type_limits.matmul_input.Supports(b)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   "b", b, context_properties.data_type_limits.matmul_input)));
  }

  if (a.data_type() != b.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The data types of first two inputs don't match."));
  }

  // matmul broadcasts the batch dimensions, so the output rank follows the
  // inputs. When either input is unranked the rank is unknown; defer the
  // shape-compatibility checks to dispatch by propagating the unranked state.
  if (!a.HasRank() || !b.HasRank()) {
    return OperandDescriptor::CreateUnranked(a.data_type());
  }

  std::vector<webnn::Dimension> a_dimensions = a.shape();
  CHECK_GE(a_dimensions.size(), 2u);
  std::vector<webnn::Dimension> b_dimensions = b.shape();
  CHECK_GE(b_dimensions.size(), 2u);

  // The number of columns in the first matrix must be equal to the number of
  // rows in the second matrix. When either dimension is dynamic, skip this
  // check at build time - it will be validated at dispatch time.
  const webnn::Dimension a_cols = a_dimensions[a_dimensions.size() - 1];
  const webnn::Dimension a_rows = a_dimensions[a_dimensions.size() - 2];
  const webnn::Dimension b_cols = b_dimensions[b_dimensions.size() - 1];
  const webnn::Dimension b_rows = b_dimensions[b_dimensions.size() - 2];
  if (DimensionsAreDefinitelyUnequal(a_cols, b_rows)) {
    auto to_string = [](const webnn::Dimension& dim) -> std::string {
      if (std::holds_alternative<uint32_t>(dim)) {
        return base::StringPrintf("%u", std::get<uint32_t>(dim));
      }
      return std::get<webnn::DynamicDimension>(dim).name.value_or("?");
    };
    return base::unexpected(ErrorWithLabel(
        label,
        base::StringPrintf(
            "The number of columns (%s) in the first matrix isn't equal to "
            "the number of rows (%s) in the second matrix.",
            to_string(a_cols).c_str(), to_string(b_rows).c_str())));
  }

  size_t output_rank = std::max(a_dimensions.size(), b_dimensions.size());
  std::vector<Dimension> output_dimensions;
  // Figure out the output shape by broadcasting all the dimensions except the
  // last two. The output is 2-D tensor of shape [M, N].
  if (a.Rank().value() > 2 && b.Rank().value() > 2) {
    std::vector<webnn::Dimension> sliced_a_dimensions(a_dimensions.begin(),
                                                      a_dimensions.end() - 2);
    std::vector<webnn::Dimension> sliced_b_dimensions(b_dimensions.begin(),
                                                      b_dimensions.end() - 2);
    std::optional<std::vector<webnn::Dimension>> optional_output_dimensions =
        BroadcastShapes(sliced_a_dimensions, sliced_b_dimensions, true);
    if (!optional_output_dimensions) {
      return base::unexpected(ErrorWithLabel(
          label, "The matmul input shapes are not broadcastable."));
    }
    output_dimensions.assign(optional_output_dimensions->begin(),
                             optional_output_dimensions->end());
    output_dimensions.push_back(a_rows);
    output_dimensions.push_back(b_cols);
  } else if (a.Rank().value() == 2 && b.Rank().value() == 2) {
    output_dimensions.push_back(a_rows);
    output_dimensions.push_back(b_cols);
  } else {
    const std::vector<webnn::Dimension>& larger_dimensions =
        a_dimensions.size() > b_dimensions.size() ? a_dimensions : b_dimensions;
    output_dimensions.assign(larger_dimensions.begin(),
                             larger_dimensions.end());
    output_dimensions[output_rank - 2] = a_rows;
    output_dimensions[output_rank - 1] = b_cols;
  }
  CHECK_EQ(output_rank, output_dimensions.size());

  return OperandDescriptor::Create(context_properties, a.data_type(),
                                   output_dimensions, label);
}

base::expected<OperandDescriptor, std::string> ValidatePadAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    base::span<const uint32_t> beginning_padding,
    base::span<const uint32_t> ending_padding,
    PaddingMode mode,
    std::string_view label) {
  if (!context_properties.data_type_limits.pad_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.pad_input)));
  }

  // The output rank equals the input rank, and the padding arrays are validated
  // against it. With an unranked input neither can be checked at build time;
  // defer to dispatch by propagating the unranked state.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  // Validate the beginning_padding and ending_padding.
  if (beginning_padding.size() != input.Rank().value()) {
    return base::unexpected(
        ErrorWithLabel(label,
                       "The length of beginningPadding must be "
                       "equal to the rank of the input tensor."));
  }
  if (ending_padding.size() != input.Rank().value()) {
    return base::unexpected(
        ErrorWithLabel(label,
                       "The length of endingPadding must be "
                       "equal to the rank of the input tensor."));
  }

  // Validate padding size restrictions for reflection mode. Reject only on a
  // definite violation (padding >= the dimension's static size); a dynamic
  // dimension defers this check to dispatch.
  switch (mode) {
    case PaddingMode::kConstant:
    case PaddingMode::kEdge: {
      // Do nothing.
      break;
    }
    case PaddingMode::kReflection: {
      for (size_t i = 0; i < input.Rank().value(); ++i) {
        const uint32_t* static_size = std::get_if<uint32_t>(&input.shape()[i]);
        if (!static_size) {
          continue;
        }
        if (beginning_padding[i] >= *static_size) {
          return base::unexpected(ErrorWithLabel(
              label,
              base::StringPrintf("The beginningPadding size (%u) must be less "
                                 "than input size (%u) of dimension (%zu).",
                                 beginning_padding[i], *static_size, i)));
        }
        if (ending_padding[i] >= *static_size) {
          return base::unexpected(ErrorWithLabel(
              label,
              base::StringPrintf("The endingPadding size (%u) must be less "
                                 "than input size (%u) of dimension (%zu).",
                                 ending_padding[i], *static_size, i)));
        }
      }
    }
  }

  // Infer the output. Each output dimension is:
  //   output_size = beginning_padding + input_size + ending_padding.
  // A static input dimension yields a static output; a dynamic input dimension
  // yields a dynamic (unnamed) output, since the size is shifted by the
  // padding.
  std::vector<Dimension> output_shape;
  output_shape.reserve(input.Rank().value());
  for (size_t i = 0; i < input.Rank().value(); ++i) {
    const Dimension& input_dimension = input.shape()[i];
    base::CheckedNumeric<uint32_t> checked_total_padding =
        base::CheckedNumeric<uint32_t>(beginning_padding[i]) +
        ending_padding[i];

    if (const uint32_t* static_size = std::get_if<uint32_t>(&input_dimension)) {
      uint32_t output_size;
      if (!(checked_total_padding + *static_size).AssignIfValid(&output_size)) {
        return base::unexpected(ErrorWithLabel(
            label, base::StringPrintf(
                       "The padding of dimension (%zu) is too large.", i)));
      }
      output_shape.emplace_back(output_size);
      continue;
    }

    // Dynamic dimension. With zero total padding the dimension passes through
    // unchanged and keeps its name (1:1 pass-through). With padding it becomes
    // a different unknown, so emit an unnamed dynamic dimension.
    const DynamicDimension& dynamic_dimension =
        std::get<DynamicDimension>(input_dimension);
    if (checked_total_padding.ValueOrDie() == 0) {
      output_shape.emplace_back(dynamic_dimension);
    } else {
      output_shape.emplace_back(DynamicDimension{});
    }
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidatePool2dAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const Pool2dAttributes& attributes,
    Pool2dKind kind) {
  const std::string& label = attributes.label;
  // Validate input operand and set its sizes.
  const SupportedTensors& tensor_constraint = [&](Pool2dKind kind) {
    switch (kind) {
      case Pool2dKind::kAverage:
        return context_properties.data_type_limits.average_pool2d_input;
      case Pool2dKind::kL2:
        return context_properties.data_type_limits.l2_pool2d_input;
      case Pool2dKind::kMax:
        return context_properties.data_type_limits.max_pool2d_input;
    }
  }(kind);

  if (!tensor_constraint.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(input, tensor_constraint)));
  }

  // pool2d is defined on a fixed 4-D input; it cannot operate on an operand of
  // unknown rank.
  if (!input.HasRank()) {
    return base::unexpected(
        ErrorWithLabel(label, kUnrankedInputNotSupportedError));
  }

  CHECK_EQ(input.shape().size(), 4u);
  // The layout option specifies the layout format of the input tensor.
  Dimension input_batches, input_channels, input_height, input_width;
  switch (attributes.layout) {
    case InputOperandLayout::kNchw:
      // "nchw": [batches, channels, height, width]
      input_batches = input.shape()[0];
      input_channels = input.shape()[1];
      input_height = input.shape()[2];
      input_width = input.shape()[3];
      break;
    case InputOperandLayout::kNhwc:
      // "nhwc": [batches, height, width, channels]
      input_batches = input.shape()[0];
      input_height = input.shape()[1];
      input_width = input.shape()[2];
      input_channels = input.shape()[3];
      break;
  }

  // Validate windowDimensions and get its values. If not present, assume global
  // pool2d operation and the output sizes are 1x1.
  Dimension output_height = uint32_t{1};
  Dimension output_width = uint32_t{1};
  if (attributes.window_dimensions) {
    if (attributes.window_dimensions->height == 0 ||
        attributes.window_dimensions->width == 0) {
      return base::unexpected(ErrorWithLabel(
          label, "All window dimensions should be greater than 0."));
    }
    uint32_t window_height = attributes.window_dimensions->height;
    uint32_t window_width = attributes.window_dimensions->width;

    const uint32_t* input_height_static = std::get_if<uint32_t>(&input_height);
    const uint32_t* input_width_static = std::get_if<uint32_t>(&input_width);

    if (attributes.output_sizes) {
      if (!input_height_static || !input_width_static) {
        return base::unexpected(
            ErrorWithLabel(label,
                           "Dyanmic input sizes is not supported when output "
                           "sizes is present"));
      }
      // Reuse ValidateAndCalculateConv2dOutputSizes to calculate pool2d output
      // sizes.
      ASSIGN_OR_RETURN(Size2d<double> output_sizes,
                       ValidateAndCalculateConv2dOutputSizes(
                           *input_height_static, *input_width_static,
                           window_height, window_width, attributes.padding,
                           attributes.strides, attributes.dilations, label));
      const uint32_t floor_output_height =
          base::ClampFloor<uint32_t>(output_sizes.height);
      const uint32_t ceil_output_height =
          base::ClampCeil<uint32_t>(output_sizes.height);
      const uint32_t floor_output_width =
          base::ClampFloor<uint32_t>(output_sizes.width);
      const uint32_t ceil_output_width =
          base::ClampCeil<uint32_t>(output_sizes.width);

      auto& output_size = attributes.output_sizes.value();
      if (output_size.height == 0 || output_size.width == 0) {
        return base::unexpected(ErrorWithLabel(
            label, "All output sizes should be greater than 0."));
      }
      uint32_t user_output_height = output_size.height;
      uint32_t user_output_width = output_size.width;

      // Check whether the user supplied output sizes is either floor or ceil
      // rounding of the calculated output sizes. The backend implementation
      // should check whether the indicated rounding type is supported.
      if ((user_output_height == floor_output_height &&
           user_output_width == floor_output_width) ||
          (user_output_height == ceil_output_height &&
           user_output_width == ceil_output_width)) {
        output_height = user_output_height;
        output_width = user_output_width;
      } else {
        return base::unexpected(ErrorWithLabel(
            label,
            (floor_output_height == ceil_output_height &&
             floor_output_width == ceil_output_width)
                ? base::StringPrintf("The output sizes should be [%u, %u].",
                                     floor_output_height, floor_output_width)
                : base::StringPrintf(
                      "The output sizes should be either [%u, %u] or [%u, %u].",
                      floor_output_height, floor_output_width,
                      ceil_output_height, ceil_output_width)));
      }
    } else {
      // Compute each spatial dimension independently. A static dimension is
      // reduced to a concrete output size (per the rounding type); a dynamic
      // one yields an unnamed dynamic output dimension (resolved at dispatch).
      const bool use_ceil = attributes.rounding_type == RoundingType::kCeil;
      if (input_height_static) {
        ASSIGN_OR_RETURN(
            double float_output_h,
            CalculateConv2dOutputSize(*input_height_static, window_height,
                                      attributes.padding.beginning.height,
                                      attributes.padding.ending.height,
                                      attributes.strides.height,
                                      attributes.dilations.height, label));
        output_height = use_ceil ? base::ClampCeil<uint32_t>(float_output_h)
                                 : base::ClampFloor<uint32_t>(float_output_h);
      } else {
        output_height = DynamicDimension{};
      }
      if (input_width_static) {
        ASSIGN_OR_RETURN(
            double float_output_w,
            CalculateConv2dOutputSize(*input_width_static, window_width,
                                      attributes.padding.beginning.width,
                                      attributes.padding.ending.width,
                                      attributes.strides.width,
                                      attributes.dilations.width, label));
        output_width = use_ceil ? base::ClampCeil<uint32_t>(float_output_w)
                                : base::ClampFloor<uint32_t>(float_output_w);
      } else {
        output_width = DynamicDimension{};
      }
    }
  }  // end if (attributes.window_dimensions)

  // The layout option specifies the layout format of the output tensor.
  std::vector<Dimension> output_shape;
  switch (attributes.layout) {
    case InputOperandLayout::kNchw:
      // "nchw": [batches, channels, height, width]
      output_shape = {input_batches, input_channels, output_height,
                      output_width};
      break;
    case InputOperandLayout::kNhwc:
      // "nhwc": [batches, height, width, channels]
      output_shape = {input_batches, output_height, output_width,
                      input_channels};
      break;
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidatePreluAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& slope,
    std::string_view label) {
  if (!context_properties.data_type_limits.prelu_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.prelu_input)));
  }

  if (!context_properties.data_type_limits.prelu_input.Supports(slope)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kSlopeParam, slope,
                   context_properties.data_type_limits.prelu_input)));
  }

  // The output has the same shape as the input; when the input is unranked the
  // slope broadcast cannot be checked, so defer to dispatch.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  if (input.data_type() != slope.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The data type of slope doesn't match the data type of input."));
  }
  // TODO(crbug.com/387892103): Use bidirectional broadcasting.
  // BroadcastShape unidirectionally broadcasts slope.dimensions to
  // input.dimensions.
  const auto slope_shape = slope.StaticShape();
  if (!slope_shape.has_value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Dynamic shape is not supported for slope."));
  }
  if (!BroadcastShapes(slope.shape(), input.shape(),
                       /*bidirectional=*/false)) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The shape of slope is not broadcastable to the shape of input."));
  }

  return input;
}

base::expected<OperandDescriptor, std::string> ValidateReduceAndInferOutput(
    const ContextProperties& context_properties,
    ReduceKind kind,
    const OperandDescriptor& input,
    std::string_view label,
    base::span<const uint32_t> axes,
    bool keep_dimensions) {
  const SupportedTensors& tensor_constraint = [&](ReduceKind kind) {
    switch (kind) {
      case ReduceKind::kL1:
        return context_properties.data_type_limits.reduce_l1_input;
      case ReduceKind::kL2:
        return context_properties.data_type_limits.reduce_l2_input;
      case ReduceKind::kLogSum:
        return context_properties.data_type_limits.reduce_log_sum_input;
      case ReduceKind::kLogSumExp:
        return context_properties.data_type_limits.reduce_log_sum_exp_input;
      case ReduceKind::kMax:
        return context_properties.data_type_limits.reduce_max_input;
      case ReduceKind::kMean:
        return context_properties.data_type_limits.reduce_mean_input;
      case ReduceKind::kMin:
        return context_properties.data_type_limits.reduce_min_input;
      case ReduceKind::kProduct:
        return context_properties.data_type_limits.reduce_product_input;
      case ReduceKind::kSum:
        return context_properties.data_type_limits.reduce_sum_input;
      case ReduceKind::kSumSquare:
        return context_properties.data_type_limits.reduce_sum_square_input;
    }
  }(kind);

  if (!tensor_constraint.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(input, tensor_constraint)));
  }

  // The axes are validated against the input rank and determine the output
  // shape; with an unranked input neither is known, so defer to dispatch.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  ASSIGN_OR_RETURN(std::vector<Dimension> output_shape,
                   ValidateReduceAxesAndInferOutput(input.shape(), axes,
                                                    keep_dimensions, label));

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

// The current WebNN spec doesn't define the calculation formula of the output
// size for resample2d. An issue has been filed to track it -
// https://github.com/webmachinelearning/webnn/issues/360.
base::expected<uint32_t, std::string> CalculateResample2dOutputSize(
    const uint32_t input_size,
    const float scale,
    std::string_view label) {
  // Calculate the output size in double precision floating point number that
  // ensures values of type uint32_t can be exactly represented.
  // https://en.wikipedia.org/wiki/Double-precision_floating-point_format#Precision_limitations_on_integer_values
  auto checked_output_size = base::CheckedNumeric<double>(input_size) * scale;

  // Check if the value is valid for rounding to uint32_t type.
  if (!checked_output_size.IsValid<uint32_t>()) {
    return base::unexpected(ErrorWithLabel(label, "The scale is too large."));
  }
  const uint32_t output_size = base::ClampFloor<uint32_t>(
      static_cast<double>(checked_output_size.ValueOrDie()));
  if (output_size == 0) {
    return base::unexpected(ErrorWithLabel(label, "The scale is too small."));
  }
  return output_size;
}

base::expected<OperandDescriptor, std::string> ValidateResample2dAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const std::variant<base::span<const float>, base::span<const uint32_t>>&
        scales_or_sizes,
    base::span<const uint32_t> axes,
    std::string_view label) {
  if (!context_properties.data_type_limits.resample2d_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.resample2d_input)));
  }

  // resample2d resizes along two axes of a fixed-rank input; it cannot operate
  // on an operand of unknown rank.
  if (!input.HasRank()) {
    return base::unexpected(
        ErrorWithLabel(label, kUnrankedInputNotSupportedError));
  }

  if (axes.size() != 2) {
    return base::unexpected(
        ErrorWithLabel(label, "The length of axes should be 2."));
  }
  RETURN_IF_ERROR(ValidateAxes(axes, input.Rank().value(), label));

  // Non-resample axes pass through (preserving any dynamic-ness); the two
  // resample axes are overwritten below based on scales or sizes.
  std::vector<Dimension> output_shape = input.shape();

  // Helper: compute the output dim along one resample axis given a scale. A
  // static input dim yields a concrete output size; a dynamic one yields an
  // unnamed dynamic output dim (resolved at dispatch).
  auto resample_dim_from_scale =
      [&](const Dimension& in_dim,
          float scale) -> base::expected<Dimension, std::string> {
    if (std::holds_alternative<uint32_t>(in_dim)) {
      ASSIGN_OR_RETURN(uint32_t out,
                       CalculateResample2dOutputSize(
                           std::get<uint32_t>(in_dim), scale, label));
      return Dimension{out};
    }
    return Dimension{DynamicDimension{}};
  };

  // Validate scales or sizes and infer the output.
  if (std::holds_alternative<base::span<const float>>(scales_or_sizes)) {
    const auto& scales = std::get<base::span<const float>>(scales_or_sizes);
    if (scales.size() != 2) {
      return base::unexpected(
          ErrorWithLabel(label, "The length of scales should be 2."));
    }
    if (scales[0] <= 0 || scales[1] <= 0) {
      return base::unexpected(
          ErrorWithLabel(label, "All scales should be greater than 0."));
    }

    ASSIGN_OR_RETURN(
        output_shape[axes[0]],
        resample_dim_from_scale(input.shape()[axes[0]], scales[0]));
    ASSIGN_OR_RETURN(
        output_shape[axes[1]],
        resample_dim_from_scale(input.shape()[axes[1]], scales[1]));
  } else if (std::holds_alternative<base::span<const uint32_t>>(
                 scales_or_sizes)) {
    const auto& sizes = std::get<base::span<const uint32_t>>(scales_or_sizes);
    if (sizes.size() != 2) {
      return base::unexpected(
          ErrorWithLabel(label, "The length of sizes should be 2."));
    }
    if (sizes[0] == 0 || sizes[1] == 0) {
      return base::unexpected(
          ErrorWithLabel(label, "All sizes should be greater than 0."));
    }

    // Sizes directly specify the output dims on the resample axes; they are
    // always static regardless of whether the corresponding input dims are
    // dynamic. However, backends that materialize a full sizes tensor for
    // Resize require the non-resample axes to be static.
    for (uint32_t i = 0; i < input.Rank().value(); ++i) {
      if (i == axes[0] || i == axes[1]) {
        continue;
      }
      if (std::holds_alternative<DynamicDimension>(input.shape()[i])) {
        return base::unexpected(ErrorWithLabel(
            label,
            "Dynamic dimension on non-resample axes is not supported when "
            "sizes are provided."));
      }
    }
    output_shape[axes[0]] = sizes[0];
    output_shape[axes[1]] = sizes[1];
  } else {
    NOTREACHED();
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidateReverseAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    base::span<const uint32_t> axes,
    std::string_view label) {
  if (!context_properties.data_type_limits.reverse_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.reverse_input)));
  }

  // The output has the same shape as the input; when the input is unranked the
  // axes cannot be range-checked, so defer to dispatch.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  RETURN_IF_ERROR(ValidateAxes(axes, input.Rank().value(), label));

  return input;
}

base::expected<OperandDescriptor, std::string> ValidateShapeAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    std::string_view label) {
  if (!context_properties.data_type_limits.shape_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.shape_input)));
  }

  // WebNN dimensions are uint32, so the shape output is a uint32 tensor. The
  // backend casts to int64 when lowering to ONNX (whose Shape op emits int64).
  if (!context_properties.data_type_limits.shape_output.data_types.Has(
          OperandDataType::kUint32)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedOpOutputTypeError(
            OperandDataType::kUint32,
            context_properties.data_type_limits.shape_output.data_types)));
  }

  // Output is a 1-D tensor with size equal to the rank of the input. When the
  // input is unranked the size is unknown, so the single output dimension is an
  // unnamed dynamic dimension resolved at dispatch (the output rank is always
  // 1).
  if (!input.HasRank()) {
    std::array<Dimension, 1> dynamic_output_shape{DynamicDimension{}};
    return OperandDescriptor::Create(context_properties,
                                     OperandDataType::kUint32,
                                     dynamic_output_shape, label);
  }

  // Output is a 1-D tensor with size equal to the rank of the input.
  std::vector<uint32_t> output_shape{input.Rank().value()};
  return OperandDescriptor::Create(context_properties, OperandDataType::kUint32,
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string>
ValidateScatterElementsAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& indices,
    const OperandDescriptor& updates,
    const uint32_t axis,
    std::string_view label) {
  if (!context_properties.data_type_limits.scatter_elements_input.Supports(
          input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input,
            context_properties.data_type_limits.scatter_elements_input)));
  }

  if (!context_properties.data_type_limits.scatter_elements_indices.Supports(
          indices)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedArgumentError(
            kIndicesParam, indices,
            context_properties.data_type_limits.scatter_elements_indices)));
  }

  if (input.data_type() != updates.data_type()) {
    return base::unexpected(
        ErrorWithLabel(label,
                       "The updates tensor data type should be the same as "
                       "input data type."));
  }

  // The output has the same shape as the input; when any operand is unranked
  // the rank-dependent checks cannot run, so defer to dispatch.
  if (!input.HasRank() || !indices.HasRank() || !updates.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  if (input.Rank().value() <= axis) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The axis must be in the range [0, N-1] where N is the rank of input "
        "tensor."));
  }

  if (indices.Rank().value() != input.Rank().value()) {
    return base::unexpected(ErrorWithLabel(
        label, "The indices and input tensors should have the same rank."));
  }

  for (uint32_t i = 0; i < input.Rank().value(); ++i) {
    if (i == axis) {
      continue;
    }
    if (DimensionsAreDefinitelyUnequal(input.shape()[i], indices.shape()[i])) {
      return base::unexpected(
          ErrorWithLabel(label,
                         "Except on the axis dimension, the input and indices "
                         "tensor must have the same dimension size."));
    }
  }

  // The updates tensor must have the same shape as indices. When any pair of
  // dims is dynamic, defer the equality check to dispatch time.
  if (indices.shape().size() != updates.shape().size()) {
    return base::unexpected(ErrorWithLabel(
        label, "The updates and indices tensors should have the same shape."));
  }
  for (size_t i = 0; i < indices.shape().size(); ++i) {
    if (DimensionsAreDefinitelyUnequal(indices.shape()[i],
                                       updates.shape()[i])) {
      return base::unexpected(ErrorWithLabel(
          label,
          "The updates and indices tensors should have the same shape."));
    }
  }

  // The output tensor has the same data type and shape as input's.
  return input;
}

base::expected<OperandDescriptor, std::string> ValidateScatterNDAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const OperandDescriptor& indices,
    const OperandDescriptor& updates,
    std::string_view label) {
  if (!context_properties.data_type_limits.scatter_nd_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.scatter_nd_input)));
  }

  if (!context_properties.data_type_limits.scatter_nd_indices.Supports(
          indices)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kIndicesParam, indices,
                   context_properties.data_type_limits.scatter_nd_indices)));
  }

  if (!context_properties.data_type_limits.scatter_nd_updates.Supports(
          updates)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kUpdatesParam, updates,
                   context_properties.data_type_limits.scatter_nd_updates)));
  }

  // Updates tensor's data type should be the same as input's.
  if (input.data_type() != updates.data_type()) {
    return base::unexpected(
        ErrorWithLabel(label,
                       "The updates tensor data type should be the same as "
                       "input data type."));
  }

  // The output has the same shape as the input; when the input, indices, or
  // updates is unranked the rank-dependent checks cannot run, so defer to
  // dispatch. updates is dereferenced below (via its data type and, at
  // dispatch, its shape), so it must be guarded here too.
  if (!input.HasRank() || !indices.HasRank() || !updates.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  // The last dimension of indices must be static because the output rank
  // depends on it.
  const Dimension& indices_last_dim =
      indices.shape()[indices.Rank().value() - 1];
  if (!std::holds_alternative<uint32_t>(indices_last_dim)) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The last dimension of indices must be static because the output rank "
        "depends on it."));
  }

  uint32_t indices_last_dim_size = std::get<uint32_t>(indices_last_dim);
  if (indices_last_dim_size > input.Rank().value()) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf(
                   "The size of the last dimension of indices tensor (%u)"
                   "should not be greater than input rank (%u).",
                   indices_last_dim_size, input.Rank().value())));
  }

  // Validate `updates.shape` =
  // `indices.shape[:-1]` + `input.shape[indices.shape[-1]:]`, where `+` denotes
  // the concatenation of shapes.
  auto checked_updates_rank =
      base::CheckedNumeric<uint32_t>(indices.Rank().value()) - 1 +
      input.Rank().value() - indices_last_dim_size;
  if (!checked_updates_rank.IsValid()) {
    return base::unexpected(
        ErrorWithLabel(label, "The expected updates rank is too large."));
  }

  std::vector<Dimension> expected_updates_shape;
  expected_updates_shape.reserve(checked_updates_rank.ValueOrDie());
  base::span<const Dimension> indices_shape = indices.shape();
  std::ranges::copy(indices_shape.first(indices_shape.size() - 1),
                    std::back_inserter(expected_updates_shape));
  base::span<const Dimension> input_shape = input.shape();
  std::ranges::copy(input_shape.subspan(indices_last_dim_size),
                    std::back_inserter(expected_updates_shape));

  if (expected_updates_shape.size() != updates.shape().size()) {
    return base::unexpected(
        ErrorWithLabel(label, "The updates tensor rank is invalid."));
  }
  // Compare each dimension, skipping pairs where either side is dynamic since
  // dynamic dimension sizes will be validated at dispatch time.
  for (size_t i = 0; i < expected_updates_shape.size(); ++i) {
    if (DimensionsAreDefinitelyUnequal(expected_updates_shape[i],
                                       updates.shape()[i])) {
      return base::unexpected(
          ErrorWithLabel(label, "The updates tensor shape is invalid."));
    }
  }

  // The output tensor has the same data type and shape as input's.
  return input;
}

base::expected<OperandDescriptor, std::string> ValidateSliceAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    const SliceAttributes& attributes) {
  const std::string& label = attributes.label;

  if (!context_properties.data_type_limits.slice_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.slice_input)));
  }

  // The starts/sizes/strides arrays are validated against the input rank and
  // the output rank equals the input rank; with an unranked input none of this
  // is known, so defer to dispatch.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  const auto input_rank = input.Rank().value();

  if (attributes.starts.size() != input_rank) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The length of starts must be equal to the rank of the input tensor."));
  }

  if (attributes.sizes.size() != input_rank) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The length of sizes must be equal to the rank of the input tensor."));
  }

  if (attributes.strides.size() != input_rank) {
    return base::unexpected(
        ErrorWithLabel(label,
                       "The length of strides must be equal to the rank of the "
                       "input tensor."));
  }

  std::vector<uint32_t> output_shape;
  output_shape.reserve(input_rank);

  for (uint32_t i = 0; i < input_rank; ++i) {
    // WebNN plans to allow 0 size dimensions and an issue has been filed to
    // track it: https://github.com/webmachinelearning/webnn/issues/391.
    if (attributes.sizes[i] == 0) {
      return base::unexpected(ErrorWithLabel(
          label, base::StringPrintf(
                     "For dimension (%u): the number of elements to slice "
                     "must not be 0.",
                     i)));
    }

    if (attributes.strides[i] < 1) {
      return base::unexpected(ErrorWithLabel(
          label,
          base::StringPrintf(
              "For dimension (%u): the stride (%u) must not be less than 1.", i,
              attributes.strides[i])));
    }

    auto checked_ending_index =
        base::CheckedNumeric<uint32_t>(attributes.starts[i]) +
        attributes.sizes[i];
    if (!checked_ending_index.IsValid<uint32_t>()) {
      return base::unexpected(ErrorWithLabel(
          label,
          base::StringPrintf(
              "For dimension (%u): the ending index to slice is too large.",
              i)));
    }

    // Range-check the slice window against the input size only when the input
    // dimension is static. A dynamic input dimension defers this check to
    // dispatch, when the concrete size is known.
    if (const uint32_t* dim_size = std::get_if<uint32_t>(&input.shape()[i])) {
      if (attributes.starts[i] >= *dim_size) {
        return base::unexpected(ErrorWithLabel(
            label, base::StringPrintf(
                       "For dimension (%u): the starting index to slice must "
                       "be less than input size (%u).",
                       i, *dim_size)));
      }

      if (checked_ending_index.ValueOrDie() > *dim_size) {
        return base::unexpected(ErrorWithLabel(
            label,
            base::StringPrintf("For dimension (%u): the ending index to slice "
                               "must not be greater than input size (%u).",
                               i, *dim_size)));
      }
    }

    if (attributes.strides[i] > attributes.sizes[i]) {
      return base::unexpected(ErrorWithLabel(
          label,
          base::StringPrintf("For dimension (%u): the stride (%u) must not be "
                             "greater than the size to slice (%u).",
                             i, attributes.strides[i], attributes.sizes[i])));
    }

    uint32_t output_size = attributes.sizes[i] / attributes.strides[i] +
                           (attributes.sizes[i] % attributes.strides[i] != 0);
    output_shape.push_back(output_size);
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidateSoftmaxAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    uint32_t axis,
    std::string_view label) {
  if (!context_properties.data_type_limits.softmax_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.softmax_input)));
  }
  // The output has the same shape as the input; when the input is unranked the
  // axis cannot be range-checked, so defer to dispatch.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }
  if (axis >= input.Rank().value()) {
    return base::unexpected(
        ErrorWithLabel(label, "Axis must be a valid dimension."));
  }
  // The output tensor of softmax is the same shape as the input tensor.
  return input;
}

base::expected<std::vector<OperandDescriptor>, std::string>
ValidateSplitAndInferOutput(const ContextProperties& context_properties,
                            const OperandDescriptor& input,
                            const SplitAttribute& attributes) {
  const std::string& label = attributes.label;
  if (!context_properties.data_type_limits.split_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.split_input)));
  }

  // The split axis is validated against the input rank and each output mirrors
  // the input shape; with an unranked input the axis cannot be range-checked
  // and the per-part sizes cannot be inferred. The number of outputs, however,
  // is fixed by `attributes.splits`, so emit that many unranked outputs and
  // defer the size checks to dispatch.
  if (!input.HasRank()) {
    static_assert(std::variant_size<decltype(attributes.splits)>() == 2,
                  "When adding new variants update the branches below.");
    size_t output_count =
        std::holds_alternative<uint32_t>(attributes.splits)
            ? std::get<uint32_t>(attributes.splits)
            : std::get<base::span<const uint32_t>>(attributes.splits).size();
    if (output_count == 0 || output_count > kMaxValidTensorCount) {
      return base::unexpected(ErrorWithLabel(
          label, base::StringPrintf(
                     "The number of splits must be in the range [1, %u].",
                     kMaxValidTensorCount)));
    }
    std::vector<OperandDescriptor> outputs;
    outputs.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
      outputs.push_back(OperandDescriptor::CreateUnranked(input.data_type()));
    }
    return outputs;
  }

  if (attributes.axis >= input.Rank().value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The axis must be in the range [0, N-1] where N is the rank of the "
        "input tensor."));
  }

  static_assert(std::variant_size<decltype(attributes.splits)>() == 2,
                "When adding new variants update the branches below.");

  // The split-axis size is only needed to (a) derive equal-split part sizes
  // and (b) cross-check explicitly provided per-part sizes. When the axis
  // dimension is dynamic, both are deferred to dispatch time, where the
  // concrete size is known. The equal-split variant still requires a static
  // axis, because the part sizes cannot be derived without it.
  const Dimension& axis_dimension = input.shape()[attributes.axis];
  std::optional<uint32_t> split_dimension_size;
  if (std::holds_alternative<uint32_t>(axis_dimension)) {
    split_dimension_size = std::get<uint32_t>(axis_dimension);
  }

  std::vector<OperandDescriptor> outputs;
  if (std::holds_alternative<uint32_t>(attributes.splits)) {
    uint32_t splits = std::get<uint32_t>(attributes.splits);
    if (splits == 0) {
      return base::unexpected(
          ErrorWithLabel(label, "The splits must be greater than zero."));
    }

    if (splits > kMaxValidTensorCount) {
      return base::unexpected(ErrorWithLabel(
          label,
          base::StringPrintf("The splits must be less than or equal to %u.",
                             kMaxValidTensorCount)));
    }

    // Determine the per-part size along the split axis. With a static axis it
    // is computed directly (and divisibility is verified); with a dynamic axis
    // each part is an unnamed dynamic dimension and the divisibility check is
    // deferred to dispatch where the concrete axis size is known.
    Dimension part_dimension;
    if (split_dimension_size.has_value()) {
      if (*split_dimension_size % splits != 0) {
        return base::unexpected(
            ErrorWithLabel(label,
                           "The dimension size of the input tensor along "
                           "options.axis must be divisible by splits."));
      }
      part_dimension = *split_dimension_size / splits;
    } else {
      part_dimension = DynamicDimension{};
    }

    outputs.reserve(splits);
    for (uint32_t i = 0; i < splits; ++i) {
      // Each of the `splits` outputs has the same shape: the input shape with
      // the split axis replaced by the per-part size.
      std::vector<Dimension> new_dimensions = input.shape();
      new_dimensions[attributes.axis] = part_dimension;
      ASSIGN_OR_RETURN(
          OperandDescriptor split_descriptor,
          OperandDescriptor::Create(context_properties, input.data_type(),
                                    new_dimensions, label));
      outputs.push_back(std::move(split_descriptor));
    }
  } else if (std::holds_alternative<base::span<const uint32_t>>(
                 attributes.splits)) {
    const auto& splits =
        std::get<base::span<const uint32_t>>(attributes.splits);
    if (splits.empty()) {
      return base::unexpected(
          ErrorWithLabel(label, "The splits should not be empty."));
    }

    if (splits.size() > kMaxValidTensorCount) {
      return base::unexpected(ErrorWithLabel(
          label, base::StringPrintf(
                     "The number of splits must be less than or equal to %u.",
                     kMaxValidTensorCount)));
    }

    if (std::ranges::any_of(splits,
                            [](uint32_t split) { return split == 0; })) {
      return base::unexpected(
          ErrorWithLabel(label, "All splits must be greater than zero."));
    }

    // The explicit per-part sizes determine the output shapes directly. Only
    // their sum needs the concrete axis size to be validated; when the axis is
    // dynamic, defer that cross-check to dispatch time.
    if (split_dimension_size.has_value()) {
      base::CheckedNumeric<uint32_t> sum = std::accumulate(
          splits.begin(), splits.end(), base::CheckedNumeric<uint32_t>(0));
      if (!sum.IsValid() || sum.ValueOrDie() != *split_dimension_size) {
        return base::unexpected(ErrorWithLabel(
            label,
            "The sum of all sizes in splits must be equal to the dimension "
            "size of the input tensor specified by options.axis."));
      }
    }

    outputs.reserve(splits.size());
    for (uint32_t split : splits) {
      std::vector<Dimension> new_dimensions = input.shape();
      new_dimensions[attributes.axis] = split;
      auto split_descriptor = OperandDescriptor::Create(
          context_properties, input.data_type(), new_dimensions, label);
      // `split_descriptor` should always be valid, since it's a subset of the
      // input.
      CHECK(split_descriptor.has_value());
      outputs.push_back(*std::move(split_descriptor));
    }
  } else {
    NOTREACHED();
  }

  return outputs;
}

base::expected<OperandDescriptor, std::string> ValidateTileAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    base::span<const uint32_t> repetitions,
    std::string_view label) {
  if (!context_properties.data_type_limits.tile_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.tile_input)));
  }

  // The repetitions array is validated against the input rank and the output
  // rank equals the input rank; with an unranked input neither is known, so
  // defer to dispatch.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  if (repetitions.size() != input.Rank().value()) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The number of values in repetitions must be the same as the rank of "
        "the input tensor."));
  }

  // Infer the output. Each output dimension is repetitions[i] * input[i].
  // A static input dimension yields a static output; a dynamic input dimension
  // yields a dynamic (unnamed) output, since the size is scaled by
  // repetitions[i].
  std::vector<Dimension> output_shape;
  output_shape.reserve(input.Rank().value());
  for (size_t i = 0; i < input.Rank().value(); ++i) {
    if (repetitions[i] == 0) {
      return base::unexpected(
          ErrorWithLabel(label, "Any value in repetitions must not be 0."));
    }
    const Dimension& input_dimension = input.shape()[i];

    if (const uint32_t* static_size = std::get_if<uint32_t>(&input_dimension)) {
      uint32_t output_size;
      if (!(base::CheckedNumeric<uint32_t>(repetitions[i]) * *static_size)
               .AssignIfValid(&output_size)) {
        return base::unexpected(
            ErrorWithLabel(label, "The tiled dimension size is too large."));
      }
      output_shape.emplace_back(output_size);
      continue;
    }

    // Dynamic dimension. A repetition of 1 leaves it unchanged and keeps its
    // name (1:1 pass-through); otherwise it becomes a different unknown, so
    // emit an unnamed dynamic dimension.
    const DynamicDimension& dynamic_dimension =
        std::get<DynamicDimension>(input_dimension);
    if (repetitions[i] == 1) {
      output_shape.emplace_back(dynamic_dimension);
    } else {
      output_shape.emplace_back(DynamicDimension{});
    }
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidateSqueezeAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    std::optional<base::span<const uint32_t>> axes,
    std::string_view label) {
  if (!context_properties.data_type_limits.squeeze_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.squeeze_input)));
  }

  // The no-axes case removes every size-1 dimension. The output rank therefore
  // depends on which dimensions are 1, which is only knowable once every
  // dimension is a concrete static size. When the input is fully static (the
  // common case at computeShapes/dispatch) the result is computed directly;
  // otherwise — an unranked input, or a ranked input with any dynamic dimension
  // — the output rank cannot be determined and is deferred to dispatch by
  // emitting an unranked output. This is the seam where dynamic rank
  // originates.
  if (!axes.has_value()) {
    std::optional<std::vector<uint32_t>> static_shape = input.StaticShape();
    if (!static_shape.has_value()) {
      return OperandDescriptor::CreateUnranked(input.data_type());
    }
    std::vector<Dimension> output_shape;
    for (uint32_t dimension : *static_shape) {
      if (dimension != 1) {
        output_shape.emplace_back(dimension);
      }
    }
    return OperandDescriptor::Create(context_properties, input.data_type(),
                                     output_shape, label);
  }

  // An unranked input cannot have its axes validated against a rank, and the
  // output rank cannot be computed; defer to dispatch by emitting an unranked
  // output.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  RETURN_IF_ERROR(ValidateAxes(*axes, *input.Rank(), label));

  // A listed axis must be provably size 1 to be removed. A dynamic dimension at
  // a listed axis cannot be proven to be 1 nor rejected at build time, so it is
  // deferred to dispatch (the dimension is removed in the inferred output and
  // re-checked once the concrete size is known).
  std::set<uint32_t> axes_to_remove(axes->begin(), axes->end());
  for (uint32_t axis : *axes) {
    const Dimension& dimension = input.shape()[axis];
    if (const uint32_t* static_size = std::get_if<uint32_t>(&dimension)) {
      if (*static_size != 1) {
        return base::unexpected(
            ErrorWithLabel(label, "The dimension to be squeezed must be 1."));
      }
    }
  }

  // Output = input dimensions minus the removed axes, survivors keep their
  // names (1:1 pass-through).
  std::vector<Dimension> output_shape;
  output_shape.reserve(*input.Rank() - axes_to_remove.size());
  for (uint32_t i = 0; i < *input.Rank(); ++i) {
    if (!axes_to_remove.contains(i)) {
      output_shape.push_back(input.shape()[i]);
    }
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidateUnsqueezeAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    base::span<const uint32_t> axes,
    std::string_view label) {
  if (!context_properties.data_type_limits.unsqueeze_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.unsqueeze_input)));
  }

  // An unranked input means the output rank (input rank + |axes|) is unknown;
  // defer to dispatch.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  // The output rank is the input rank plus the number of inserted dimensions.
  // `axes` are expressed in the output axis numbering.
  base::CheckedNumeric<uint32_t> checked_output_rank =
      base::CheckedNumeric<uint32_t>(*input.Rank()) + axes.size();
  uint32_t output_rank;
  if (!checked_output_rank.AssignIfValid(&output_rank)) {
    return base::unexpected(
        ErrorWithLabel(label, "The output rank is too large."));
  }
  RETURN_IF_ERROR(ValidateAxes(axes, output_rank, label));

  // Insert size-1 dimensions at the listed output axes; fill the remaining
  // positions with the input dimensions in order (names preserved).
  std::set<uint32_t> inserted_axes(axes.begin(), axes.end());
  std::vector<Dimension> output_shape;
  output_shape.reserve(output_rank);
  uint32_t input_index = 0;
  for (uint32_t i = 0; i < output_rank; ++i) {
    if (inserted_axes.contains(i)) {
      output_shape.emplace_back(uint32_t{1});
    } else {
      output_shape.push_back(input.shape()[input_index++]);
    }
  }

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidateFlattenAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    uint32_t axis,
    std::string_view label) {
  if (!context_properties.data_type_limits.flatten_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedInputArgumentError(
                   input, context_properties.data_type_limits.flatten_input)));
  }

  // An unranked input means neither product can be computed and `axis` cannot
  // be range-checked; defer to dispatch with an unranked output.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  if (base::MakeStrictNum(axis) > *input.Rank()) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf("The axis must be in the range [0, %u].",
                                  *input.Rank())));
  }

  // Each output dimension is the product of one side of the split. If a side
  // contains any dynamic dimension, the product cannot be known at build time
  // and is emitted as an unnamed dynamic dimension (a derived dim never carries
  // a name). An empty side has a static product of 1.
  auto product = [&](base::span<const Dimension> dims)
      -> base::expected<Dimension, std::string> {
    base::CheckedNumeric<uint32_t> checked_size = 1;
    for (const Dimension& dimension : dims) {
      if (!std::holds_alternative<uint32_t>(dimension)) {
        return Dimension(DynamicDimension{});
      }
      checked_size *= std::get<uint32_t>(dimension);
    }
    uint32_t size;
    if (!checked_size.AssignIfValid(&size)) {
      return base::unexpected(
          ErrorWithLabel(label, "The flattened dimension size is too large."));
    }
    return Dimension(size);
  };

  ASSIGN_OR_RETURN(Dimension first,
                   product(base::span(input.shape()).first(axis)));
  ASSIGN_OR_RETURN(Dimension second,
                   product(base::span(input.shape()).subspan(axis)));

  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   std::vector<Dimension>{first, second},
                                   label);
}

base::expected<OperandDescriptor, std::string> ValidateTransposeAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    base::span<const uint32_t> permutation,
    std::string_view label) {
  if (!context_properties.data_type_limits.transpose_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.transpose_input)));
  }

  // The permutation is validated against the input rank and selects the output
  // dimensions from the input; with an unranked input neither can be done, so
  // defer to dispatch by propagating the unranked state.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  if (permutation.size() != static_cast<size_t>(input.Rank().value())) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The number of values in permutation must be the same as the rank of "
        "the input tensor."));
  }
  RETURN_IF_ERROR(ValidateAxes(permutation, input.Rank().value(), label));

  std::vector<Dimension> output_shape(input.Rank().value());
  for (uint32_t i = 0; i < input.Rank().value(); ++i) {
    output_shape[i] = input.shape()[permutation[i]];
  }
  return OperandDescriptor::Create(context_properties, input.data_type(),
                                   output_shape, label);
}

base::expected<OperandDescriptor, std::string> ValidateTriangularAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& input,
    std::string_view label) {
  if (!context_properties.data_type_limits.triangular_input.Supports(input)) {
    return base::unexpected(ErrorWithLabel(
        label,
        NotSupportedInputArgumentError(
            input, context_properties.data_type_limits.triangular_input)));
  }

  // The output tensor of triangular is the same shape and the same type as the
  // input tensor, so an unranked input propagates to an unranked output.
  if (!input.HasRank()) {
    return OperandDescriptor::CreateUnranked(input.data_type());
  }

  return input;
}

base::expected<OperandDescriptor, std::string> ValidateWhereAndInferOutput(
    const ContextProperties& context_properties,
    const OperandDescriptor& condition,
    const OperandDescriptor& true_value,
    const OperandDescriptor& false_value,
    std::string_view label) {
  if (!context_properties.data_type_limits.where_condition.Supports(
          condition)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kConditionParam, condition,
                   context_properties.data_type_limits.where_condition)));
  }

  if (!context_properties.data_type_limits.where_value.Supports(true_value)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kTrueValueParam, true_value,
                   context_properties.data_type_limits.where_value)));
  }

  if (!context_properties.data_type_limits.where_value.Supports(false_value)) {
    return base::unexpected(ErrorWithLabel(
        label, NotSupportedArgumentError(
                   kFalseValueParam, false_value,
                   context_properties.data_type_limits.where_value)));
  }

  if (true_value.data_type() != false_value.data_type()) {
    return base::unexpected(ErrorWithLabel(
        label, "The data types of trueValue and falseValue don't match."));
  }

  // The output rank is the broadcast of condition/trueValue/falseValue; if any
  // is unranked the broadcast is unknown, so defer to dispatch.
  if (!condition.HasRank() || !true_value.HasRank() || !false_value.HasRank()) {
    return OperandDescriptor::CreateUnranked(true_value.data_type());
  }

  const std::optional<std::vector<Dimension>> value_shape =
      BroadcastShapes(true_value.shape(), false_value.shape(),
                      /*bidirectional=*/true);
  if (!value_shape) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The shapes of trueValue and falseValue are not broadcastable."));
  }

  std::optional<std::vector<Dimension>> output_shape =
      BroadcastShapes(condition.shape(), value_shape.value(),
                      /*bidirectional=*/true);
  if (!output_shape) {
    return base::unexpected(ErrorWithLabel(
        label,
        "The condition shape is not broadcastable to the shape broadcasted "
        "from trueValue and falseValue."));
  }
  return OperandDescriptor::Create(context_properties, true_value.data_type(),
                                   *output_shape, label);
}

base::expected<void, std::string> ValidateAxes(base::span<const uint32_t> axes,
                                               uint32_t rank,
                                               std::string_view label) {
  if (std::ranges::any_of(axes, [rank](uint32_t axis) {
        return base::MakeStrictNum(axis) >= rank;
      })) {
    return base::unexpected(ErrorWithLabel(
        label, base::StringPrintf(
                   "The values in axes must be in the range [0, %u).", rank)));
  }

  if (axes.size() != std::set<uint32_t>(axes.begin(), axes.end()).size()) {
    return base::unexpected(ErrorWithLabel(
        label, "Two or more values are same in the axes sequence."));
  }

  return base::ok();
}

base::expected<void, std::string> ValidateTensor(
    const ContextProperties& context_properties,
    OperandDescriptor descriptor) {
  // TODO(crbug.com/343638938): Consider adding more constraints to MLTensor
  // creation, such as whether an MLTensor...
  // - may be empty
  // - may have a max size (in addition to `OperandDescriptor` restrictions)

  // TODO(crbug.com/356905054): Consider adding `DataTypeLimits` specific to
  // `MLTensor` rather than using `input`.
  if (!context_properties.data_type_limits.input.data_types.Has(
          descriptor.data_type())) {
    return base::unexpected(NotSupportedMLTensorTypeError(
        descriptor.data_type(),
        context_properties.data_type_limits.input.data_types));
  }

  const size_t byte_length = descriptor.PackedByteLength();
  if (byte_length > context_properties.tensor_byte_length_limit) {
    return base::unexpected(NotSupportedTensorSizeError(
        byte_length, context_properties.tensor_byte_length_limit));
  }

  return base::ok();
}

std::optional<std::vector<uint32_t>> BroadcastShapes(
    base::span<const uint32_t> dims_lhs,
    base::span<const uint32_t> dims_rhs,
    bool bidirectional) {
  // If bidirectional is true, the rank of the output shape is the maximum rank
  // of the input shapes. Otherwise it is as the same as the rhs' rank.
  auto rank_lhs = dims_lhs.size(), rank_rhs = dims_rhs.size();
  if (!bidirectional && rank_lhs > rank_rhs) {
    return std::nullopt;
  }

  auto rank_output = bidirectional ? std::max(rank_lhs, rank_rhs) : rank_rhs;
  std::vector<uint32_t> dims_output(rank_output);
  for (size_t i = 0; i < rank_output; ++i) {
    auto dim_lhs = i < rank_lhs ? dims_lhs[rank_lhs - i - 1] : 1;
    DCHECK_GT(dim_lhs, static_cast<uint32_t>(0));
    auto dim_rhs = i < rank_rhs ? dims_rhs[rank_rhs - i - 1] : 1;
    DCHECK_GT(dim_rhs, static_cast<uint32_t>(0));
    // If bidirectional is true, two dimensions are compatible when they are
    // equal, or one of them is 1. Otherwise, two dimensions are compatible when
    // they are equal, or the lhs dimension is 1.
    if (bidirectional) {
      if (dim_lhs != dim_rhs && dim_lhs != 1 && dim_rhs != 1) {
        return std::nullopt;
      }
    } else if (dim_lhs != dim_rhs && dim_lhs != 1) {
      return std::nullopt;
    }
    // If bidirectional is true, for each dimension of the output tensor, its
    // size is the maximum size along that dimension of the input shapes.
    // Otherwise, its size is the same as the rhs.
    dims_output[rank_output - i - 1] =
        bidirectional ? std::max(dim_lhs, dim_rhs) : dim_rhs;
  }
  return dims_output;
}

std::optional<std::vector<Dimension>> BroadcastShapes(
    base::span<const Dimension> dims_lhs,
    base::span<const Dimension> dims_rhs,
    bool bidirectional) {
  // If bidirectional is true, the rank of the output shape is the maximum rank
  // of the input shapes. Otherwise it is as the same as the rhs' rank.
  auto rank_lhs = dims_lhs.size(), rank_rhs = dims_rhs.size();
  if (!bidirectional && rank_lhs > rank_rhs) {
    return std::nullopt;
  }

  auto rank_output = bidirectional ? std::max(rank_lhs, rank_rhs) : rank_rhs;
  std::vector<Dimension> dims_output(rank_output);
  for (size_t i = 0; i < rank_output; ++i) {
    Dimension dim_lhs;
    if (i < rank_lhs) {
      dim_lhs = dims_lhs[rank_lhs - i - 1];
    } else {
      dim_lhs = 1u;
    }

    Dimension dim_rhs;
    if (i < rank_rhs) {
      dim_rhs = dims_rhs[rank_rhs - i - 1];
    } else {
      dim_rhs = 1u;
    }

    // If bidirectional is true, two dimensions are compatible when they are
    // equal, or one of them is 1. Otherwise, two dimensions are compatible when
    // they are equal, or the lhs dimension is 1.
    bool lhs_is_1 = std::holds_alternative<uint32_t>(dim_lhs) &&
                    std::get<uint32_t>(dim_lhs) == 1;
    bool rhs_is_1 = std::holds_alternative<uint32_t>(dim_rhs) &&
                    std::get<uint32_t>(dim_rhs) == 1;

    // When either dimension is dynamic, consider them compatible at build
    // time. The actual compatibility will be validated at dispatch time with
    // real sizes. `DimensionsAreDefinitelyUnequal` returns false whenever any
    // side is dynamic, so the dynamic cases fall through the checks below.
    if (bidirectional) {
      if (DimensionsAreDefinitelyUnequal(dim_lhs, dim_rhs) && !lhs_is_1 &&
          !rhs_is_1) {
        return std::nullopt;
      }
    } else {
      if (DimensionsAreDefinitelyUnequal(dim_lhs, dim_rhs) && !lhs_is_1) {
        return std::nullopt;
      }
    }

    // If bidirectional is true, for each dimension of the output tensor, its
    // size is the maximum size along that dimension of the input shapes.
    // Otherwise, its size is the same as the rhs.
    if (lhs_is_1) {
      dims_output[rank_output - i - 1] = dim_rhs;
    } else if (rhs_is_1) {
      dims_output[rank_output - i - 1] = dim_lhs;
    } else if (dim_lhs == dim_rhs) {
      // Structurally identical (both static same value, both same-named
      // dynamic, or both unnamed dynamic). Pass the dimension through
      // unchanged; for two unnamed dims this matches the degrade branch's
      // unnamed result anyway.
      dims_output[rank_output - i - 1] = dim_lhs;
    } else {
      // At least one side is dynamic and the two differ (unknown relationship).
      // Degrade to an unnamed dynamic dimension — the actual compatibility is
      // validated at dispatch time with real sizes.
      dims_output[rank_output - i - 1] = DynamicDimension{};
    }
  }
  return dims_output;
}

std::optional<std::vector<Dimension>> ExpandShape(
    base::span<const Dimension> input_shape,
    base::span<const Dimension> new_shape,
    base::span<const DynamicDimension> known_dynamic_dims) {
  // Unidirectionally broadcast input_shape to new_shape.
  auto rank_input = input_shape.size();
  auto rank_new = new_shape.size();

  if (rank_input > rank_new) {
    return std::nullopt;
  }

  std::vector<Dimension> output_shape(rank_new);
  for (size_t i = 0; i < rank_new; ++i) {
    Dimension dim_input;
    if (i < rank_input) {
      dim_input = input_shape[rank_input - i - 1];
    } else {
      dim_input = 1u;
    }

    Dimension dim_new = new_shape[rank_new - i - 1];

    bool input_is_1 = std::holds_alternative<uint32_t>(dim_input) &&
                      std::get<uint32_t>(dim_input) == 1;
    bool new_is_dynamic = std::holds_alternative<DynamicDimension>(dim_new);

    // Check if dimensions are compatible for unidirectional broadcasting.
    // Reject only on a definite size disagreement: a dynamic dimension on
    // either side may still broadcast at dispatch (the values become known
    // then), so it is deferred rather than rejected here.
    if (DimensionsAreDefinitelyUnequal(dim_input, dim_new) && !input_is_1) {
      return std::nullopt;
    }

    // A static dimension of size 1 cannot be expanded to a dynamic dimension
    // unless it's a known dynamic dimension from an input operand.
    if (input_is_1 && new_is_dynamic) {
      const DynamicDimension& new_dyn = std::get<DynamicDimension>(dim_new);
      if (!std::ranges::contains(known_dynamic_dims, new_dyn)) {
        return std::nullopt;
      }
    }

    // Output dimension is the new_shape dimension.
    output_shape[rank_new - i - 1] = dim_new;
  }
  return output_shape;
}

base::expected<uint32_t, std::string> CalculateConvTranspose2dOutputSize(
    const uint32_t input_size,
    const uint32_t filter_size,
    const uint32_t beginning_padding,
    const uint32_t ending_padding,
    const uint32_t stride,
    const uint32_t dilation,
    const uint32_t output_padding) {
  // Calculate the dilated filter sizes.
  auto checked_effective_filter_size =
      (base::CheckedNumeric<uint32_t>(filter_size) - 1) * dilation + 1;
  if (!checked_effective_filter_size.IsValid()) {
    return base::unexpected("The effective filter size is too large.");
  }
  auto checked_output_size =
      (base::CheckedNumeric<uint32_t>(input_size) - 1) * stride +
      checked_effective_filter_size - beginning_padding - ending_padding +
      output_padding;
  if (!checked_output_size.IsValid()) {
    return base::unexpected(
        "The stride is too large or the input size is too small for padding.");
  }

  return checked_output_size.ValueOrDie();
}

bool IsDepthwiseConv2d(uint32_t input_channels,
                       uint32_t output_channels,
                       uint32_t groups) {
  return groups == input_channels && groups == output_channels && groups != 1;
}

}  // namespace webnn
