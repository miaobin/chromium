// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

#include <functional>

#include "base/numerics/safe_conversions.h"
#include "base/strings/strcat.h"
#include "base/types/expected_macros.h"
#include "services/webnn/public/cpp/graph_validation_utils.h"
#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/public/cpp/webnn_errors.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_dynamic_dimension.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_mldynamicdimension_unsignedlongenforcerange.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_constant_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"

namespace blink {

// static
base::expected<MLOperand*, String> MLOperand::ValidateAndCreateInput(
    const webnn::ContextProperties& context_properties,
    MLGraphBuilder* builder,
    V8MLOperandDataType::Enum v8_data_type,
    std::vector<webnn::Dimension> dimensions,
    String name) {
  if (name.empty()) {
    return base::unexpected("The name is empty.");
  }

  const webnn::OperandDataType data_type = FromBlinkDataType(v8_data_type);

  ASSIGN_OR_RETURN(
      webnn::OperandDescriptor descriptor,
      webnn::OperandDescriptor::Create(
          context_properties, data_type, dimensions,
          webnn::GetErrorLabelPrefix(base::StrCat({"input ", name.Utf8()}))),
      [](std::string error) { return String(error); });

  if (!context_properties.data_type_limits.input.Supports(descriptor)) {
    return base::unexpected(String(webnn::NotSupportedInputError(
        name.Utf8(), descriptor, context_properties.data_type_limits.input)));
  }

  auto* input = MakeGarbageCollected<MLOperand>(
      builder, webnn::mojom::blink::Operand::Kind::kInput,
      std::move(descriptor));
  input->name_ = std::move(name);
  return input;
}

// static
MLOperand* MLOperand::CreateOutput(MLGraphBuilder* builder,
                                   webnn::OperandDescriptor descriptor,
                                   MLOperator* ml_operator) {
  CHECK(ml_operator);

  auto* output = MakeGarbageCollected<MLOperand>(
      builder, webnn::mojom::blink::Operand::Kind::kOutput,
      std::move(descriptor));
  output->operator_ = ml_operator;
  return output;
}

MLOperand::MLOperand(MLGraphBuilder* builder,
                     webnn::mojom::blink::Operand::Kind kind,
                     webnn::OperandDescriptor descriptor)
    : builder_(builder), kind_(kind), descriptor_(std::move(descriptor)) {}

MLOperand::~MLOperand() = default;

MLGraphBuilder* MLOperand::Builder() const {
  return builder_.Get();
}

webnn::mojom::blink::Operand::Kind MLOperand::Kind() const {
  return kind_;
}

const String& MLOperand::Name() const {
  CHECK_EQ(kind_, webnn::mojom::blink::Operand::Kind::kInput);
  return name_;
}

MLOperator* MLOperand::Operator() const {
  CHECK_EQ(kind_, webnn::mojom::blink::Operand::Kind::kOutput);
  return operator_.Get();
}

HeapHashSet<Member<MLOperator>>& MLOperand::DependentOperators() {
  return dependent_operators_;
}

const webnn::OperandDescriptor& MLOperand::Descriptor() const {
  return descriptor_;
}

webnn::OperandDataType MLOperand::DataType() const {
  return descriptor_.data_type();
}

const std::vector<webnn::Dimension>& MLOperand::Shape() const {
  return descriptor_.shape();
}

size_t MLOperand::NumberOfElements() const {
  return descriptor_.NumberOfElements();
}

size_t MLOperand::ByteLength() const {
  return descriptor_.PackedByteLength();
}

wtf_size_t MLOperand::Rank() const {
  static_assert(sizeof(descriptor_.Rank()) == sizeof(wtf_size_t));
  return static_cast<wtf_size_t>(descriptor_.Rank());
}

MLDynamicShape MLOperand::shape() const {
  MLDynamicShape result;
  result.ReserveInitialCapacity(descriptor_.shape().size());
  for (const auto& dim : descriptor_.shape()) {
    if (std::holds_alternative<uint32_t>(dim)) {
      result.push_back(
          MakeGarbageCollected<MLDimension>(std::get<uint32_t>(dim)));
    } else {
      const auto& dynamic_dim = std::get<webnn::DynamicDimension>(dim);
      auto* v8_dynamic_dim = MLDynamicDimension::Create();
      v8_dynamic_dim->setName(String::FromUTF8(dynamic_dim.name));
      v8_dynamic_dim->setMaxSize(dynamic_dim.max_size);
      v8_dynamic_dim->setMinSize(dynamic_dim.min_size);
      result.push_back(MakeGarbageCollected<MLDimension>(v8_dynamic_dim));
    }
  }
  return result;
}

V8MLOperandDataType MLOperand::dataType() const {
  return ToBlinkDataType(descriptor_.data_type());
}

MLConstantOperand* MLOperand::AsConstantOperand() {
  CHECK_EQ(kind_, webnn::mojom::blink::Operand::Kind::kConstant);
  return static_cast<MLConstantOperand*>(this);
}

void MLOperand::AddDependentOperator(MLOperator* ml_operator) {
  dependent_operators_.insert(ml_operator);
}

void MLOperand::Trace(Visitor* visitor) const {
  visitor->Trace(builder_);
  visitor->Trace(operator_);
  visitor->Trace(dependent_operators_);
  ScriptWrappable::Trace(visitor);
}

}  // namespace blink
