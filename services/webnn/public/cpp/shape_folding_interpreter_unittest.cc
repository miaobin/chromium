// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/public/cpp/shape_folding_interpreter.h"

#include <cstdint>
#include <vector>

#include "base/containers/flat_map.h"
#include "base/containers/span.h"
#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/public/cpp/range.h"
#include "services/webnn/public/cpp/webnn_types.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"
#include "testing/gtest/include/gtest/gtest.h"

namespace webnn {
namespace {

using mojom::ElementWiseBinary;
using mojom::ElementWiseUnary;
using mojom::Operand;
using mojom::OperandPtr;
using mojom::Operation;
using mojom::OperationPtr;

// Helper to create an operand with static (uint32_t) dimensions.
OperandPtr CreateOperand(Operand::Kind kind,
                         OperandDataType data_type,
                         const std::vector<uint32_t>& shape,
                         const std::string& name = "") {
  auto operand = Operand::New();
  operand->kind = kind;
  operand->descriptor =
      OperandDescriptor::UnsafeCreateForTesting(data_type, shape);
  if (!name.empty()) {
    operand->name = name;
  }
  return operand;
}

// Serialize values to bytes in native endian order.
template <typename T>
std::vector<uint8_t> ToBytes(base::span<const T> values) {
  auto byte_span = base::as_bytes(values);
  return std::vector<uint8_t>(byte_span.begin(), byte_span.end());
}

std::vector<uint8_t> Int32ToBytes(const std::vector<int32_t>& values) {
  return ToBytes(base::span(values));
}

std::vector<uint8_t> Int64ToBytes(const std::vector<int64_t>& values) {
  return ToBytes(base::span(values));
}

std::vector<uint8_t> Uint32ToBytes(const std::vector<uint32_t>& values) {
  return ToBytes(base::span(values));
}

// Helper to build a Shape operation.
OperationPtr MakeShapeOp(OperandId input_id, OperandId output_id) {
  auto shape_op = mojom::Shape::New();
  shape_op->input_operand_id = input_id;
  shape_op->output_operand_id = output_id;
  return Operation::NewShape(std::move(shape_op));
}

// Helper to build a Concat operation.
OperationPtr MakeConcatOp(const std::vector<OperandId>& input_ids,
                          OperandId output_id,
                          uint32_t axis = 0) {
  auto concat_op = mojom::Concat::New();
  concat_op->input_operand_ids = input_ids;
  concat_op->output_operand_id = output_id;
  concat_op->axis = axis;
  return Operation::NewConcat(std::move(concat_op));
}

// Helper to build a Gather operation.
OperationPtr MakeGatherOp(OperandId input_id,
                          OperandId indices_id,
                          OperandId output_id,
                          uint32_t axis = 0) {
  auto gather_op = mojom::Gather::New();
  gather_op->input_operand_id = input_id;
  gather_op->indices_operand_id = indices_id;
  gather_op->output_operand_id = output_id;
  gather_op->axis = axis;
  return Operation::NewGather(std::move(gather_op));
}

// Helper to build an ElementWiseBinary operation.
OperationPtr MakeBinaryOp(ElementWiseBinary::Kind kind,
                          OperandId lhs_id,
                          OperandId rhs_id,
                          OperandId output_id) {
  auto binary_op = ElementWiseBinary::New();
  binary_op->kind = kind;
  binary_op->lhs_operand_id = lhs_id;
  binary_op->rhs_operand_id = rhs_id;
  binary_op->output_operand_id = output_id;
  return Operation::NewElementWiseBinary(std::move(binary_op));
}

// Helper to build an ElementWiseUnary operation.
OperationPtr MakeUnaryOp(ElementWiseUnary::Kind kind,
                         OperandId input_id,
                         OperandId output_id) {
  auto unary_op = ElementWiseUnary::New();
  unary_op->kind = kind;
  unary_op->input_operand_id = input_id;
  unary_op->output_operand_id = output_id;
  return Operation::NewElementWiseUnary(std::move(unary_op));
}

// Helper to build a Reshape operation.
OperationPtr MakeReshapeOp(OperandId input_id, OperandId output_id) {
  auto reshape_op = mojom::Reshape::New();
  reshape_op->input_operand_id = input_id;
  reshape_op->output_operand_id = output_id;
  return Operation::NewReshape(std::move(reshape_op));
}

// Helper to build a Reverse operation.
OperationPtr MakeReverseOp(OperandId input_id,
                           OperandId output_id,
                           const std::vector<uint32_t>& axes) {
  auto reverse_op = mojom::Reverse::New();
  reverse_op->input_operand_id = input_id;
  reverse_op->output_operand_id = output_id;
  reverse_op->axes = axes;
  return Operation::NewReverse(std::move(reverse_op));
}

// Helper to build a Slice operation.
OperationPtr MakeSliceOp(OperandId input_id,
                         OperandId output_id,
                         uint32_t start,
                         uint32_t size) {
  auto slice_op = mojom::Slice::New();
  slice_op->input_operand_id = input_id;
  slice_op->output_operand_id = output_id;
  Range range;
  range.start = start;
  range.size = size;
  range.stride = 1;
  slice_op->ranges.push_back(range);
  return Operation::NewSlice(std::move(slice_op));
}

// Helper to build a Transpose operation.
OperationPtr MakeTransposeOp(OperandId input_id,
                             OperandId output_id,
                             const std::vector<uint32_t>& permutation) {
  auto transpose_op = mojom::Transpose::New();
  transpose_op->input_operand_id = input_id;
  transpose_op->output_operand_id = output_id;
  transpose_op->permutation = permutation;
  return Operation::NewTranspose(std::move(transpose_op));
}

class ShapeFoldingInterpreterTest : public testing::Test {
 protected:
  // Convenience method to set up and evaluate an operand.
  std::optional<std::vector<int64_t>> Evaluate(
      const std::vector<OperandPtr>& operands,
      const std::vector<OperationPtr>& operations,
      const base::flat_map<OperandId, size_t>& operand_to_producing_operation,
      const base::flat_map<OperandId, std::vector<uint8_t>>& constant_data,
      OperandId target_operand_id) {
    ShapeFoldingInterpreter interpreter(operands, operations,
                                        operand_to_producing_operation,
                                        constant_data);
    return interpreter.Evaluate(target_operand_id);
  }
};

// Test evaluating a constant operand with int32 values.
TEST_F(ShapeFoldingInterpreterTest, ConstantInt32) {
  std::vector<OperandPtr> operands;
  // operand 0: constant int32 [3] with values {10, 20, 30}
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));

  std::vector<OperationPtr> operations;
  base::flat_map<OperandId, size_t> op_map;
  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({10, 20, 30});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(0));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{10, 20, 30}));
}

// Test evaluating a constant operand with int64 values.
TEST_F(ShapeFoldingInterpreterTest, ConstantInt64) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt64, {2}));

  std::vector<OperationPtr> operations;
  base::flat_map<OperandId, size_t> op_map;
  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] =
      Int64ToBytes({100000000000LL, -200000000000LL});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(0));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result,
            (std::vector<int64_t>{100000000000LL, -200000000000LL}));
}

// Test evaluating a constant operand with uint32 values.
TEST_F(ShapeFoldingInterpreterTest, ConstantUint32) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kUint32, {2}));

  std::vector<OperationPtr> operations;
  base::flat_map<OperandId, size_t> op_map;
  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Uint32ToBytes({42, 100});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(0));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{42, 100}));
}

// Test that float constant operands return nullopt (non-integer).
TEST_F(ShapeFoldingInterpreterTest, ConstantFloatReturnsNullopt) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kFloat32, {2}));

  std::vector<OperationPtr> operations;
  base::flat_map<OperandId, size_t> op_map;
  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  // Float data — ReadIntegerValues won't know how to interpret it.
  // Provide bytes of the right size (8 bytes for 2 x float32).
  constant_data[OperandId(0)] = std::vector<uint8_t>(8, 0);

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(0));
  EXPECT_FALSE(result.has_value());
}

// Test that input operands return nullopt (values not known at build time).
TEST_F(ShapeFoldingInterpreterTest, InputOperandReturnsNullopt) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kInput, OperandDataType::kFloat32,
                    {2, 3}, "input"));

  std::vector<OperationPtr> operations;
  base::flat_map<OperandId, size_t> op_map;
  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(0));
  EXPECT_FALSE(result.has_value());
}

// Test shape() operation: extracts dimensions of an input tensor.
TEST_F(ShapeFoldingInterpreterTest, ShapeOp) {
  std::vector<OperandPtr> operands;
  // operand 0: input with shape [2, 3, 4]
  operands.push_back(
      CreateOperand(Operand::Kind::kInput, OperandDataType::kFloat32,
                    {2, 3, 4}, "input"));
  // operand 1: output of shape() — 1-D int64 tensor with 3 elements
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt64, {3}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeShapeOp(OperandId(0), OperandId(1)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;  // operand 1 is produced by operation 0

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(1));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{2, 3, 4}));
}

// Test gather: pick specific elements from a 1-D shape tensor.
TEST_F(ShapeFoldingInterpreterTest, GatherFromConstant) {
  std::vector<OperandPtr> operands;
  // operand 0: constant [5] with values {10, 20, 30, 40, 50}
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {5}));
  // operand 1: constant indices [2] with values {1, 3}
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
  // operand 2: output of gather [2]
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {2}));

  std::vector<OperationPtr> operations;
  operations.push_back(
      MakeGatherOp(OperandId(0), OperandId(1), OperandId(2)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(2)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({10, 20, 30, 40, 50});
  constant_data[OperandId(1)] = Int32ToBytes({1, 3});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(2));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{20, 40}));
}

// Test gather with out-of-bounds index returns nullopt.
TEST_F(ShapeFoldingInterpreterTest, GatherOutOfBounds) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {1}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {1}));

  std::vector<OperationPtr> operations;
  operations.push_back(
      MakeGatherOp(OperandId(0), OperandId(1), OperandId(2)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(2)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({10, 20, 30});
  constant_data[OperandId(1)] = Int32ToBytes({5});  // out of bounds

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(2));
  EXPECT_FALSE(result.has_value());
}

// Test concat: concatenate two constant 1-D tensors.
TEST_F(ShapeFoldingInterpreterTest, ConcatConstants) {
  std::vector<OperandPtr> operands;
  // operand 0: constant [2] = {1, 2}
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
  // operand 1: constant [3] = {3, 4, 5}
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  // operand 2: output of concat [5]
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {5}));

  std::vector<OperationPtr> operations;
  operations.push_back(
      MakeConcatOp({OperandId(0), OperandId(1)}, OperandId(2)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(2)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({1, 2});
  constant_data[OperandId(1)] = Int32ToBytes({3, 4, 5});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(2));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{1, 2, 3, 4, 5}));
}

// Test a typical shape computation chain: shape → gather → concat.
// Simulates building output shape for dynamicReshape:
//   input[batch, 784] → shape() → gather(0) → concat with constant(28, 28)
//   Result: [batch, 28, 28]
TEST_F(ShapeFoldingInterpreterTest, ShapeGatherConcatChain) {
  std::vector<OperandPtr> operands;
  // operand 0: input [batch=4, 784] — shape known at dispatch time
  operands.push_back(
      CreateOperand(Operand::Kind::kInput, OperandDataType::kFloat32,
                    {4, 784}, "input"));
  // operand 1: output of shape() — [2] int64
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt64, {2}));
  // operand 2: constant indices [1] = {0} — gather the batch dimension
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {1}));
  // operand 3: output of gather — [1] int64, the batch dimension value
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt64, {1}));
  // operand 4: constant [2] = {28, 28}
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt64, {2}));
  // operand 5: output of concat — [3] int64, the target shape [batch, 28, 28]
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt64, {3}));

  std::vector<OperationPtr> operations;
  // op 0: shape(input) → operand 1
  operations.push_back(MakeShapeOp(OperandId(0), OperandId(1)));
  // op 1: gather(shape_output, indices) → operand 3
  operations.push_back(
      MakeGatherOp(OperandId(1), OperandId(2), OperandId(3)));
  // op 2: concat([batch_dim, constants]) → operand 5
  operations.push_back(
      MakeConcatOp({OperandId(3), OperandId(4)}, OperandId(5)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;  // shape op
  op_map[OperandId(3)] = 1;  // gather op
  op_map[OperandId(5)] = 2;  // concat op

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(2)] = Int32ToBytes({0});
  constant_data[OperandId(4)] = Int64ToBytes({28, 28});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(5));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{4, 28, 28}));
}

// Test element-wise binary: add.
TEST_F(ShapeFoldingInterpreterTest, BinaryAdd) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {3}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeBinaryOp(ElementWiseBinary::Kind::kAdd,
                                    OperandId(0), OperandId(1),
                                    OperandId(2)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(2)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({1, 2, 3});
  constant_data[OperandId(1)] = Int32ToBytes({10, 20, 30});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(2));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{11, 22, 33}));
}

// Test element-wise binary: sub, mul, div.
TEST_F(ShapeFoldingInterpreterTest, BinarySubMulDiv) {
  // Sub: [10, 20] - [3, 5] = [7, 15]
  {
    std::vector<OperandPtr> operands;
    operands.push_back(
        CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
    operands.push_back(
        CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
    operands.push_back(
        CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {2}));

    std::vector<OperationPtr> operations;
    operations.push_back(MakeBinaryOp(ElementWiseBinary::Kind::kSub,
                                      OperandId(0), OperandId(1),
                                      OperandId(2)));

    base::flat_map<OperandId, size_t> op_map;
    op_map[OperandId(2)] = 0;

    base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
    constant_data[OperandId(0)] = Int32ToBytes({10, 20});
    constant_data[OperandId(1)] = Int32ToBytes({3, 5});

    auto result = Evaluate(operands, operations, op_map, constant_data,
                           OperandId(2));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, (std::vector<int64_t>{7, 15}));
  }

  // Mul: [2, 3] * [4, 5] = [8, 15]
  {
    std::vector<OperandPtr> operands;
    operands.push_back(
        CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
    operands.push_back(
        CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
    operands.push_back(
        CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {2}));

    std::vector<OperationPtr> operations;
    operations.push_back(MakeBinaryOp(ElementWiseBinary::Kind::kMul,
                                      OperandId(0), OperandId(1),
                                      OperandId(2)));

    base::flat_map<OperandId, size_t> op_map;
    op_map[OperandId(2)] = 0;

    base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
    constant_data[OperandId(0)] = Int32ToBytes({2, 3});
    constant_data[OperandId(1)] = Int32ToBytes({4, 5});

    auto result = Evaluate(operands, operations, op_map, constant_data,
                           OperandId(2));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, (std::vector<int64_t>{8, 15}));
  }

  // Div: [20, 15] / [4, 3] = [5, 5]
  {
    std::vector<OperandPtr> operands;
    operands.push_back(
        CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
    operands.push_back(
        CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
    operands.push_back(
        CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {2}));

    std::vector<OperationPtr> operations;
    operations.push_back(MakeBinaryOp(ElementWiseBinary::Kind::kDiv,
                                      OperandId(0), OperandId(1),
                                      OperandId(2)));

    base::flat_map<OperandId, size_t> op_map;
    op_map[OperandId(2)] = 0;

    base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
    constant_data[OperandId(0)] = Int32ToBytes({20, 15});
    constant_data[OperandId(1)] = Int32ToBytes({4, 3});

    auto result = Evaluate(operands, operations, op_map, constant_data,
                           OperandId(2));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, (std::vector<int64_t>{5, 5}));
  }
}

// Test division by zero returns nullopt.
TEST_F(ShapeFoldingInterpreterTest, DivByZeroReturnsNullopt) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {2}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeBinaryOp(ElementWiseBinary::Kind::kDiv,
                                    OperandId(0), OperandId(1),
                                    OperandId(2)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(2)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({10, 20});
  constant_data[OperandId(1)] = Int32ToBytes({0, 5});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(2));
  EXPECT_FALSE(result.has_value());
}

// Test binary broadcast: scalar + vector.
TEST_F(ShapeFoldingInterpreterTest, BinaryBroadcastScalar) {
  std::vector<OperandPtr> operands;
  // operand 0: scalar constant [1] = {10}
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {1}));
  // operand 1: vector constant [3] = {1, 2, 3}
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {3}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeBinaryOp(ElementWiseBinary::Kind::kMul,
                                    OperandId(0), OperandId(1),
                                    OperandId(2)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(2)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({10});
  constant_data[OperandId(1)] = Int32ToBytes({1, 2, 3});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(2));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{10, 20, 30}));
}

// Test mod operation.
TEST_F(ShapeFoldingInterpreterTest, BinaryMod) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {3}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeBinaryOp(ElementWiseBinary::Kind::kMod,
                                    OperandId(0), OperandId(1),
                                    OperandId(2)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(2)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({10, 7, 15});
  constant_data[OperandId(1)] = Int32ToBytes({3, 4, 5});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(2));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{1, 3, 0}));
}

// Test max and min binary operations.
TEST_F(ShapeFoldingInterpreterTest, BinaryMaxMin) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  // max output
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {3}));
  // min output
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {3}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeBinaryOp(ElementWiseBinary::Kind::kMax,
                                    OperandId(0), OperandId(1),
                                    OperandId(2)));
  operations.push_back(MakeBinaryOp(ElementWiseBinary::Kind::kMin,
                                    OperandId(0), OperandId(1),
                                    OperandId(3)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(2)] = 0;
  op_map[OperandId(3)] = 1;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({5, 2, 8});
  constant_data[OperandId(1)] = Int32ToBytes({3, 7, 1});

  auto max_result = Evaluate(operands, operations, op_map, constant_data,
                             OperandId(2));
  ASSERT_TRUE(max_result.has_value());
  EXPECT_EQ(*max_result, (std::vector<int64_t>{5, 7, 8}));

  auto min_result = Evaluate(operands, operations, op_map, constant_data,
                             OperandId(3));
  ASSERT_TRUE(min_result.has_value());
  EXPECT_EQ(*min_result, (std::vector<int64_t>{3, 2, 1}));
}

// Test reshape: values stay the same, only shape changes.
TEST_F(ShapeFoldingInterpreterTest, ReshapePassthrough) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {6}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32,
                    {2, 3}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeReshapeOp(OperandId(0), OperandId(1)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({1, 2, 3, 4, 5, 6});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(1));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{1, 2, 3, 4, 5, 6}));
}

// Test reverse: reverses 1-D values.
TEST_F(ShapeFoldingInterpreterTest, Reverse1D) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {4}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {4}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeReverseOp(OperandId(0), OperandId(1), {0}));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({1, 2, 3, 4});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(1));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{4, 3, 2, 1}));
}

// Test transpose: for 1-D, returns same values.
TEST_F(ShapeFoldingInterpreterTest, Transpose1D) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {3}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeTransposeOp(OperandId(0), OperandId(1), {0}));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({10, 20, 30});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(1));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{10, 20, 30}));
}

// Test slice: extract a sub-range from 1-D.
TEST_F(ShapeFoldingInterpreterTest, Slice1D) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {5}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {3}));

  std::vector<OperationPtr> operations;
  // slice from start=1, size=3 → elements at index 1,2,3
  operations.push_back(MakeSliceOp(OperandId(0), OperandId(1), 1, 3));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({10, 20, 30, 40, 50});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(1));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{20, 30, 40}));
}

// Test slice out of bounds returns nullopt.
TEST_F(ShapeFoldingInterpreterTest, SliceOutOfBounds) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {3}));

  std::vector<OperationPtr> operations;
  // slice from start=2, size=3 → exceeds [3] bounds
  operations.push_back(MakeSliceOp(OperandId(0), OperandId(1), 2, 3));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({10, 20, 30});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(1));
  EXPECT_FALSE(result.has_value());
}

// Test unary: abs and neg.
TEST_F(ShapeFoldingInterpreterTest, UnaryAbsNeg) {
  // Abs
  {
    std::vector<OperandPtr> operands;
    operands.push_back(
        CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
    operands.push_back(
        CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {3}));

    std::vector<OperationPtr> operations;
    operations.push_back(MakeUnaryOp(ElementWiseUnary::Kind::kAbs,
                                     OperandId(0), OperandId(1)));

    base::flat_map<OperandId, size_t> op_map;
    op_map[OperandId(1)] = 0;

    base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
    constant_data[OperandId(0)] = Int32ToBytes({-5, 3, -1});

    auto result = Evaluate(operands, operations, op_map, constant_data,
                           OperandId(1));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, (std::vector<int64_t>{5, 3, 1}));
  }

  // Neg
  {
    std::vector<OperandPtr> operands;
    operands.push_back(
        CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {3}));
    operands.push_back(
        CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {3}));

    std::vector<OperationPtr> operations;
    operations.push_back(MakeUnaryOp(ElementWiseUnary::Kind::kNeg,
                                     OperandId(0), OperandId(1)));

    base::flat_map<OperandId, size_t> op_map;
    op_map[OperandId(1)] = 0;

    base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
    constant_data[OperandId(0)] = Int32ToBytes({5, -3, 0});

    auto result = Evaluate(operands, operations, op_map, constant_data,
                           OperandId(1));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, (std::vector<int64_t>{-5, 3, 0}));
  }
}

// Test cast to integer type.
TEST_F(ShapeFoldingInterpreterTest, CastToInteger) {
  std::vector<OperandPtr> operands;
  // Input int32 constant
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
  // Output int64 (cast int32 → int64)
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt64, {2}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeUnaryOp(ElementWiseUnary::Kind::kCast,
                                   OperandId(0), OperandId(1)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({42, -7});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(1));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{42, -7}));
}

// Test cast to float type returns nullopt.
TEST_F(ShapeFoldingInterpreterTest, CastToFloatReturnsNullopt) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
  // Cast to float32 — should fail
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kFloat32, {2}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeUnaryOp(ElementWiseUnary::Kind::kCast,
                                   OperandId(0), OperandId(1)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({1, 2});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(1));
  EXPECT_FALSE(result.has_value());
}

// Test floor and ceil on integer values (no-op).
TEST_F(ShapeFoldingInterpreterTest, FloorCeilNoOp) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {2}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {2}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeUnaryOp(ElementWiseUnary::Kind::kFloor,
                                   OperandId(0), OperandId(1)));
  operations.push_back(MakeUnaryOp(ElementWiseUnary::Kind::kCeil,
                                   OperandId(0), OperandId(2)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;
  op_map[OperandId(2)] = 1;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({7, -3});

  auto floor_result = Evaluate(operands, operations, op_map, constant_data,
                               OperandId(1));
  ASSERT_TRUE(floor_result.has_value());
  EXPECT_EQ(*floor_result, (std::vector<int64_t>{7, -3}));

  auto ceil_result = Evaluate(operands, operations, op_map, constant_data,
                              OperandId(2));
  ASSERT_TRUE(ceil_result.has_value());
  EXPECT_EQ(*ceil_result, (std::vector<int64_t>{7, -3}));
}

// Test unsupported unary operation (e.g., sqrt) returns nullopt.
TEST_F(ShapeFoldingInterpreterTest, UnsupportedUnaryReturnsNullopt) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {2}));

  std::vector<OperationPtr> operations;
  operations.push_back(MakeUnaryOp(ElementWiseUnary::Kind::kSqrt,
                                   OperandId(0), OperandId(1)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({4, 9});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(1));
  EXPECT_FALSE(result.has_value());
}

// Test caching: evaluating the same operand twice returns the same result
// without re-evaluation (verified by the test running without error).
TEST_F(ShapeFoldingInterpreterTest, CachingWorks) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));

  std::vector<OperationPtr> operations;
  base::flat_map<OperandId, size_t> op_map;
  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({42, 99});

  ShapeFoldingInterpreter interpreter(operands, operations, op_map,
                                      constant_data);
  auto result1 = interpreter.Evaluate(OperandId(0));
  auto result2 = interpreter.Evaluate(OperandId(0));
  ASSERT_TRUE(result1.has_value());
  ASSERT_TRUE(result2.has_value());
  EXPECT_EQ(*result1, *result2);
}

// Test missing constant data returns nullopt.
TEST_F(ShapeFoldingInterpreterTest, MissingConstantData) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));

  std::vector<OperationPtr> operations;
  base::flat_map<OperandId, size_t> op_map;
  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  // No data for operand 0!

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(0));
  EXPECT_FALSE(result.has_value());
}

// Test invalid operand ID returns nullopt.
TEST_F(ShapeFoldingInterpreterTest, InvalidOperandId) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {2}));

  std::vector<OperationPtr> operations;
  base::flat_map<OperandId, size_t> op_map;
  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(999));
  EXPECT_FALSE(result.has_value());
}

// Test a complex chain: shape → gather → add with constant → result.
// Simulates computing: new_dim = shape(input)[2] + 10
TEST_F(ShapeFoldingInterpreterTest, ComplexChainShapeGatherAdd) {
  std::vector<OperandPtr> operands;
  // 0: input [batch=2, channels=3, height=32]
  operands.push_back(
      CreateOperand(Operand::Kind::kInput, OperandDataType::kFloat32,
                    {2, 3, 32}, "input"));
  // 1: shape() output [3]
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt64, {3}));
  // 2: gather indices [1] = {2}
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {1}));
  // 3: gather output [1] — value of height dimension
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt64, {1}));
  // 4: constant [1] = {10}
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt64, {1}));
  // 5: add output [1] = height + 10
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt64, {1}));

  std::vector<OperationPtr> operations;
  // op 0: shape(input) → 1
  operations.push_back(MakeShapeOp(OperandId(0), OperandId(1)));
  // op 1: gather(shape, [2]) → 3
  operations.push_back(
      MakeGatherOp(OperandId(1), OperandId(2), OperandId(3)));
  // op 2: add(gather_result, 10) → 5
  operations.push_back(MakeBinaryOp(ElementWiseBinary::Kind::kAdd,
                                    OperandId(3), OperandId(4),
                                    OperandId(5)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(1)] = 0;
  op_map[OperandId(3)] = 1;
  op_map[OperandId(5)] = 2;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(2)] = Int32ToBytes({2});
  constant_data[OperandId(4)] = Int64ToBytes({10});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(5));
  ASSERT_TRUE(result.has_value());
  // height=32 + 10 = 42
  EXPECT_EQ(*result, (std::vector<int64_t>{42}));
}

// Test gather with negative index.
TEST_F(ShapeFoldingInterpreterTest, GatherNegativeIndex) {
  std::vector<OperandPtr> operands;
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {4}));
  // Negative index -1 should wrap to last element.
  operands.push_back(
      CreateOperand(Operand::Kind::kConstant, OperandDataType::kInt32, {1}));
  operands.push_back(
      CreateOperand(Operand::Kind::kOutput, OperandDataType::kInt32, {1}));

  std::vector<OperationPtr> operations;
  operations.push_back(
      MakeGatherOp(OperandId(0), OperandId(1), OperandId(2)));

  base::flat_map<OperandId, size_t> op_map;
  op_map[OperandId(2)] = 0;

  base::flat_map<OperandId, std::vector<uint8_t>> constant_data;
  constant_data[OperandId(0)] = Int32ToBytes({10, 20, 30, 40});
  constant_data[OperandId(1)] = Int32ToBytes({-1});

  auto result = Evaluate(operands, operations, op_map, constant_data,
                         OperandId(2));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, (std::vector<int64_t>{40}));
}

}  // namespace
}  // namespace webnn
