// META: title=validation tests for WebNN API input interface
// META: global=window,worker
// META: variant=?cpu
// META: variant=?gpu
// META: variant=?npu
// META: script=../resources/utils_validation.js

'use strict';

// A dimension is a number (static), a string (named dynamic dim), or null
// (unnamed dynamic dim).

// Tests for input(name, descriptor)
const tests = [
  {
    testName: '[input] Test building a 0-D scalar input with empty shape',
    name: 'input',
    descriptor: {dataType: 'float32', shape: []},
    output: {dataType: 'float32', shape: []},
  },
  {
    testName: '[input] Test building a 1-D input with int64 data type',
    name: 'input',
    descriptor: {dataType: 'int64', shape: [3]},
    output: {dataType: 'int64', shape: [3]},
  },
  {
    testName: '[input] Test building a 2-D input without errors',
    name: 'input',
    descriptor: {dataType: 'float32', shape: [3, 4]},
    output: {dataType: 'float32', shape: [3, 4]},
  },
  {
    testName: '[input] Throw if the name is empty',
    name: '',
    descriptor: {dataType: 'float32', shape: [3, 4]}
  },
  {
    testName: '[input] Throw if a dimension size is 0',
    name: 'input',
    descriptor: {dataType: 'float32', shape: [3, 0]}
  },
  {
    testName:
        '[input] Throw if the value of any element in dimensions is outside the \'unsigned long\' value range',
    name: 'input',
    descriptor: {dataType: 'float32', shape: [kMaxUnsignedLong + 1]}
  },
  {
    testName: '[input] Throw if the number of elements is too large',
    name: 'input',
    descriptor: {
      dataType: 'float32',
      shape: [kMaxUnsignedLong, kMaxUnsignedLong, kMaxUnsignedLong]
    }
  },
  {
    testName: '[input] Test building an input with a named dynamic dimension',
    name: 'input',
    descriptor: {dataType: 'float32', shape: [2, 'N', 3]},
    output: {dataType: 'float32', shape: [2, 'N', 3]}
  },
  {
    testName:
        '[input] Test building an input with an unnamed dynamic dimension',
    name: 'input',
    descriptor: {dataType: 'float32', shape: [2, null, 3]},
    output: {dataType: 'float32', shape: [2, null, 3]}
  },
  {
    testName: '[input] Test building an input with multiple dynamic dimensions',
    name: 'input',
    descriptor: {dataType: 'float32', shape: ['batch', null, 224, 224]},
    output: {dataType: 'float32', shape: ['batch', null, 224, 224]}
  }
];

tests.forEach(
    test => promise_test(async t => {
      const builder = new MLGraphBuilder(context);
      if (test.output) {
        const inputOperand = builder.input(test.name, test.descriptor);
        assert_equals(inputOperand.dataType, test.output.dataType);
        // Compare shapes element by element to handle dynamic dimensions. A dim
        // is a number (static), a string (named dynamic), or null (unnamed
        // dynamic).
        assert_equals(inputOperand.shape.length, test.output.shape.length);
        for (let i = 0; i < inputOperand.shape.length; i++) {
          assert_equals(inputOperand.shape[i], test.output.shape[i]);
        }
      } else {
        assert_throws_js(
            TypeError, () => builder.input(test.name, test.descriptor));
      }
    }, test.testName));

promise_test(async t => {
  const builder = new MLGraphBuilder(context);

  const inputDescriptor = {
      dataType: 'float32',
      shape: [context.opSupportLimits().maxTensorByteLength / 4 + 1]};

  assert_throws_js(
    TypeError, () => builder.input('input', inputDescriptor));
}, '[input] throw if the output tensor byte length exceeds limit');

promise_test(async t => {
  const builder = new MLGraphBuilder(context);

  // The same named dynamic dimension may be reused across multiple inputs.
  builder.input('input1', {dataType: 'float32', shape: [2, 'N']});
  const input2 =
      builder.input('input2', {dataType: 'float32', shape: ['N', 3]});

  assert_equals(input2.dataType, 'float32');
  assert_equals(input2.shape.length, 2);
  assert_equals(input2.shape[0], 'N');
  assert_equals(input2.shape[1], 3);
}, '[input] Allow a dynamic dimension with the same name across inputs');
