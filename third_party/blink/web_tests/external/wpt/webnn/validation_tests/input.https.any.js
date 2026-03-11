// META: title=validation tests for WebNN API input interface
// META: global=window,worker
// META: variant=?cpu
// META: variant=?gpu
// META: variant=?npu
// META: script=../resources/utils_validation.js

'use strict';

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
    testName: '[input] Throw if dynamic dimension minSize is 0',
    name: 'input',
    descriptor:
        {dataType: 'float32', shape: [2, {name: 'N', maxSize: 5, minSize: 0}]}
  },
  {
    testName: '[input] Throw if dynamic dimension maxSize is 0',
    name: 'input',
    descriptor:
        {dataType: 'float32', shape: [{name: 'N', maxSize: 0, minSize: 1}, 3]}
  },
  {
    testName:
        '[input] Throw if dynamic dimension minSize is greater than maxSize',
    name: 'input',
    descriptor: {
      dataType: 'float32',
      shape: [2, {name: 'N', maxSize: 3, minSize: 5}, 4]
    }
  },
  {
    testName: '[input] Test building an input with valid dynamic dimension',
    name: 'input',
    descriptor: {
      dataType: 'float32',
      shape: [2, {name: 'N', maxSize: 10, minSize: 1}, 3]
    },
    output: {
      dataType: 'float32',
      shape: [2, {name: 'N', maxSize: 10, minSize: 1}, 3]
    }
  },
  {
    testName:
        '[input] Test building an input with dynamic dimension where minSize equals maxSize',
    name: 'input',
    descriptor:
        {dataType: 'float32', shape: [{name: 'N', maxSize: 5, minSize: 5}]},
    output: {dataType: 'float32', shape: [{name: 'N', maxSize: 5, minSize: 5}]}
  }
];

tests.forEach(
    test => promise_test(async t => {
      const builder = new MLGraphBuilder(context);
      if (test.output) {
        const inputOperand = builder.input(test.name, test.descriptor);
        assert_equals(inputOperand.dataType, test.output.dataType);
        // Compare shapes element by element to handle dynamic dimensions
        assert_equals(inputOperand.shape.length, test.output.shape.length);
        for (let i = 0; i < inputOperand.shape.length; i++) {
          const actualDim = inputOperand.shape[i];
          const expectedDim = test.output.shape[i];
          if (typeof actualDim === 'number') {
            assert_equals(actualDim, expectedDim);
          } else {
            assert_equals(typeof actualDim, 'object');
            assert_equals(actualDim.name, expectedDim.name);
            assert_equals(actualDim.maxSize, expectedDim.maxSize);
            assert_equals(actualDim.minSize, expectedDim.minSize);
          }
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

  // First input with dynamic dimension 'N' having minSize=1, maxSize=10
  builder.input(
      'input1',
      {dataType: 'float32', shape: [2, {name: 'N', maxSize: 10, minSize: 1}]});

  // Second input with same dynamic dimension name but different maxSize
  assert_throws_js(TypeError, () => {
    builder.input('input2', {
      dataType: 'float32',
      shape: [{name: 'N', maxSize: 20, minSize: 1}, 3]
    });
  });
}, '[input] Throw if dynamic dimension has same name but different maxSize across inputs');

promise_test(async t => {
  const builder = new MLGraphBuilder(context);

  // First input with dynamic dimension 'N' having minSize=2, maxSize=10
  builder.input(
      'input1',
      {dataType: 'float32', shape: [{name: 'N', maxSize: 10, minSize: 2}, 4]});

  // Second input with same dynamic dimension name but different minSize
  assert_throws_js(TypeError, () => {
    builder.input('input2', {
      dataType: 'float32',
      shape: [3, {name: 'N', maxSize: 10, minSize: 5}]
    });
  });
}, '[input] Throw if dynamic dimension has same name but different minSize across inputs');

promise_test(async t => {
  const builder = new MLGraphBuilder(context);

  // First input with dynamic dimension 'N' having minSize=1, maxSize=10
  builder.input(
      'input1',
      {dataType: 'float32', shape: [2, {name: 'N', maxSize: 10, minSize: 1}]});

  // Second input with same dynamic dimension name but both different minSize
  // and maxSize
  assert_throws_js(TypeError, () => {
    builder.input('input2', {
      dataType: 'float32',
      shape: [{name: 'N', maxSize: 20, minSize: 5}, 3]
    });
  });
}, '[input] Throw if dynamic dimension has same name but different minSize and maxSize across inputs');

promise_test(async t => {
  const builder = new MLGraphBuilder(context);

  // First input with dynamic dimension 'N'
  builder.input(
      'input1',
      {dataType: 'float32', shape: [2, {name: 'N', maxSize: 10, minSize: 1}]});

  // Second input with same dynamic dimension name and same minSize, maxSize
  // should succeed
  const input2 = builder.input(
      'input2',
      {dataType: 'float32', shape: [{name: 'N', maxSize: 10, minSize: 1}, 3]});

  assert_equals(input2.dataType, 'float32');
  assert_equals(input2.shape.length, 2);
  assert_equals(input2.shape[0].name, 'N');
  assert_equals(input2.shape[0].maxSize, 10);
  assert_equals(input2.shape[0].minSize, 1);
  assert_equals(input2.shape[1], 3);
}, '[input] Allow dynamic dimension with same name, minSize and maxSize across inputs');
