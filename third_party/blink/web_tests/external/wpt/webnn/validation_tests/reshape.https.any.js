// META: title=validation tests for WebNN API reshape operation
// META: global=window
// META: variant=?cpu
// META: variant=?gpu
// META: variant=?npu
// META: script=../resources/utils_validation.js

'use strict';

multi_builder_test(async (t, builder, otherBuilder) => {
  const inputFromOtherBuilder =
      otherBuilder.input('input', {dataType: 'float32', shape: [1, 2, 3]});

  const newShape = [3, 2, 1];
  assert_throws_js(
      TypeError, () => builder.reshape(inputFromOtherBuilder, newShape));
}, '[reshape] throw if input is from another builder');

const tests = [
  {
    name: '[reshape] Test with new shape=[3, 8].',
    input: {dataType: 'float32', shape: [2, 3, 4]},
    newShape: [3, 8],
    output: {dataType: 'float32', shape: [3, 8]}
  },
  {
    name: '[reshape] Test with new shape=[24], src shape=[2, 3, 4].',
    input: {dataType: 'float32', shape: [2, 3, 4]},
    newShape: [24],
    output: {dataType: 'float32', shape: [24]}
  },
  {
    name: '[reshape] Test with new shape=[1], src shape=[1].',
    input: {dataType: 'float32', shape: [1]},
    newShape: [1],
    output: {dataType: 'float32', shape: [1]}
  },
  {
    name: '[reshape] Test reshaping a 1-D 1-element tensor to scalar.',
    input: {dataType: 'float32', shape: [1]},
    newShape: [],
    output: {dataType: 'float32', shape: []}
  },
  {
    name: '[reshape] Test reshaping a scalar to 1-D 1-element tensor.',
    input: {dataType: 'float32', shape: []},
    newShape: [1],
    output: {dataType: 'float32', shape: [1]}
  },
  {
    name:
        '[reshape] Test with dynamic dimension in input, reshaping to different layout.',
    input: {dataType: 'float32', shape: [{name: 'batch', maxSize: 10}, 3, 4]},
    newShape: [{name: 'batch', maxSize: 10}, 12],
    output: {dataType: 'float32', shape: [{name: 'batch', maxSize: 10}, 12]}
  },
  {
    name:
        '[reshape] Test with multiple dynamic dimensions in input, preserving one.',
    input: {
      dataType: 'float32',
      shape: [{name: 'batch', maxSize: 5}, {name: 'seq', maxSize: 20}, 8]
    },
    newShape: [{name: 'batch', maxSize: 5}, 160],
  },
  {
    name: '[reshape] Test with dynamic dimension, reshaping from 1-D to 2-D.',
    input: {dataType: 'float32', shape: [{name: 'batch', maxSize: 100}]},
    newShape: [{name: 'batch', maxSize: 100}, 1],
    output: {dataType: 'float32', shape: [{name: 'batch', maxSize: 100}, 1]}
  },
  {
    name: '[reshape] Throw if one value of new shape is 0.',
    input: {dataType: 'float32', shape: [2, 4]},
    newShape: [2, 4, 0],
  },
  {
    name:
        '[reshape] Throw if the number of elements implied by new shape is not equal to the number of elements in the input tensor when new shape=[].',
    input: {dataType: 'float32', shape: [2, 3, 4]},
    newShape: [],
  },
  {
    name:
        '[reshape] Throw if the number of elements implied by new shape is not equal to the number of elements in the input tensor.',
    input: {dataType: 'float32', shape: [2, 3, 4]},
    newShape: [3, 9],
  },
  {
    name:
        '[reshape] Throw if dynamic dimension in output shape does not exist in input shape.',
    input: {dataType: 'float32', shape: [{name: 'batch', maxSize: 10}, 3, 4]},
    newShape: [{name: 'width', maxSize: 12}, 12],
  },
  {
    name:
        '[reshape] Throw if dynamic dimension appears multiple times in output shape but only once in input.',
    input: {dataType: 'float32', shape: [{name: 'batch', maxSize: 10}, 3, 4]},
    newShape: [{name: 'batch', maxSize: 10}, {name: 'batch', maxSize: 10}, 6],
  },
  {
    name:
        '[reshape] Throw if output shape uses dynamic dimension that appears multiple times in input.',
    input: {
      dataType: 'float32',
      shape: [{name: 'seq', maxSize: 20}, 3, {name: 'seq', maxSize: 20}]
    },
    newShape: [{name: 'seq', maxSize: 20}, 9],
  },
  {
    name:
        '[reshape] Throw if mixing dynamic dimensions with incompatible static size.',
    input: {dataType: 'float32', shape: [{name: 'batch', maxSize: 10}, 6]},
    newShape: [{name: 'batch', maxSize: 10}, 3, 3],
  },
];

tests.forEach(
    test => promise_test(async t => {
      const builder = new MLGraphBuilder(context);
      const input = builder.input('input', test.input);
      if (test.output) {
        const output = builder.reshape(input, test.newShape);
        assert_equals(output.dataType, test.output.dataType);
        // Compare shapes element by element to handle dynamic dimensions
        assert_equals(output.shape.length, test.output.shape.length);
        for (let i = 0; i < output.shape.length; i++) {
          const outputDim = output.shape[i];
          const expectedDim = test.output.shape[i];
          if (typeof outputDim === 'number') {
            assert_equals(outputDim, expectedDim);
          } else {
            assert_equals(typeof outputDim, 'object');
            assert_equals(outputDim.name, expectedDim.name);
            assert_equals(outputDim.maxSize, expectedDim.maxSize);
          }
        }
      } else {
        const label = 'reshape_xxx';
        const options = {label};
        const regrexp = new RegExp('\\[' + label + '\\]');
        assert_throws_with_label(
            () => builder.reshape(input, test.newShape, options), regrexp);
      }
    }, test.name));

promise_test(async t => {
  const builder = new MLGraphBuilder(context);

  const input = builder.input('input', {dataType: 'float32', shape: [2]});
  const newShape =
      new Array(context.opSupportLimits().expand.output.rankRange.max + 1)
          .fill(1);
  newShape[0] = 2;

  const label = 'reshape_xxx';
  const options = {label};
  const regrexp = new RegExp('\\[' + label + '\\]');
  assert_throws_with_label(
      () => builder.reshape(input, newShape, options), regrexp);
}, '[expand] throw if new shape rank exceeds limit');
