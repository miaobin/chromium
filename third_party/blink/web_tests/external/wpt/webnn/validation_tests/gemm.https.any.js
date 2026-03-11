// META: title=validation tests for WebNN API gemm operation
// META: global=window
// META: variant=?cpu
// META: variant=?gpu
// META: variant=?npu
// META: script=../resources/utils_validation.js

'use strict';

const label = 'gemm_xxx';
const kExampleInputDescriptor = {
  dataType: 'float32',
  shape: [2, 2]
};

validateTwoInputsFromMultipleBuilders('gemm');
validateTwoBroadcastableInputsTensorLimit('gemm', label);

multi_builder_test(async (t, builder, otherBuilder) => {
  const cFromOtherBuilder = otherBuilder.input('c', kExampleInputDescriptor);
  const options = {c: cFromOtherBuilder};

  const a = builder.input('a', kExampleInputDescriptor);
  const b = builder.input('b', kExampleInputDescriptor);
  assert_throws_js(TypeError, () => builder.gemm(a, b, options));
}, '[gemm] throw if c option is from another builder');

const tests = [
  {
    name: '[gemm] Test building gemm with default option.',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float32', shape: [3, 4]},
    output: {dataType: 'float32', shape: [2, 4]}
  },
  {
    name:
        '[gemm] Throw if inputShapeA[1] is not equal to inputShapeB[0] default options.',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float32', shape: [2, 4]},
    options: {label}
  },
  {
    name: '[gemm] Test building gemm with aTranspose=true.',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float32', shape: [2, 4]},
    options: {
      aTranspose: true,
    },
    output: {dataType: 'float32', shape: [3, 4]}
  },
  {
    name:
        '[gemm] Throw if inputShapeA[0] is not equal to inputShapeB[0] with aTranspose=true.',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float32', shape: [3, 4]},
    options: {
      aTranspose: true,
      label: label,
    },
  },
  {
    name: '[gemm] Test building gemm with bTranspose=true.',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float32', shape: [4, 3]},
    options: {
      bTranspose: true,
    },
    output: {dataType: 'float32', shape: [2, 4]}
  },
  {
    name:
        '[gemm] Throw if inputShapeA[0] is not equal to inputShapeB[0] with bTranspose=true.',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float32', shape: [3, 4]},
    options: {
      bTranspose: true,
      label: label,
    },
  },
  {
    name: '[gemm] Throw if the rank of inputA is not 2.',
    a: {dataType: 'float32', shape: [2, 3, 1]},
    b: {dataType: 'float32', shape: [2, 4]},
    options: {label}
  },
  {
    name: '[gemm] Throw if the rank of inputB is not 2.',
    a: {dataType: 'float32', shape: [2, 4]},
    b: {dataType: 'float32', shape: [2, 3, 1]},
    options: {label}
  },
  {
    name: '[gemm] Throw if data types of two inputs do not match.',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float16', shape: [3, 4]},
    options: {label}
  },
  {
    name: '[gemm] Test building gemm with inputC.',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float32', shape: [3, 4]},
    options: {
      c: {dataType: 'float32', shape: [4]},
    },
    output: {dataType: 'float32', shape: [2, 4]}
  },
  {
    name: '[gemm] Test building gemm with scalar inputC.',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float32', shape: [3, 4]},
    options: {
      c: {dataType: 'float32', shape: []},
    },
    output: {dataType: 'float32', shape: [2, 4]}
  },
  {
    name:
        '[gemm] Throw if inputShapeC is not unidirectionally broadcastable to the output shape [inputShapeA[0], inputShapeB[1]].',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float32', shape: [3, 4]},
    options: {
      c: {dataType: 'float32', shape: [2, 3]},
      label: label,
    },
  },
  {
    name: '[gemm] Throw if the input data type is not floating point.',
    a: {dataType: 'int32', shape: [2, 3]},
    b: {dataType: 'int32', shape: [3, 4]},
    options: {label}
  },
  {
    name:
        '[gemm] Throw if data type of inputC does not match ones of inputA and inputB.',
    a: {dataType: 'float32', shape: [3, 2]},
    b: {dataType: 'float32', shape: [4, 3]},
    options: {
      c: {dataType: 'float16', shape: [2, 4]},
      aTranspose: true,
      bTranspose: true,
      label: label,
    },
  },
  {
    name: '[gemm] Throw if the rank of inputC is 3.',
    a: {dataType: 'float32', shape: [3, 2]},
    b: {dataType: 'float32', shape: [4, 3]},
    options: {
      c: {dataType: 'float32', shape: [2, 3, 4]},
      aTranspose: true,
      bTranspose: true,
      label: label,
    },
  },
  {
    name: '[gemm] Test building gemm with dynamic dimension that matches.',
    a: {dataType: 'float32', shape: [2, {name: 'k', maxSize: 100}]},
    b: {dataType: 'float32', shape: [{name: 'k', maxSize: 100}, 4]},
    output: {dataType: 'float32', shape: [2, 4]}
  },
  {
    name: '[gemm] Test building gemm with dynamic dimension in output.',
    a: {dataType: 'float32', shape: [{name: 'batch', maxSize: 10}, 3]},
    b: {dataType: 'float32', shape: [3, 4]},
    output: {dataType: 'float32', shape: [{name: 'batch', maxSize: 10}, 4]}
  },
  {
    name: '[gemm] Test building gemm with multiple dynamic dimensions.',
    a: {
      dataType: 'float32',
      shape: [{name: 'batch', maxSize: 10}, {name: 'k', maxSize: 100}]
    },
    b: {
      dataType: 'float32',
      shape: [{name: 'k', maxSize: 100}, {name: 'n', maxSize: 50}]
    },
    output: {
      dataType: 'float32',
      shape: [{name: 'batch', maxSize: 10}, {name: 'n', maxSize: 50}]
    }
  },
  {
    name:
        '[gemm] Throw if dynamic dimension in a does not match static dimension in b.',
    a: {dataType: 'float32', shape: [2, {name: 'k', maxSize: 100}]},
    b: {dataType: 'float32', shape: [5, 4]},
    options: {label}
  },
  {
    name:
        '[gemm] Throw if static dimension in a does not match dynamic dimension in b.',
    a: {dataType: 'float32', shape: [2, 3]},
    b: {dataType: 'float32', shape: [{name: 'k', maxSize: 100}, 4]},
    options: {label}
  },
  {
    name:
        '[gemm] Throw if different dynamic dimensions are used for contracting dimension.',
    a: {dataType: 'float32', shape: [2, {name: 'k1', maxSize: 100}]},
    b: {dataType: 'float32', shape: [{name: 'k2', maxSize: 100}, 4]},
    options: {label}
  },
  {
    name:
        '[gemm] Test building gemm with aTranspose=true and dynamic dimensions.',
    a: {
      dataType: 'float32',
      shape: [{name: 'k', maxSize: 100}, {name: 'batch', maxSize: 10}]
    },
    b: {dataType: 'float32', shape: [{name: 'k', maxSize: 100}, 4]},
    options: {
      aTranspose: true,
    },
    output: {dataType: 'float32', shape: [{name: 'batch', maxSize: 10}, 4]}
  },
  {
    name:
        '[gemm] Test building gemm with bTranspose=true and dynamic dimensions.',
    a: {dataType: 'float32', shape: [2, {name: 'k', maxSize: 100}]},
    b: {
      dataType: 'float32',
      shape: [{name: 'n', maxSize: 50}, {name: 'k', maxSize: 100}]
    },
    options: {
      bTranspose: true,
    },
    output: {dataType: 'float32', shape: [2, {name: 'n', maxSize: 50}]}
  },
];

tests.forEach(
    test => promise_test(async t => {
      const builder = new MLGraphBuilder(context);
      const a = builder.input('a', test.a);
      const b = builder.input('b', test.b);
      if (test.options && test.options.c) {
        test.options.c = builder.input('c', test.options.c);
      }
      if (test.output) {
        const output = builder.gemm(a, b, test.options);
        assert_equals(output.dataType, test.output.dataType);
        assert_equals(output.shape.length, test.output.shape.length);
        for (let i = 0; i < output.shape.length; ++i) {
          const outputDim = output.shape[i];
          const expectedDim = test.output.shape[i];
          if (typeof expectedDim === 'object') {
            assert_equals(typeof outputDim, 'object');
            assert_equals(outputDim.name, expectedDim.name);
            assert_equals(outputDim.maxSize, expectedDim.maxSize);
          } else {
            assert_equals(outputDim, expectedDim);
          }
        }
      } else {
        const regrexp = new RegExp('\\[' + label + '\\]');
        assert_throws_with_label(
            () => builder.gemm(a, b, test.options), regrexp);
      }
    }, test.name));
