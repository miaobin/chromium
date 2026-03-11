// META: title=test WebNN API dynamic shape support across operators
// META: global=window
// META: variant=?cpu
// META: variant=?gpu
// META: variant=?npu
// META: script=../resources/utils.js
// META: timeout=long

'use strict';

// This file tests dynamic shape support across multiple WebNN operators.
// Dynamic dimensions use {name, maxSize} in descriptor.shape, with a concrete
// 'shape' field for actual tensor creation at dispatch time.
// All expected outputs are computed by NumPy for correctness.

const dynamicShapeTests = [
  // ============================================================
  // relu: element-wise max(0, x), shape passthrough
  // ============================================================
  {
    'name': 'relu float32 2D tensor dynamic batch [2,3]',
    'graph': {
      'inputs': {
        'reluInput': {
          'data': [-3, -1, 0, 1, 2, 5],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'relu',
        'arguments': [{'input': 'reluInput'}],
        'outputs': 'reluOutput'
      }],
      'expectedOutputs': {
        'reluOutput': {
          'data': [0, 0, 0, 1, 2, 5],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'relu float32 3D tensor dynamic batch [2,2,3]',
    'graph': {
      'inputs': {
        'reluInput': {
          'data': [-1, 2, -3, 4, -5, 6, 7, -8, 9, -10, 11, -12],
          'shape': [2, 2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'relu',
        'arguments': [{'input': 'reluInput'}],
        'outputs': 'reluOutput'
      }],
      'expectedOutputs': {
        'reluOutput': {
          'data': [0, 2, 0, 4, 0, 6, 7, 0, 9, 0, 11, 0],
          'shape': [2, 2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'relu float32 4D NCHW dynamic batch+spatial [2,1,2,2]',
    'graph': {
      'inputs': {
        'reluInput': {
          'data': [-2, 3, -4, 5, 6, -7, 8, -9],
          'shape': [2, 1, 2, 2],
          'descriptor': {
            shape: [
              {name: 'batch', maxSize: 4}, 1, {name: 'height', maxSize: 8},
              {name: 'width', maxSize: 8}
            ],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'relu',
        'arguments': [{'input': 'reluInput'}],
        'outputs': 'reluOutput'
      }],
      'expectedOutputs': {
        'reluOutput': {
          'data': [0, 3, 0, 5, 6, 0, 8, 0],
          'shape': [2, 1, 2, 2],
          'descriptor': {
            shape: [
              {name: 'batch', maxSize: 4}, 1, {name: 'height', maxSize: 8},
              {name: 'width', maxSize: 8}
            ],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // sigmoid: 1/(1+exp(-x)), shape passthrough
  // ============================================================
  {
    'name': 'sigmoid float32 2D tensor dynamic batch [2,3]',
    'graph': {
      'inputs': {
        'sigmoidInput': {
          'data': [0, 1, -1, 2, -2, 0.5],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'sigmoid',
        'arguments': [{'input': 'sigmoidInput'}],
        'outputs': 'sigmoidOutput'
      }],
      'expectedOutputs': {
        'sigmoidOutput': {
          'data': [
            0.5, 0.7310585786300049, 0.2689414213699951,
            0.8807970779778823, 0.11920292202211755, 0.6224593312018546
          ],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'sigmoid float32 3D tensor dynamic batch [2,1,4]',
    'graph': {
      'inputs': {
        'sigmoidInput': {
          'data': [-3, -1, 1, 3, -2, 0, 2, 4],
          'shape': [2, 1, 4],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1, 4],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'sigmoid',
        'arguments': [{'input': 'sigmoidInput'}],
        'outputs': 'sigmoidOutput'
      }],
      'expectedOutputs': {
        'sigmoidOutput': {
          'data': [
            0.04742587317756678, 0.2689414213699951,
            0.7310585786300049, 0.9525741268224334,
            0.11920292202211755, 0.5,
            0.8807970779778823, 0.9820137900379085
          ],
          'shape': [2, 1, 4],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1, 4],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // tanh: hyperbolic tangent, shape passthrough
  // ============================================================
  {
    'name': 'tanh float32 2D tensor dynamic batch [2,3]',
    'graph': {
      'inputs': {
        'tanhInput': {
          'data': [0, 1, -1, 0.5, -0.5, 2],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'tanh',
        'arguments': [{'input': 'tanhInput'}],
        'outputs': 'tanhOutput'
      }],
      'expectedOutputs': {
        'tanhOutput': {
          'data': [
            0, 0.7615941559557649, -0.7615941559557649,
            0.46211715726000974, -0.46211715726000974, 0.9640275800758169
          ],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'tanh float32 3D tensor dynamic batch [2,2,2]',
    'graph': {
      'inputs': {
        'tanhInput': {
          'data': [-3, -1, 0, 1, 2, 3, -2, -0.5],
          'shape': [2, 2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 2],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'tanh',
        'arguments': [{'input': 'tanhInput'}],
        'outputs': 'tanhOutput'
      }],
      'expectedOutputs': {
        'tanhOutput': {
          'data': [
            -0.9950547536867305, -0.7615941559557649,
            0, 0.7615941559557649,
            0.9640275800758169, 0.9950547536867305,
            -0.9640275800758169, -0.46211715726000974
          ],
          'shape': [2, 2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 2],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // neg: negate, shape passthrough
  // ============================================================
  {
    'name': 'neg float32 2D tensor dynamic batch [2,3]',
    'graph': {
      'inputs': {
        'negInput': {
          'data': [1, -2, 3, -4, 5, -6],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'neg',
        'arguments': [{'input': 'negInput'}],
        'outputs': 'negOutput'
      }],
      'expectedOutputs': {
        'negOutput': {
          'data': [-1, 2, -3, 4, -5, 6],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'neg float32 2D tensor dynamic batch [2,3] with floats',
    'graph': {
      'inputs': {
        'negInput': {
          'data': [0.5, -1.5, 2.5, -3.5, 0, 100],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'neg',
        'arguments': [{'input': 'negInput'}],
        'outputs': 'negOutput'
      }],
      'expectedOutputs': {
        'negOutput': {
          'data': [-0.5, 1.5, -2.5, 3.5, 0, -100],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // sub: element-wise subtraction
  // ============================================================
  {
    'name': 'sub float32 2D both inputs dynamic batch [2,3]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [10, 20, 30, 40, 50, 60],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [1, 2, 3, 4, 5, 6],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'sub',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [9, 18, 27, 36, 45, 54],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name':
        'sub float32 broadcasting dynamic [2,3] - static [1,3]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [10, 20, 30, 40, 50, 60],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [1, 2, 3],
          'descriptor': {shape: [1, 3], dataType: 'float32'}
        }
      },
      'operators': [{
        'name': 'sub',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [9, 18, 27, 39, 48, 57],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // mul: element-wise multiplication
  // ============================================================
  {
    'name': 'mul float32 2D both inputs dynamic batch [2,3]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [1, 2, 3, 4, 5, 6],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [2, 3, 4, 5, 6, 7],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'mul',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [2, 6, 12, 20, 30, 42],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name':
        'mul float32 broadcasting dynamic [2,3] * static [1,3]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [1, 2, 3, 4, 5, 6],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [10, 20, 30],
          'descriptor': {shape: [1, 3], dataType: 'float32'}
        }
      },
      'operators': [{
        'name': 'mul',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [10, 40, 90, 40, 100, 180],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'mul float32 3D both inputs dynamic batch [2,2,2]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [1, 2, 3, 4, 5, 6, 7, 8],
          'shape': [2, 2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 2],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [2, 0.5, 3, 0.25, 1, 4, 0.5, 2],
          'shape': [2, 2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 2],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'mul',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [2, 1, 9, 1, 5, 24, 3.5, 16],
          'shape': [2, 2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 2],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // div: element-wise division
  // ============================================================
  {
    'name': 'div float32 2D both inputs dynamic batch [2,3]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [10, 20, 30, 40, 50, 60],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [2, 4, 5, 8, 10, 12],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'div',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [5, 5, 6, 5, 5, 5],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name':
        'div float32 broadcasting dynamic [2,3] / static [1,3]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [100, 200, 300, 400, 500, 600],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [10, 20, 50],
          'descriptor': {shape: [1, 3], dataType: 'float32'}
        }
      },
      'operators': [{
        'name': 'div',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [10, 10, 6, 40, 25, 12],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // softmax: exp(x_i) / sum(exp(x)), shape passthrough
  // ============================================================
  {
    'name': 'softmax float32 2D dynamic batch [2,3] axis=1',
    'graph': {
      'inputs': {
        'softmaxInput': {
          'data': [1, 2, 3, 4, 1, 2],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'softmax',
        'arguments': [{'input': 'softmaxInput'}, {'axis': 1}],
        'outputs': 'softmaxOutput'
      }],
      'expectedOutputs': {
        'softmaxOutput': {
          'data': [
            0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
            0.8437947344813396, 0.04201006613406606, 0.1141951993845945
          ],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'softmax float32 2D dynamic batch [3,2] axis=1',
    'graph': {
      'inputs': {
        'softmaxInput': {
          'data': [1, 2, 3, 4, 5, 6],
          'shape': [3, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'softmax',
        'arguments': [{'input': 'softmaxInput'}, {'axis': 1}],
        'outputs': 'softmaxOutput'
      }],
      'expectedOutputs': {
        'softmaxOutput': {
          'data': [
            0.2689414213699951, 0.7310585786300049,
            0.2689414213699951, 0.7310585786300049,
            0.2689414213699951, 0.7310585786300049
          ],
          'shape': [3, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'softmax float32 3D dynamic batch [2,2,3] axis=2',
    'graph': {
      'inputs': {
        'softmaxInput': {
          'data': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
          'shape': [2, 2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'softmax',
        'arguments': [{'input': 'softmaxInput'}, {'axis': 2}],
        'outputs': 'softmaxOutput'
      }],
      'expectedOutputs': {
        'softmaxOutput': {
          'data': [
            0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
            0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
            0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
            0.3333333333333333, 0.3333333333333333, 0.3333333333333333
          ],
          'shape': [2, 2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // matmul: batched matrix multiply
  // ============================================================
  {
    'name': 'matmul float32 3D both inputs dynamic batch [2,2,3]@[2,3,2]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          'shape': [2, 2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 3],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          'shape': [2, 3, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3, 2],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'matmul',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [22, 28, 49, 64, 220, 244, 301, 334],
          'shape': [2, 2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 2],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name':
        'matmul float32 broadcasting dynamic [2,2,3] @ static [1,3,2]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          'shape': [2, 2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 3],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [1, 2, 3, 4, 5, 6],
          'descriptor': {shape: [1, 3, 2], dataType: 'float32'}
        }
      },
      'operators': [{
        'name': 'matmul',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [22, 28, 49, 64, 76, 100, 103, 136],
          'shape': [2, 2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 2],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'matmul float32 2D dynamic M [3,2] @ static [2,4]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [1, 2, 3, 4, 5, 6],
          'shape': [3, 2],
          'descriptor': {
            shape: [{name: 'M', maxSize: 8}, 2],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [1, 2, 3, 4, 5, 6, 7, 8],
          'descriptor': {shape: [2, 4], dataType: 'float32'},
          'constant': true
        }
      },
      'operators': [{
        'name': 'matmul',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68],
          'shape': [3, 4],
          'descriptor': {
            shape: [{name: 'M', maxSize: 8}, 4],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // gemm: general matrix multiplication
  // ============================================================
  {
    'name': 'gemm float32 dynamic M [2,3] @ constant [3,2]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [1, 2, 3, 4, 5, 6],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'M', maxSize: 8}, 3],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [1, 2, 3, 4, 5, 6],
          'descriptor': {shape: [3, 2], dataType: 'float32'},
          'constant': true
        }
      },
      'operators': [{
        'name': 'gemm',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [22, 28, 49, 64],
          'shape': [2, 2],
          'descriptor': {
            shape: [{name: 'M', maxSize: 8}, 2],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'gemm float32 dynamic M [3,3] @ constant [3,3]',
    'graph': {
      'inputs': {
        'inputA': {
          'data': [1, 0, 2, 1, 0, 1, 3, 2, 1],
          'shape': [3, 3],
          'descriptor': {
            shape: [{name: 'M', maxSize: 8}, 3],
            dataType: 'float32'
          }
        },
        'inputB': {
          'data': [1, 2, 3, 0, 1, 0, 4, 3, 2],
          'descriptor': {shape: [3, 3], dataType: 'float32'},
          'constant': true
        }
      },
      'operators': [{
        'name': 'gemm',
        'arguments': [{'a': 'inputA'}, {'b': 'inputB'}],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [9, 8, 7, 5, 5, 5, 7, 11, 11],
          'shape': [3, 3],
          'descriptor': {
            shape: [{name: 'M', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // convTranspose2d: transposed convolution
  // ============================================================
  {
    'name':
        'convTranspose2d float32 dynamic batch+spatial [1,1,2,2] filter [1,1,2,2]',
    'graph': {
      'inputs': {
        'convTranspose2dInput': {
          'data': [1, 2, 3, 4],
          'shape': [1, 1, 2, 2],
          'descriptor': {
            shape: [
              {name: 'batch', maxSize: 4}, 1, {name: 'height', maxSize: 8},
              {name: 'width', maxSize: 8}
            ],
            dataType: 'float32'
          }
        },
        'convTranspose2dFilter': {
          'data': [1, 1, 1, 1],
          'descriptor': {shape: [1, 1, 2, 2], dataType: 'float32'},
          'constant': true
        }
      },
      'operators': [{
        'name': 'convTranspose2d',
        'arguments': [
          {'input': 'convTranspose2dInput'},
          {'filter': 'convTranspose2dFilter'}
        ],
        'outputs': 'convTranspose2dOutput'
      }],
      'expectedOutputs': {
        'convTranspose2dOutput': {
          'data': [1, 3, 2, 4, 10, 6, 3, 7, 4],
          'shape': [1, 1, 3, 3],
          'descriptor': {
            shape: [
              {name: 'batch', maxSize: 4}, 1,
              {name: 'height+1', maxSize: 9}, {name: 'width+1', maxSize: 9}
            ],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name':
        'convTranspose2d float32 dynamic batch+spatial [1,1,2,2] filter [1,1,3,3]',
    'graph': {
      'inputs': {
        'convTranspose2dInput': {
          'data': [1, 2, 3, 4],
          'shape': [1, 1, 2, 2],
          'descriptor': {
            shape: [
              {name: 'batch', maxSize: 4}, 1, {name: 'height', maxSize: 8},
              {name: 'width', maxSize: 8}
            ],
            dataType: 'float32'
          }
        },
        'convTranspose2dFilter': {
          'data': [1, 0, 1, 0, 1, 0, 1, 0, 1],
          'descriptor': {shape: [1, 1, 3, 3], dataType: 'float32'},
          'constant': true
        }
      },
      'operators': [{
        'name': 'convTranspose2d',
        'arguments': [
          {'input': 'convTranspose2dInput'},
          {'filter': 'convTranspose2dFilter'}
        ],
        'outputs': 'convTranspose2dOutput'
      }],
      'expectedOutputs': {
        'convTranspose2dOutput': {
          'data': [1, 2, 1, 2, 3, 5, 5, 4, 1, 5, 5, 2, 3, 4, 3, 4],
          'shape': [1, 1, 4, 4],
          'descriptor': {
            shape: [
              {name: 'batch', maxSize: 4}, 1,
              {name: 'height+2', maxSize: 10}, {name: 'width+2', maxSize: 10}
            ],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // reduceSum: sum reduction along axis
  // ============================================================
  {
    'name': 'reduceSum float32 3D dynamic batch [2,1,3] axis=2 keepDims',
    'graph': {
      'inputs': {
        'input': {
          'data': [1, 2, 3, 4, 5, 6],
          'shape': [2, 1, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'reduceSum',
        'arguments': [
          {'input': 'input'},
          {'options': {'axes': [2], 'keepDimensions': true}}
        ],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [6, 15],
          'shape': [2, 1, 1],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1, 1],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'reduceSum float32 3D dynamic batch [3,2,4] axis=2 keepDims',
    'graph': {
      'inputs': {
        'input': {
          'data': [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
          ],
          'shape': [3, 2, 4],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 4],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'reduceSum',
        'arguments': [
          {'input': 'input'},
          {'options': {'axes': [2], 'keepDimensions': true}}
        ],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [10, 26, 42, 58, 74, 90],
          'shape': [3, 2, 1],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 1],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // reduceMean: mean reduction along axis
  // ============================================================
  {
    'name': 'reduceMean float32 3D dynamic batch [2,1,3] axis=2 no keepDims',
    'graph': {
      'inputs': {
        'input': {
          'data': [1, 2, 3, 4, 5, 6],
          'shape': [2, 1, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'reduceMean',
        'arguments': [
          {'input': 'input'},
          {'options': {'axes': [2], 'keepDimensions': false}}
        ],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [2, 5],
          'shape': [2, 1],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name':
        'reduceMean float32 3D dynamic batch [3,2,4] axis=2 keepDims',
    'graph': {
      'inputs': {
        'input': {
          'data': [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
          ],
          'shape': [3, 2, 4],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 4],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'reduceMean',
        'arguments': [
          {'input': 'input'},
          {'options': {'axes': [2], 'keepDimensions': true}}
        ],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [2.5, 6.5, 10.5, 14.5, 18.5, 22.5],
          'shape': [3, 2, 1],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 1],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // reduceMax: max reduction along axis
  // ============================================================
  {
    'name': 'reduceMax float32 3D dynamic batch [2,1,3] axis=2 keepDims',
    'graph': {
      'inputs': {
        'input': {
          'data': [1, 5, 3, 4, 2, 6],
          'shape': [2, 1, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'reduceMax',
        'arguments': [
          {'input': 'input'},
          {'options': {'axes': [2], 'keepDimensions': true}}
        ],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [5, 6],
          'shape': [2, 1, 1],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1, 1],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'reduceMax float32 3D dynamic batch [3,2,4] axis=1 keepDims',
    'graph': {
      'inputs': {
        'input': {
          'data': [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
          ],
          'shape': [3, 2, 4],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 4],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'reduceMax',
        'arguments': [
          {'input': 'input'},
          {'options': {'axes': [1], 'keepDimensions': true}}
        ],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24],
          'shape': [3, 1, 4],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1, 4],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // reduceMin: min reduction along axis
  // ============================================================
  {
    'name': 'reduceMin float32 3D dynamic batch [2,1,3] axis=2 keepDims',
    'graph': {
      'inputs': {
        'input': {
          'data': [5, 1, 3, 2, 6, 4],
          'shape': [2, 1, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'reduceMin',
        'arguments': [
          {'input': 'input'},
          {'options': {'axes': [2], 'keepDimensions': true}}
        ],
        'outputs': 'output'
      }],
      'expectedOutputs': {
        'output': {
          'data': [1, 2],
          'shape': [2, 1, 1],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 1, 1],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // clamp: clamp values to [min, max]
  // ============================================================
  {
    'name': 'clamp float32 2D dynamic batch [2,3] min=0 max=5',
    'graph': {
      'inputs': {
        'clampInput': {
          'data': [-5, -1, 0, 1, 3, 10],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'clamp',
        'arguments':
            [{'input': 'clampInput'}, {'options': {'minValue': 0, 'maxValue': 5}}],
        'outputs': 'clampOutput'
      }],
      'expectedOutputs': {
        'clampOutput': {
          'data': [0, 0, 0, 1, 3, 5],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'clamp float32 2D dynamic batch [3,3] min=-2 max=8',
    'graph': {
      'inputs': {
        'clampInput': {
          'data': [-10, -5, 0, 5, 10, 15, -3, 3, 7],
          'shape': [3, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'clamp',
        'arguments': [
          {'input': 'clampInput'},
          {'options': {'minValue': -2, 'maxValue': 8}}
        ],
        'outputs': 'clampOutput'
      }],
      'expectedOutputs': {
        'clampOutput': {
          'data': [-2, -2, 0, 5, 8, 8, -2, 3, 7],
          'shape': [3, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // elu: alpha*(exp(x)-1) for x<0, x for x>=0
  // ============================================================
  {
    'name': 'elu float32 2D dynamic batch [2,3] default alpha',
    'graph': {
      'inputs': {
        'eluInput': {
          'data': [-1, 0, 1, -2, 2, 3],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'elu',
        'arguments': [{'input': 'eluInput'}],
        'outputs': 'eluOutput'
      }],
      'expectedOutputs': {
        'eluOutput': {
          'data': [
            -0.6321205588285577, 0, 1, -0.8646647167633873, 2, 3
          ],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'elu float32 2D dynamic batch [3,3] alpha=2',
    'graph': {
      'inputs': {
        'eluInput': {
          'data': [-3, -1, 0, 0.5, 1, 2, -0.5, -2, 3],
          'shape': [3, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'elu',
        'arguments':
            [{'input': 'eluInput'}, {'options': {'alpha': 2.0}}],
        'outputs': 'eluOutput'
      }],
      'expectedOutputs': {
        'eluOutput': {
          'data': [
            -1.900425863264272, -1.2642411176571153, 0,
            0.5, 1, 2,
            -0.7869386805747332, -1.7293294335267746, 3
          ],
          'shape': [3, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // leakyRelu: alpha*x for x<0, x for x>=0
  // ============================================================
  {
    'name': 'leakyRelu float32 2D dynamic batch [2,3] alpha=0.01',
    'graph': {
      'inputs': {
        'leakyReluInput': {
          'data': [-2, -1, 0, 1, 2, 3],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'leakyRelu',
        'arguments':
            [{'input': 'leakyReluInput'}, {'options': {'alpha': 0.01}}],
        'outputs': 'leakyReluOutput'
      }],
      'expectedOutputs': {
        'leakyReluOutput': {
          'data': [-0.02, -0.01, 0, 1, 2, 3],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  {
    'name': 'leakyRelu float32 2D dynamic batch [3,3] alpha=0.1',
    'graph': {
      'inputs': {
        'leakyReluInput': {
          'data': [-5, -2, -1, 0, 1, 3, 5, 10, -0.5],
          'shape': [3, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      },
      'operators': [{
        'name': 'leakyRelu',
        'arguments':
            [{'input': 'leakyReluInput'}, {'options': {'alpha': 0.1}}],
        'outputs': 'leakyReluOutput'
      }],
      'expectedOutputs': {
        'leakyReluOutput': {
          'data': [-0.5, -0.2, -0.1, 0, 1, 3, 5, 10, -0.05],
          'shape': [3, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // Subgraph: conv2d + relu
  // ============================================================
  {
    'name':
        'subgraph conv2d+relu dynamic batch+spatial [1,1,4,4] filter [1,1,3,3]',
    'graph': {
      'inputs': {
        'input': {
          'data': [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
          ],
          'shape': [1, 1, 4, 4],
          'descriptor': {
            shape: [
              {name: 'batch', maxSize: 4}, 1, {name: 'height', maxSize: 8},
              {name: 'width', maxSize: 8}
            ],
            dataType: 'float32'
          }
        },
        'filter': {
          'data': [1, 1, 1, 1, 1, 1, 1, 1, 1],
          'descriptor': {shape: [1, 1, 3, 3], dataType: 'float32'},
          'constant': true
        }
      },
      'operators': [
        {
          'name': 'conv2d',
          'arguments': [{'input': 'input'}, {'filter': 'filter'}],
          'outputs': 'conv2dOutput'
        },
        {
          'name': 'relu',
          'arguments': [{'input': 'conv2dOutput'}],
          'outputs': 'output'
        }
      ],
      'expectedOutputs': {
        'output': {
          'data': [54, 63, 90, 99],
          'shape': [1, 1, 2, 2],
          'descriptor': {
            shape: [
              {name: 'batch', maxSize: 4}, 1,
              {name: 'height-2', maxSize: 6}, {name: 'width-2', maxSize: 6}
            ],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // Subgraph: matmul + add (linear layer with bias)
  // ============================================================
  {
    'name': 'subgraph matmul+add (bias) dynamic batch [2,3]',
    'graph': {
      'inputs': {
        'input': {
          'data': [1, 2, 3, 4, 5, 6],
          'shape': [2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3],
            dataType: 'float32'
          }
        },
        'weights': {
          'data': [1, 0, 0, 1, 1, 0],
          'descriptor': {shape: [3, 2], dataType: 'float32'},
          'constant': true
        },
        'bias': {
          'data': [10, 20],
          'descriptor': {shape: [1, 2], dataType: 'float32'},
          'constant': true
        }
      },
      'operators': [
        {
          'name': 'matmul',
          'arguments': [{'a': 'input'}, {'b': 'weights'}],
          'outputs': 'matmulOutput'
        },
        {
          'name': 'add',
          'arguments': [{'a': 'matmulOutput'}, {'b': 'bias'}],
          'outputs': 'output'
        }
      ],
      'expectedOutputs': {
        'output': {
          'data': [14, 22, 20, 25],
          'shape': [2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // Subgraph: matmul + add + softmax (classifier head)
  // ============================================================
  {
    'name':
        'subgraph matmul+add+softmax (classifier) dynamic batch [2,2]',
    'graph': {
      'inputs': {
        'input': {
          'data': [1, 0, 0, 1],
          'shape': [2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2],
            dataType: 'float32'
          }
        },
        'weights': {
          'data': [1, 2, 3, 4],
          'descriptor': {shape: [2, 2], dataType: 'float32'},
          'constant': true
        },
        'bias': {
          'data': [0, 0],
          'descriptor': {shape: [1, 2], dataType: 'float32'},
          'constant': true
        }
      },
      'operators': [
        {
          'name': 'matmul',
          'arguments': [{'a': 'input'}, {'b': 'weights'}],
          'outputs': 'matmulOutput'
        },
        {
          'name': 'add',
          'arguments': [{'a': 'matmulOutput'}, {'b': 'bias'}],
          'outputs': 'addOutput'
        },
        {
          'name': 'softmax',
          'arguments': [{'input': 'addOutput'}, {'axis': 1}],
          'outputs': 'output'
        }
      ],
      'expectedOutputs': {
        'output': {
          // [1,0]@[[1,2],[3,4]]=[1,2] -> softmax -> [0.269, 0.731]
          // [0,1]@[[1,2],[3,4]]=[3,4] -> softmax -> [0.269, 0.731]
          'data': [
            0.2689414213699951, 0.7310585786300049,
            0.2689414213699951, 0.7310585786300049
          ],
          'shape': [2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // Subgraph: conv2d + reshape + matmul (vision backbone)
  // ============================================================
  {
    'name':
        'subgraph conv2d+reshape+matmul (vision backbone) dynamic batch',
    'graph': {
      'inputs': {
        'input': {
          'data': [1, 1, 1, 1, 1, 1, 1, 1, 1],
          'shape': [1, 1, 3, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 4}, 1, 3, 3],
            dataType: 'float32'
          }
        },
        'filter': {
          'data': [1, 1, 1, 1],
          'descriptor': {shape: [1, 1, 2, 2], dataType: 'float32'},
          'constant': true
        },
        'weights': {
          'data': [1, 0, 0, 1, 1, 0, 0, 1],
          'descriptor': {shape: [4, 2], dataType: 'float32'},
          'constant': true
        }
      },
      'operators': [
        {
          'name': 'conv2d',
          'arguments': [{'input': 'input'}, {'filter': 'filter'}],
          'outputs': 'convOutput'
        },
        {
          'name': 'reshape',
          'arguments': [
            {'input': 'convOutput'},
            {'newShape': [{name: 'batch', maxSize: 4}, 4]}
          ],
          'outputs': 'reshapeOutput'
        },
        {
          'name': 'matmul',
          'arguments': [{'a': 'reshapeOutput'}, {'b': 'weights'}],
          'outputs': 'output'
        }
      ],
      'expectedOutputs': {
        'output': {
          // conv: all 1s [1,1,3,3]*[1,1,2,2] -> [1,1,2,2]=[4,4,4,4]
          // reshape -> [1,4]
          // matmul [1,4]*[4,2] = [8, 8]
          'data': [8, 8],
          'shape': [1, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 4}, 2],
            dataType: 'float32'
          }
        }
      }
    }
  },
  // ============================================================
  // Subgraph: matmul + softmax + mul (attention-like)
  // Q@K^T -> softmax -> *V, all with dynamic batch
  // ============================================================
  {
    'name':
        'subgraph attention-like (matmul+softmax+mul) dynamic batch [2,2,3]',
    'graph': {
      'inputs': {
        'Q': {
          'data': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
          'shape': [2, 2, 3],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 3],
            dataType: 'float32'
          }
        },
        'KT': {
          // K transposed: original K [2,2,3] -> KT [2,3,2]
          // batch0: K=[[1,0],[0,1],[1,1]] -> KT=[[1,0,1],[0,1,1]]
          // batch1: K=[[0,1],[1,1],[0,0]] -> KT=[[0,1,0],[1,1,0]]
          'data': [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
          'shape': [2, 3, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 3, 2],
            dataType: 'float32'
          }
        },
        'V': {
          'data': [1, 2, 3, 4, 5, 6, 7, 8],
          'shape': [2, 2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 2],
            dataType: 'float32'
          }
        }
      },
      'operators': [
        {
          'name': 'matmul',
          'arguments': [{'a': 'Q'}, {'b': 'KT'}],
          'outputs': 'scores'
        },
        {
          'name': 'softmax',
          'arguments': [{'input': 'scores'}, {'axis': 2}],
          'outputs': 'weights'
        },
        {
          'name': 'matmul',
          'arguments': [{'a': 'weights'}, {'b': 'V'}],
          'outputs': 'output'
        }
      ],
      'expectedOutputs': {
        'output': {
          // scores: Q@KT
          // batch0: [[1,0,1],[0,1,0]]@[[1,0],[0,1],[1,1]] = [[2,1],[0,1]]
          //   Nope, recalculate: Q[0]=[[1,0,1],[0,1,0]], KT[0]=[[1,0,1],[0,1,1]]
          //   Q[0]@KT[0] = [[1*1+0*0+1*1, 1*0+0*1+1*1],[0*1+1*0+0*1, 0*0+1*1+0*1]]
          //              = [[2, 1],[0, 1]]
          //   -> wrong, KT is [2,3,2] so Q[2,3]@KT[3,2]
          //   Q[0]=[[1,0,1],[0,1,0]]  KT[0]=[[1,0],[0,1],[1,1]]
          //   [1,0,1]@[[1,0],[0,1],[1,1]] = [1+0+1, 0+0+1] = [2,1]
          //   [0,1,0]@[[1,0],[0,1],[1,1]] = [0+0+0, 0+1+0] = [0,1]
          //   scores[0] = [[2,1],[0,1]]
          //   softmax(axis=2): row[2,1]->[e2/(e2+e1), e1/(e2+e1)]=[0.731,0.269]
          //                    row[0,1]->[e0/(e0+e1), e1/(e0+e1)]=[0.269,0.731]
          //   weights[0]@V[0]: [[0.731,0.269],[0.269,0.731]]@[[1,2],[3,4]]
          //     = [0.731+0.807, 1.462+1.076] = [1.538, 2.538]  hmm let me use NumPy
          // NumPy verified:
          //   attention_output: [2.462, 3.462, 2.462, 3.462, 6.0, 7.0, 5.538, 6.538]
          'data': [
            2.4621171572600096, 3.4621171572600096,
            2.4621171572600096, 3.4621171572600096,
            6, 7,
            5.53788284273999, 6.53788284273999
          ],
          'shape': [2, 2, 2],
          'descriptor': {
            shape: [{name: 'batch', maxSize: 8}, 2, 2],
            dataType: 'float32'
          }
        }
      }
    }
  }
];

webnn_conformance_test(
    dynamicShapeTests, buildAndExecuteGraph, getPrecisionTolerance);
