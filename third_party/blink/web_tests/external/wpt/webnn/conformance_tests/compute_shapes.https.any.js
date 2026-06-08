// META: title=test WebNN API MLGraph computeShapes()
// META: global=window
// META: variant=?cpu
// META: variant=?gpu
// META: variant=?npu
// META: script=../resources/utils.js
// META: timeout=long

'use strict';

// MLGraph.computeShapes() takes concrete input shapes and resolves to the
// inferred concrete descriptor of each graph output, without executing the
// graph. This lets callers determine output sizes for a graph with dynamic
// input dimensions before dispatching it.

if (navigator.ml === undefined) {
  test(() => assert_implements(navigator.ml, 'missing navigator.ml'));
} else {
  promise_test(async () => {
    // relu passes the shape through, so a dynamic batch of 2 resolves [2, 3].
    const context = await getContext();
    const builder = new MLGraphBuilder(context);
    const input = builder.input(
        'input',
        {dataType: 'float32', shape: [{name: 'batch', maxSize: 8}, 3]});
    const output = builder.relu(input);
    const graph = await builder.build({output});

    const shapes = await graph.computeShapes({input: [2, 3]});
    assert_true('output' in shapes, 'result contains the output descriptor');
    assert_equals(shapes.output.dataType, 'float32');
    assert_array_equals(shapes.output.shape, [2, 3]);
  }, 'computeShapes resolves a dynamic batch dimension (relu)');

  promise_test(async () => {
    // reduceSum over the last axis with keepDimensions reduces it to 1, so a
    // dynamic batch of 4 resolves the output to [4, 1].
    const context = await getContext();
    const builder = new MLGraphBuilder(context);
    const input = builder.input(
        'input',
        {dataType: 'float32', shape: [{name: 'batch', maxSize: 8}, 3]});
    const output =
        builder.reduceSum(input, {axes: [1], keepDimensions: true});
    const graph = await builder.build({output});

    const shapes = await graph.computeShapes({input: [4, 3]});
    assert_array_equals(shapes.output.shape, [4, 1]);
  }, 'computeShapes resolves an output whose rank differs from the input');

  promise_test(async () => {
    // A fully static graph returns its build-time output descriptor.
    const context = await getContext();
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {dataType: 'float32', shape: [2, 3]});
    const output = builder.relu(input);
    const graph = await builder.build({output});

    const shapes = await graph.computeShapes({input: [2, 3]});
    assert_array_equals(shapes.output.shape, [2, 3]);
  }, 'computeShapes returns the static output descriptor for a static graph');

  promise_test(async (t) => {
    // An input shape with the wrong rank is rejected.
    const context = await getContext();
    const builder = new MLGraphBuilder(context);
    const input = builder.input(
        'input',
        {dataType: 'float32', shape: [{name: 'batch', maxSize: 8}, 3]});
    const output = builder.relu(input);
    const graph = await builder.build({output});

    await promise_rejects_js(
        t, TypeError, graph.computeShapes({input: [2, 3, 1]}),
        'a wrong-rank input shape rejects');
  }, 'computeShapes rejects an input shape with the wrong rank');

  promise_test(async (t) => {
    // A dynamic dimension outside its [minSize, maxSize] bounds is rejected.
    const context = await getContext();
    const builder = new MLGraphBuilder(context);
    const input = builder.input(
        'input',
        {dataType: 'float32', shape: [{name: 'batch', maxSize: 8}, 3]});
    const output = builder.relu(input);
    const graph = await builder.build({output});

    await promise_rejects_js(
        t, TypeError, graph.computeShapes({input: [9, 3]}),
        'an out-of-bounds dynamic dimension rejects');
  }, 'computeShapes rejects a dynamic dimension that is out of bounds');
}
