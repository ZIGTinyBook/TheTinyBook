const std = @import("std");
const mcw = @import("modelCreationWrapper");
const layer = @import("layers");
const Model = @import("model").Model;
const ActivationType = @import("activation_function").ActivationType;
const tensor = @import("tensor");

test "Model creation wrapper description test" {
    std.debug.print("\n--- Running model creation wrapper test\n", .{});
}

test "get dense layer" {
    std.debug.print("\n     get dense layer", .{});
    const allocator = std.heap.raw_c_allocator;
    var rng = std.Random.Xoshiro256.init(12345);

    const n_input: usize = 12;
    const n_neurons: usize = 3;
    const parameters_type: type = f64;
    const activation: ActivationType = ActivationType.ReLU;

    const result: *layer.Layer = mcw.getDenseLayer(
        parameters_type,
        &allocator,
        n_input,
        n_neurons,
        &rng,
        activation
    );

    var innerLayer = layer.DenseLayer(parameters_type, &allocator){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
        .activationFunction = activation,
    };
    var layer = layer.Layer(parameters_type, &allocator){
        .denseLayer = &innerLayer,
    };
    try layer.init(n_input, n_neurons, &rng);

    // Make sure they are initialized the same way
    var resultInnerLayer = result.denseLayer;
    innerLayer = layer.denseLayer;

    // Compare weights
    var resultWeights: tensor.Tensor(parameters_type) = resultInnerLayer.weights;
    var weights: tensor.Tensor(parameters_type) = innerLayer.weights;

    var resultShape: []usize = resultWeights.shape;
    var shape: []usize = weights.shape;

    try std.testing.expectEqual(shape.len, resultShape.len);

    for (0..weights.getSize()) |i| {
        try std.testing.expectEqual(weights.get(i), resultWeights.get(i));
    }

    // Compare biases
    var resultBias: tensor.Tensor(parameters_type) = resultInnerLayer.bias;
    var bias: tensor.Tensor(parameters_type) = innerLayer.bias;

    resultShape = resultWeights.shape;
    shape = weights.shape;

    try std.testing.expectEqual(shape.len, resultShape.len);

    for (0..bias.getSize()) |i| {
        try std.testing.expectEqual(bias.get(i), resultBias.get(i));
    }

    // Compare n_inputs and n_neurons
    try std.testing.expectEqual(innerLayer.n_inputs, resultInnerLayer.n_inputs);
    try std.testing.expectEqual(innerLayer.n_neurons, resultInnerLayer.n_neurons);

    // Compare Activation function
    try std.testing.expectEqual(innerLayer.activationFunction, resultInnerLayer.activationFunction);

    // Compare allocator (has to point to the same object)
    try std.testing.expectEqual(innerLayer.allocator, resultInnerLayer.allocator);
}

test "get sequential model" {
    std.debug.print("\n     get sequential model", .{});
    const allocator = std.heap.raw_c_allocator;
    var rng = std.Random.Xoshiro256.init(12345);
    const parameters_type: type = f64;
    const activation: ActivationType = ActivationType.ReLU;
    const final_activation: ActivationType = ActivationType.Softmax;

    const layers: [_]layer.Layer = .{
        mcw.getDenseLayer(parameters_type, &allocator, 24, 12, &rng, activation),
        mcw.getDenseLayer(parameters_type, &allocator, 12, 6, &rng, activation),
        mcw.getDenseLayer(parameters_type, &allocator, 6, 3, &rng, final_activation),
    };

    const result: *Model = mcw.getSequentialModel(&allocator, &layers);

    var model = Model(parameters_type, &allocator) {
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    for (layers) |layer| {
        model.addLayer(&layer);
    }

    // Compare layers number
    var resultLayersNumber: usize = result.layers.len;
    var layersNumber: usize = model.layers.len;

    try std.testing.expectEqual(layersNumber, resultLayersNumber);

    // Compare weights shape for each layer
    var resultShape: usize;
    var shape: usize;

    for (0..layerNumber) |i| {
        resultShape = result.layers[i].denseLayer.weights.shape;
        shape = model.layers[i].denseLayer.weights.shape;
        try std.testing.expectEqual(shape, resultShape);
    }
}