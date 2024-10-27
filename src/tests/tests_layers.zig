const std = @import("std");
const DenseLayer = @import("layers").DenseLayer;
const Layer = @import("layers").Layer;
const layer_ = @import("layers");

const tensor = @import("tensor");
const ActivationType = @import("activation_function").ActivationType;

test "Layer test description" {
    std.debug.print("\n--- Running Layer tests\n", .{});
}

test "Rand n and zeros" {
    var rng = std.Random.Xoshiro256.init(12345);

    const randomArray = try layer_.randn(f32, 5, 5, &rng);
    const zerosArray = try layer_.zeros(f32, 5, 5);

    //test dimension
    try std.testing.expectEqual(randomArray.len, 5);
    try std.testing.expectEqual(randomArray[0].len, 5);
    try std.testing.expectEqual(zerosArray.len, 5);
    try std.testing.expectEqual(zerosArray[0].len, 5);

    //test values
    for (0..5) |i| {
        for (0..5) |j| {
            try std.testing.expect(randomArray[i][j] != 0.0);
            try std.testing.expect(zerosArray[i][j] == 0.0);
        }
    }
}

test "DenseLayer forward and backward test" {
    std.debug.print("\n     test: DenseLayer forward test ", .{});
    const allocator = &std.testing.allocator;
    var rng = std.Random.Xoshiro256.init(12345);

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var dense_layer = DenseLayer(f64, allocator){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = allocator,
        .activationFunction = ActivationType.ReLU,
    };

    var layer1 = Layer(f64, allocator){
        .denseLayer = &dense_layer,
    };

    // n_input = 4, n_neurons= 2
    try layer1.init(4, 2, &rng);
    defer layer1.deinit();

    // Define an input tensor with 5x4 shape, an input for each neuron
    var inputArray: [5][4]f64 = [_][4]f64{
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, 6.0, 2.0 },
        [_]f64{ 14.0, 15.0, 16.0, 12.0 },
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, 6.0, 2.0 },
    };
    var shape: [2]usize = [_]usize{ 5, 4 };

    var input_tensor = try tensor.Tensor(f64).fromArray(allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    const output_tensor = try layer1.forward(&input_tensor);
    try std.testing.expectEqual(output_tensor.shape[0], 5);
    try std.testing.expectEqual(output_tensor.shape[1], 2);

    // Check that after forward, output does not contain zeros
    for (0..5) |i| {
        for (0..2) |j| {
            try std.testing.expect(output_tensor.data[i * 2 + j] != 0.0);
        }
    }

    // Test backward, create array with right dimensions and random values as gradients
    var gradArray: [5][2]f64 = [_][2]f64{
        [_]f64{ 0.1, 0.2 },
        [_]f64{ 0.3, 0.4 },
        [_]f64{ 0.5, 0.6 },
        [_]f64{ 0.7, 0.8 },
        [_]f64{ 0.9, 1.0 },
    };
    var gradShape: [2]usize = [_]usize{ 5, 2 };

    var grad = try tensor.Tensor(f64).fromArray(allocator, &gradArray, &gradShape);
    defer grad.deinit();

    _ = try layer1.backward(&grad);

    // Check that bias and gradients are valid (non-zero)
    for (0..2) |i| {
        try std.testing.expect(try layer1.denseLayer.bias.get(i) != 0.0);
        for (0..4) |j| {
            try std.testing.expect(try layer1.denseLayer.w_gradients.get(i + j) != 0.0);
        }
    }
}

test "test getters " {
    std.debug.print("\n     test: getters ", .{});
    const allocator = &std.testing.allocator;
    var rng = std.Random.Xoshiro256.init(12345);

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var dense_layer = DenseLayer(f64, allocator){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = allocator,
        .activationFunction = ActivationType.ReLU,
    };

    var layer1 = Layer(f64, allocator){
        .denseLayer = &dense_layer,
    };

    // n_input = 4, n_neurons= 2
    try layer1.init(4, 2, &rng);
    defer layer1.deinit();

    // Define an input tensor with 5x4 shape, an input for each neuron
    var inputArray: [5][4]f64 = [_][4]f64{
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, 6.0, 2.0 },
        [_]f64{ 14.0, 15.0, 16.0, 12.0 },
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, 6.0, 2.0 },
    };
    var shape: [2]usize = [_]usize{ 5, 4 };

    var input_tensor = try tensor.Tensor(f64).fromArray(allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    var output_tensor = try layer1.forward(&input_tensor);
    defer output_tensor.deinit();

    // Test backward, create array with right dimensions and random values as gradients
    var gradArray: [5][2]f64 = [_][2]f64{
        [_]f64{ 0.1, 0.2 },
        [_]f64{ 0.3, 0.4 },
        [_]f64{ 0.5, 0.6 },
        [_]f64{ 0.7, 0.8 },
        [_]f64{ 0.9, 1.0 },
    };
    var gradShape: [2]usize = [_]usize{ 5, 2 };

    var grad = try tensor.Tensor(f64).fromArray(allocator, &gradArray, &gradShape);
    defer grad.deinit();

    _ = try layer1.backward(&grad);

    //check n_inputs
    try std.testing.expect(dense_layer.n_inputs == try layer1.get_n_inputs());

    //check n_neurons
    try std.testing.expect(dense_layer.n_neurons == try layer1.get_n_neurons());

    //check get_weights
    for (0..dense_layer.weights.data.len) |i| {
        try std.testing.expect(dense_layer.weights.data[i] == (try layer1.get_weights()).data[i]);
    }

    //check get_bias
    for (0..dense_layer.bias.data.len) |i| {
        try std.testing.expect(dense_layer.bias.data[i] == (try layer1.get_bias()).data[i]);
    }

    //check get_input
    for (0..dense_layer.input.data.len) |i| {
        try std.testing.expect(dense_layer.input.data[i] == (try layer1.get_input()).data[i]);
    }

    //check get_output
    for (0..dense_layer.output.data.len) |i| {
        try std.testing.expect(dense_layer.output.data[i] == (try layer1.get_output()).data[i]);
    }

    //check get_outputActivation
    for (0..dense_layer.output.data.len) |i| {
        try std.testing.expect(dense_layer.outputActivation.data[i] == (try layer1.get_outputActivation()).data[i]);
    }

    //check get_activationFunction
    try std.testing.expect(dense_layer.activationFunction == try layer1.get_activationFunction());

    //check get_weightGradients
    for (0..dense_layer.w_gradients.data.len) |i| {
        try std.testing.expect(dense_layer.w_gradients.data[i] == (try layer1.get_weightGradients()).data[i]);
    }

    //check get_biasGradients
    for (0..dense_layer.b_gradients.data.len) |i| {
        try std.testing.expect(dense_layer.b_gradients.data[i] == (try layer1.get_biasGradients()).data[i]);
    }
}
