const std = @import("std");
const DenseLayer = @import("denselayer").DenseLayer;
const ConvolutionalLayer = @import("convLayer").ConvolutionalLayer;
const ActivationLayer = @import("activationlayer").ActivationLayer;
const Layer = @import("layer").Layer;
const layer_ = @import("layer");
const Tensor = @import("tensor").Tensor;
const TensMath = @import("tensor_math");
const ConvLayer = @import("convLayer");
const ActivationFunction = @import("activation_function");
const LayerError = @import("errorHandler").LayerError;
const TensorError = @import("errorHandler").TensorError;
const TensorMathError = @import("errorHandler").TensorMathError;
const tensor = @import("tensor");
const ActivationType = @import("activation_function").ActivationType;

test "Layer test description" {
    std.debug.print("\n--- Running Layer tests\n", .{});
}

test "Rand n and zeros" {
    const randomArray = try layer_.randn(f32, 5, 5);
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

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var dense_layer = DenseLayer(f64, allocator){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = allocator,
    };
    const layer1 = DenseLayer(f64, allocator).create(&dense_layer);

    // n_input = 4, n_neurons= 2
    try layer1.init(4, 2);
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
    var myDense: *DenseLayer(f64, allocator) = @ptrCast(@alignCast(layer1.layer_ptr));
    for (0..2) |i| {
        try std.testing.expect(try myDense.bias.get(i) != 0.0);
        for (0..4) |j| {
            try std.testing.expect(try myDense.w_gradients.get(i + j) != 0.0);
        }
    }
}

test "test getters " {
    std.debug.print("\n     test: getters ", .{});
    const allocator = &std.testing.allocator;

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var dense_layer = DenseLayer(f64, allocator){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = allocator,
    };
    const layer1 = DenseLayer(f64, allocator).create(&dense_layer);

    // n_input = 4, n_neurons= 2
    try layer1.init(4, 2);
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

    _ = try layer1.forward(&input_tensor);
    //defer output_tensor.deinit();

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
    try std.testing.expect(dense_layer.n_inputs == layer1.get_n_inputs());

    //check n_neurons
    try std.testing.expect(dense_layer.n_neurons == layer1.get_n_neurons());

    //utils myDense cast, is the only way to access and anyopaque
    //const myDense: *DenseLayer(f64, allocator) = @ptrCast(@alignCast(layer1.layer_ptr));

    //check get_input
    for (0..dense_layer.input.data.len) |i| {
        try std.testing.expect(dense_layer.input.data[i] == layer1.get_input().data[i]);
    }

    //check get_output
    for (0..dense_layer.output.data.len) |i| {
        try std.testing.expect(dense_layer.output.data[i] == layer1.get_output().data[i]);
    }
}

test "ActivationLayer forward and backward test" {
    std.debug.print("\n     test: DenseLayer forward test ", .{});
    const allocator = &std.testing.allocator;

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var activ_layer = ActivationLayer(f64, allocator){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = allocator,
    };
    const layer1 = ActivationLayer(f64, allocator).create(&activ_layer);
    // n_input = 5, n_neurons= 4
    try layer1.init(5, 4);
    defer layer1.deinit();

    // Define an input tensor with 5x4 shape, an input for each neuron
    var inputArray: [5][4]f64 = [_][4]f64{
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, -6.0, -2.0 },
        [_]f64{ -14.0, -15.0, 16.0, 12.0 },
        [_]f64{ 1.0, 2.0, -3.0, -1.0 },
        [_]f64{ 4.0, 5.0, -6.0, -2.0 },
    };
    var shape: [2]usize = [_]usize{ 5, 4 };

    var input_tensor = try tensor.Tensor(f64).fromArray(allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    const output_tensor = try layer1.forward(&input_tensor);
    for (0..output_tensor.data.len) |i| {
        try std.testing.expect(output_tensor.data[i] >= 0);
    }

    // Test backward
    _ = try layer1.backward(&input_tensor);
}

test "Complete test of the new convolutional layer functionalities" {
    std.debug.print("\n     Test: Conv forward test ", .{});

    const allocator = &std.testing.allocator;

    // Define input data
    var input_data = [_]f64{
        // Batch 1, Channel 1
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        // Batch 1, Channel 2
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
        // Batch 2, Channel 1
        19, 20, 21,
        22, 23, 24,
        25, 26, 27,
        // Batch 2, Channel 2
        28, 29, 30,
        31, 32, 33,
        34, 35, 36,
    };
    var input_shape = [_]usize{ 2, 2, 3, 3 }; // [batch_size, in_channels, height, width]
    var input = try Tensor(f64).fromArray(allocator, &input_data, &input_shape);
    defer input.deinit();

    // Create the convolutional layer
    var conv_layer = ConvLayer.ConvolutionalLayer(f64, allocator){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .input_channels = 0,
        .output_channels = 0,
        .kernel_size = undefined,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = allocator,
    };
    var layer = conv_layer.create();

    // Initialize the convolutional layer
    try layer.convInit(2, 5, .{ 2, 2 }); // input_channels=2, output_channels=2, kernel_size=[2,2]

    // Perform the forward pass
    var output = try layer.forward(&input);
    std.debug.print("Output shape: {any}\n", .{output.shape});

    // Print the output for verification
    std.debug.print("Output of forward pass:\n", .{});
    output.info();

    // Create a dummy gradient for the backward pass (same shape as output)
    var dValues_data = try allocator.alloc(f64, output.size);
    _ = &dValues_data;
    defer allocator.free(dValues_data);
    for (dValues_data) |*val| {
        val.* = 1; // Set the gradient to 1 for simplicity
    }
    var dValues = try Tensor(f64).fromArray(allocator, dValues_data, output.shape);
    defer dValues.deinit();

    // Perform the backward pass
    var dInput = try layer.backward(&dValues);
    defer dInput.deinit();

    // Print the gradients for verification
    std.debug.print("Weight gradients:\n", .{});
    // conv_layer.w_gradients.printMultidim();
    conv_layer.w_gradients.info();

    std.debug.print("Input gradients:\n", .{});
    // dInput.printMultidim();

    // Verify the shapes
    // try std.testing.expectEqual(conv_layer.w_gradients.shape, conv_layer.weights.shape);
    // try std.testing.expectEqual(dInput.shape, input.shape);

    // Clean up resources
    layer.deinit();
    _ = &input_data;
    _ = &conv_layer;
}
