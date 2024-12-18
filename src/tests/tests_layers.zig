const std = @import("std");
const DenseLayer = @import("denselayer").DenseLayer;
const ConvolutionalLayer = @import("convLayer").ConvolutionalLayer;
const FlattenLayer = @import("flattenLayer").FlattenLayer;
const ActivationLayer = @import("activationlayer").ActivationLayer;
const PoolingLayer = @import("poolingLayer").PoolingLayer;
const PoolingType = @import("poolingLayer").PoolingType;
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
const pkg_allocator = @import("pkgAllocator");

test "Layer test description" {
    std.debug.print("\n--- Running Layer tests\n", .{});
}

test "Rand n and zeros" {
    const allocator = &pkg_allocator.allocator;
    const randomArray = try layer_.randn(f32, allocator, 2, 2);
    defer allocator.free(randomArray);
    const zerosArray = try layer_.zeros(f32, allocator, 2, 2);
    defer allocator.free(zerosArray);

    //test dimension
    try std.testing.expectEqual(randomArray.len, 4);
    try std.testing.expectEqual(zerosArray.len, 4);

    //test values
    for (0..2) |i| {
        for (0..2) |j| {
            try std.testing.expect(randomArray[i * 2 + j] != 0.0);
            try std.testing.expect(zerosArray[i * 2 + j] == 0.0);
        }
    }
}

test "DenseLayer forward and backward test" {
    std.debug.print("\n     test: DenseLayer forward test and backward testx", .{});
    const allocator = &pkg_allocator.allocator;

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var dense_layer = DenseLayer(f64){
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
    const layer1 = DenseLayer(f64).create(&dense_layer);

    // n_input = 4, n_neurons= 2
    try layer1.init(allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 4,
        .n_neurons = 2,
    }));
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

    var backward = try layer1.backward(&grad);
    defer backward.deinit();

    // Check that bias and gradients are valid (non-zero)
    var myDense: *DenseLayer(f64) = @ptrCast(@alignCast(layer1.layer_ptr));
    for (0..2) |i| {
        try std.testing.expect(try myDense.bias.get(i) != 0.0);
        for (0..4) |j| {
            try std.testing.expect(try myDense.w_gradients.get(i + j) != 0.0);
        }
    }
}

test "test getters " {
    std.debug.print("\n     test: getters ", .{});
    const allocator = &pkg_allocator.allocator;

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var dense_layer = DenseLayer(f64){
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
    const layer1 = DenseLayer(f64).create(&dense_layer);

    // n_input = 4, n_neurons= 2
    try layer1.init(allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 4,
        .n_neurons = 2,
    }));
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

    var backward = try layer1.backward(&grad);
    defer backward.deinit();

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
    const allocator = &pkg_allocator.allocator;

    // const argsStruct = struct {
    //     n_inputs: usize,
    //     n_neurons: usize,
    // };

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var activ_layer = ActivationLayer(f64){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = allocator,
    };
    const layer1 = ActivationLayer(f64).create(&activ_layer);
    // n_input = 5, n_neurons= 4
    try layer1.init(allocator, @constCast(&struct {
        n_inputs: usize,
        n_neurons: usize,
    }{
        .n_inputs = 5,
        .n_neurons = 4,
    }));
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
    var result = try layer1.backward(&input_tensor);
    result.deinit();
}

test "Complete test of the new convolutional layer functionalities" {
    std.debug.print("\n     Test: Conv forward test ", .{});

    const allocator = &pkg_allocator.allocator;

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
    var conv_layer = ConvLayer.ConvolutionalLayer(f64){
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
    // input_channels=2, output_channels=2, kernel_size=[2,2]
    try layer.init(
        &pkg_allocator.allocator,
        @constCast(&struct {
            input_channels: usize,
            output_channels: usize,
            kernel_size: [2]usize,
        }{
            .input_channels = 2,
            .output_channels = 5,
            .kernel_size = .{ 2, 2 },
        }),
    );
    defer layer.deinit();

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

    // Clean up resourcess
    _ = &input_data;
    _ = &conv_layer;
}

test "Complete test of the Flatten layer functionalities with first dimension unflattened" {
    std.debug.print("\n     Test: Flatten layer forward and backward test ", .{});

    const allocator = &pkg_allocator.allocator;

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
    var input_shape = [_]usize{ 2, 2, 3, 3 }; // [batch_size=2, channels=2, height=3, width=3]
    var input = try Tensor(f64).fromArray(allocator, &input_data, input_shape[0..]);
    defer input.deinit();

    // Create the Flatten layer
    var flatten_layer = FlattenLayer(f64){
        .input = undefined,
        .output = undefined,
        .allocator = allocator,
    };
    var layer = flatten_layer.create();

    // Initialize the Flatten layer with placeholder args
    var init_args = FlattenLayer(f64).FlattenInitArgs{
        .placeholder = true,
    };
    try layer.init(allocator, &init_args);

    // Perform the forward pass
    std.debug.print("\nFlatten forward pass test...\n", .{});
    var output = try layer.forward(&input);
    std.debug.print("Output shape after flatten: {any}\n", .{output.shape});
    std.debug.print("Output of forward pass:\n", .{});
    output.info();

    // Verify the output size (should be [2, 18])
    // Because input is [2,2,3,3], the last dimensions product: 2*3*3=18
    // So output.shape should be [2, 18]
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 18), output.shape[1]);

    // Create a dummy gradient for the backward pass (same shape as output: [2, 18])
    var dValues_data = try allocator.alloc(f64, output.size);
    _ = &dValues_data;
    defer allocator.free(dValues_data);
    for (dValues_data) |*val| {
        val.* = 1; // Set all gradients to 1 for simplicity
    }
    var dValues = try Tensor(f64).fromArray(allocator, dValues_data, output.shape);
    defer dValues.deinit();

    // Perform the backward pass
    std.debug.print("\nFlatten backward pass test...\n", .{});
    var dInput = try layer.backward(&dValues);
    defer dInput.deinit();
    std.debug.print("dInput shape after backward (should match the original input shape): {any}\n", .{dInput.shape});
    dInput.info();

    // Verify the shapes are the same as original input
    try std.testing.expectEqualSlices(usize, dInput.shape, input.shape);

    // Clean up resources
    layer.deinit();
    _ = &input_data;
    _ = &flatten_layer;
}

test "Pooling layer forward and backward test (Max Pooling 2D, stride=1)" {
    std.debug.print("\n     Test: Pooling layer forward and backward test (Max Pooling, stride=1)\n", .{});

    const allocator = &pkg_allocator.allocator;

    // 3x3 input
    var input_data = [_]f64{
        1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,
        40.0, 50.0, 60.0,
    };
    var input_shape = [_]usize{ 3, 3 };
    var input = try tensor.Tensor(f64).fromArray(allocator, &input_data, input_shape[0..]);
    defer input.deinit();

    // Create the Pooling layer (Max Pooling) with kernel=2x2 and stride=1x1
    var pooling_layer = PoolingLayer(f64){
        .input = undefined,
        .output = undefined,
        .used_input = undefined,
        .kernel = .{ 2, 2 },
        .stride = .{ 1, 1 },
        .poolingType = .Max,
        .allocator = allocator,
    };
    var layer = try pooling_layer.create();

    const InitArgs = struct {
        kernel: [2]usize,
        stride: [2]usize,
        poolingType: PoolingType,
    };

    var init_args = InitArgs{
        .kernel = .{ 2, 2 },
        .stride = .{ 1, 1 },
        .poolingType = .Max,
    };

    // Initialize the layer
    try layer.init(allocator, &init_args);

    // Forward pass
    std.debug.print("\nPooling forward pass test...\n", .{});
    var output = try layer.forward(&input);
    //defer output.deinit();

    std.debug.print("Output shape after pooling: {any}\n", .{output.shape});
    output.info();

    // Check output shape
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[1]);

    // Check output values
    // Expected:
    // [ [5,  6],
    //   [50, 60] ]
    try std.testing.expectEqual(@as(f64, 5.0), output.data[0 * 2 + 0]);
    try std.testing.expectEqual(@as(f64, 6.0), output.data[0 * 2 + 1]);
    try std.testing.expectEqual(@as(f64, 50.0), output.data[1 * 2 + 0]);
    try std.testing.expectEqual(@as(f64, 60.0), output.data[1 * 2 + 1]);

    // Backward pass
    std.debug.print("\nPooling backward pass test...\n", .{});

    // Create dValues (2x2) and set all gradients to 1
    var dValues_data = try allocator.alloc(f64, output.size);
    _ = &dValues_data;
    defer allocator.free(dValues_data);
    for (dValues_data) |*val| {
        val.* = 1.0;
    }
    var dValues = try tensor.Tensor(f64).fromArray(allocator, dValues_data, output.shape);
    defer dValues.deinit();

    var dInput = try layer.backward(&dValues);
    defer dInput.deinit();

    std.debug.print("dInput shape after backward: {any}\n", .{dInput.shape});
    dInput.info();

    // Check that dInput shape matches the original input
    try std.testing.expectEqualSlices(usize, dInput.shape, input.shape);

    // Expected dInput:
    // [ [0,0,0],
    //   [0,1,1],
    //   [0,1,1] ]
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[0]); // (0,0)
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[1]); // (0,1)
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[2]); // (0,2)

    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[3]); // (1,0)
    try std.testing.expectEqual(@as(f64, 1.0), dInput.data[4]); // (1,1)
    try std.testing.expectEqual(@as(f64, 1.0), dInput.data[5]); // (1,2)

    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[6]); // (2,0)
    try std.testing.expectEqual(@as(f64, 1.0), dInput.data[7]); // (2,1)
    try std.testing.expectEqual(@as(f64, 1.0), dInput.data[8]); // (2,2)

    // Deinitialize the layer to avoid leaks
    layer.deinit();
    _ = &input_data;
    _ = &pooling_layer;
}

test "Pooling layer forward and backward test (Max Pooling 2D, stride=1), Multiple Gradient Distribution" {
    std.debug.print("\n     Test: Pooling layer forward and backward test (Max Pooling, stride=1)\n", .{});

    const allocator = &pkg_allocator.allocator;

    // 3x3 input
    var input_data = [_]f64{
        1.0,  2.0,  3.0,
        4.0,  70.0, 6.0,
        40.0, 50.0, 60.0,
    };
    var input_shape = [_]usize{ 3, 3 };
    var input = try tensor.Tensor(f64).fromArray(allocator, &input_data, input_shape[0..]);
    defer input.deinit();

    // Create the Pooling layer (Max Pooling) with kernel=2x2 and stride=1x1
    var pooling_layer = PoolingLayer(f64){
        .input = undefined,
        .output = undefined,
        .used_input = undefined,
        .kernel = .{ 2, 2 },
        .stride = .{ 1, 1 },
        .poolingType = .Max,
        .allocator = allocator,
    };
    var layer = try pooling_layer.create();

    const InitArgs = struct {
        kernel: [2]usize,
        stride: [2]usize,
        poolingType: PoolingType,
    };

    var init_args = InitArgs{
        .kernel = .{ 2, 2 },
        .stride = .{ 1, 1 },
        .poolingType = .Max,
    };

    // Initialize the layer
    try layer.init(allocator, &init_args);

    // Forward pass
    std.debug.print("\nPooling forward pass test...\n", .{});
    var output = try layer.forward(&input);
    //defer output.deinit();

    std.debug.print("Output shape after pooling: {any}\n", .{output.shape});
    output.info();

    // Check output shape
    try std.testing.expectEqual(@as(usize, 2), output.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), output.shape[1]);

    // Check output values
    // Expected:
    // [ [5,  6],
    //   [50, 60] ]
    try std.testing.expectEqual(@as(f64, 70.0), output.data[0 * 2 + 0]);
    try std.testing.expectEqual(@as(f64, 70.0), output.data[0 * 2 + 1]);
    try std.testing.expectEqual(@as(f64, 70.0), output.data[1 * 2 + 0]);
    try std.testing.expectEqual(@as(f64, 70.0), output.data[1 * 2 + 1]);

    // Backward pass
    std.debug.print("\nPooling backward pass test...\n", .{});

    // Create dValues (2x2) and set all gradients to 1
    var dValues_data = try allocator.alloc(f64, output.size);
    _ = &dValues_data;
    defer allocator.free(dValues_data);
    for (dValues_data) |*val| {
        val.* = 1.0;
    }
    var dValues = try tensor.Tensor(f64).fromArray(allocator, dValues_data, output.shape);
    defer dValues.deinit();

    var dInput = try layer.backward(&dValues);
    defer dInput.deinit();

    std.debug.print("dInput shape after backward: {any}\n", .{dInput.shape});
    dInput.info();

    // Check that dInput shape matches the original input
    try std.testing.expectEqualSlices(usize, dInput.shape, input.shape);

    // Expected dInput:
    // [ [0,0,0],
    //   [0,1,1],
    //   [0,1,1] ]
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[0]); // (0,0)
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[1]); // (0,1)
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[2]); // (0,2)

    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[3]); // (1,0)
    try std.testing.expectEqual(@as(f64, 4.0), dInput.data[4]); // (1,1)
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[5]); // (1,2)

    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[6]); // (2,0)
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[7]); // (2,1)
    try std.testing.expectEqual(@as(f64, 0.0), dInput.data[8]); // (2,2)

    // Deinitialize the layer to avoid leaks
    layer.deinit();
    _ = &input_data;
    _ = &pooling_layer;
}
