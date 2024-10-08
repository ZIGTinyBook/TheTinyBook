const std = @import("std");
const tensor = @import("tensor.zig");
const layer = @import("layers.zig");
const Model = @import("model.zig").Model;

test "Model with multiple layers forward test" {
    std.debug.print("\n     test: Model with multiple layers forward test", .{});
    const allocator = std.heap.page_allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    var rng = std.Random.Xoshiro256.init(12345);

    var layer1 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
        .activation = undefined,
    };
    try layer1.init(3, 2, &rng, "ReLU");
    try model.addLayer(&layer1);

    var layer2 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
        .activation = undefined,
    };
    try layer2.init(2, 3, &rng, "ReLU");
    try model.addLayer(&layer2);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    var output = try model.forward(&input_tensor);

    //std.debug.print("Output tensor shape: {}\n", .{output.shape});
    //std.debug.print("Output tensor data: {}\n", .{output.data});
    output.info();

    model.deinit();
}

test "Model with multiple layers training test" {
    std.debug.print("\n     test: Model with multiple layers training test", .{});
    const allocator = std.heap.page_allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    var rng = std.Random.Xoshiro256.init(12345);

    var layer1 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
        .activation = undefined,
    };
    //layer 1: 3 inputs, 2 neurons
    try layer1.init(3, 2, &rng, "ReLU");
    try model.addLayer(&layer1);

    var layer2 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
        .activation = undefined,
    };
    //layer 2: 2 inputs, 3 neurons
    try layer2.init(2, 3, &rng, "ReLU");
    try model.addLayer(&layer2);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    var target_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer target_tensor.deinit();

    try model.train(&input_tensor, &target_tensor, 2);

    //std.debug.print("Output tensor shape: {any}\n", .{output.shape});
    //std.debug.print("Output tensor data: {any}\n", .{output.data});

    model.deinit();
}
