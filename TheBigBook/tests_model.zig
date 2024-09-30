const std = @import("std");
const tensor = @import("tensor.zig");
const layer = @import("layers.zig");
const Model = @import("model.zig").Model;

test "Model with multiple layers forward test" {
    std.debug.print("\n     test: Model with multiple layers forward test", .{});
    const allocator = std.testing.allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
    };
    try model.init();

    var rng = std.Random.Xoshiro256.init(12345);

    var layer1 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .weightShape = undefined,
        .biasShape = undefined,
        .allocator = undefined,
    };
    try layer1.init(3, 2, &rng);
    try model.addLayer(&layer1);

    var layer2 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .weightShape = undefined,
        .biasShape = undefined,
        .allocator = undefined,
    };
    try layer2.init(2, 3, &rng);
    try model.addLayer(&layer2);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    const output = try model.forward(&input_tensor);

    std.debug.print("Output tensor shape: {any}\n", .{output.shape});
    std.debug.print("Output tensor data: {any}\n", .{output.data});

    model.deinit();
}
