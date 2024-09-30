const std = @import("std");
const DenseLayer = @import("layers.zig").DenseLayer;
const tensor = @import("tensor.zig");

test "DenseLayer forward test" {
    const allocator = &std.testing.allocator;

    var rng = std.Random.Xoshiro256.init(12345);

    // Definition of the DenseLayer with 4 inputs and 2 neurons
    var dense_layer = DenseLayer(f64, allocator){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .weightShape = undefined,
        .biasShape = undefined,
        .allocator = allocator,
    };

    // n_input = 4, n_neurons= 2
    try dense_layer.init(4, 2, &rng);

    std.debug.print("Pesi e bias inizializzati\n", .{});

    //Define an input tensor with 2x4 shape, an input for each neuron
    var inputArray: [2][4]f64 = [_][4]f64{
        [_]f64{ 1.0, 2.0, 3.0, 1.0 },
        [_]f64{ 4.0, 5.0, 6.0, 2.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 4 };

    var input_tensor = try tensor.Tensor(f64).fromArray(allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    _ = try dense_layer.forward(&input_tensor);

    try std.testing.expectEqual(dense_layer.output.shape[0], 2);
    try std.testing.expectEqual(dense_layer.output.shape[1], 2);

    try std.testing.expect(dense_layer.output.data[0] != 0);
    try std.testing.expect(dense_layer.output.data[1] != 0);

    dense_layer.deinit();
    input_tensor.deinit();
}
