const std = @import("std");
const DenseLayer = @import("layers.zig").DenseLayer;
const tensor = @import("tensor.zig");

test " DenseLayer forward test" {
    std.debug.print("\n     test: DenseLayer forward test", .{});
    const allocator = &std.heap.page_allocator;
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
        .activation = undefined,
    };

    // n_input = 4, n_neurons= 2
    try dense_layer.init(4, 2, &rng, "ReLU");
    defer dense_layer.deinit();

    //Define an input tensor with 2x4 shape, an input for each neuron
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

    var output_tensor = try dense_layer.forward(&input_tensor);

    // try std.testing.expectEqual(dense_layer.output.shape[0], 2);
    // try std.testing.expectEqual(dense_layer.output.shape[1], 2);

    // try std.testing.expect(dense_layer.outputActivation.data[0] >= 0);
    // try std.testing.expect(dense_layer.outputActivation.data[1] >= 0);

    // std.debug.print("\n>>>>>>>>>> input_tensor", .{});
    // input_tensor.info();
    // std.debug.print("\n>>>>>>>>>> dense_layer.weights", .{});
    // dense_layer.weights.info();
    // std.debug.print("\n>>>>>>>>>>  dense_layer.bias", .{});
    // dense_layer.bias.info();
    // std.debug.print("\n>>>>>>>>>> dense_layer.output", .{});
    // dense_layer.output.info();
    // std.debug.print("\n>>>>>>>>>> dense_layer.outputActivation", .{});
    // dense_layer.outputActivation.info();
    std.debug.print("\n>>>>>>>>>> output_tensor", .{});
    output_tensor.info();
}
