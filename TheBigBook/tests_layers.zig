const std = @import("std");
const DenseLayer = @import("layers.zig").DenseLayer;
const tensor = @import("tensor.zig");
const randn = @import("layers.zig").randn;
const zeros = @import("layers.zig").zeros;
const TensMath = @import("tensor_math.zig");
const Architectures = @import("architectures.zig");

test "DenseLayer init and forward test" {
    const allocator = std.testing.allocator;

    var rng = std.rand.Random.Xoshiro256.init(12345);

    var dense_layer = DenseLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .weightShape = undefined,
        .biasShape = undefined,
    };

    try dense_layer.init(2, 2, &rng);

    std.debug.print("Dopo l'inizializzazione, pesi e bias sono stati stampati\n", .{});

    var inputArray: [2][2]f64 = [_][2]f64{
        [_]f64{ 1.0, 2.0 },
        [_]f64{ 4.0, 5.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 2 };

    var t1 = try tensor.Tensor(f64).init(&allocator, shape[0..]);
    _ = try t1.fromArray(&allocator, &inputArray, shape[0..]);

    // Chiamiamo la funzione forward del DenseLayer
    _ = try dense_layer.forward(t1);

    // Stampiamo l'output dopo il forward pass
    std.debug.print("Dopo il forward pass, l'output è: {any}\n", .{dense_layer.output});

    // Verifica delle dimensioni dell'output
    try std.testing.expectEqual(dense_layer.output.shape[0], 2);
    try std.testing.expectEqual(dense_layer.output.shape[1], 2);

    // Verifica dei valori di output
    try std.testing.expect(dense_layer.output.data[0] != 0);
    try std.testing.expect(dense_layer.output.data[1] != 0);

    // Pulizia della memoria allocata
    dense_layer.output.deinit();
    t1.deinit();
}
