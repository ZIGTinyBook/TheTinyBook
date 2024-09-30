const std = @import("std");
const tensor = @import("tensor.zig");
const layer = @import("layers.zig");
const Model = @import("model.zig");
const OptimizerSGD = @import("optim.zig").optimizer_SGD;
//Test that it runs and prints the initial and updated weights must test with back prop
test "SGD Optimizer No Update with Zero Gradients (Print Only)" {
    const allocator = std.testing.allocator;

    var model = Model.Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
    };
    try model.init();

    var rng = std.rand.Random.Xoshiro256.init(12345);

    var dense_layer = layer.DenseLayer(f64, &allocator){
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
    try dense_layer.init(3, 2, &rng); // Layer con 3 input e 2 neuroni
    try model.addLayer(&dense_layer);

    // Stampa informazioni iniziali dei pesi
    std.debug.print("Weights before:\n", .{});
    dense_layer.weights.info();

    var optimizer = OptimizerSGD(f64, 0.01, &allocator){};

    try optimizer.step(&model);

    // Stampa i pesi dopo l'aggiornamento
    std.debug.print("\nWeights After:\n", .{});
    dense_layer.weights.info();

    model.deinit();
}
