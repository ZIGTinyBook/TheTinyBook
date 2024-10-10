const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layers");
const Model = @import("model");
const Optim = @import("optim");

//Test that it runs and prints the initial and updated weights must test with back prop
test "SGD Optimizer No Update with Zero Gradients (Print Only)" {
    std.debug.print("\n     test: SGD Optimizer No Update with Zero Gradients (Print Only)", .{});
    const allocator = std.heap.page_allocator;
    const lr: f64 = 0.05;

    var model = Model.Model(f64, &allocator, lr){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    var rng = std.Random.Xoshiro256.init(12345);

    var dense_layer = layer.DenseLayer(f64, &allocator){
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
        .activation = undefined,
    };
    try dense_layer.init(3, 2, &rng, "ReLU"); // Layer con 3 input e 2 neuroni
    try model.addLayer(&dense_layer);

    // Stampa informazioni iniziali dei pesi
    std.debug.print("Weights before:\n", .{});
    dense_layer.weights.info();

    var optimizer = Optim.Optimizer(f64, Optim.optimizer_SGD, 0.05, &allocator){ // Here we pass the actual instance of the optimizer
    };
    try optimizer.step(&model);

    // Stampa i pesi dopo l'aggiornamento
    std.debug.print("\nWeights After:\n", .{});
    dense_layer.weights.info();

    model.deinit();
}
