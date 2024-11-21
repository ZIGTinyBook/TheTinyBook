const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layers");
const Model = @import("model");
const Optim = @import("optim");
const ActivationType = @import("activation_function").ActivationType;

//Test that it runs and prints the initial and updated weights must test with back prop
test "SGD Optimizer No Update with Zero Gradients (Print Only)" {
    std.debug.print("\n     test: SGD Optimizer No Update with Zero Gradients (Print Only)", .{});
    const allocator = std.testing.allocator;
    const lr: f64 = 0.05;

    var model = Model.Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    var dense_layer = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
    };
    var layer1_ = layer.DenseLayer(f64, &allocator).create(&dense_layer);
    try layer1_.init(3, 2);
    try model.addLayer(&layer1_);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    _ = try model.forward(&input_tensor);

    var optimizer = Optim.Optimizer(f64, f64, f64, Optim.optimizer_SGD, lr, &allocator){ // Here we pass the actual instance of the optimizer
    };
    try optimizer.step(&model);

    model.deinit();
}
