const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layer");
const denselayer = @import("denselayer");
const Model = @import("model").Model;
const ActivationType = @import("activation_function").ActivationType;
const Trainer = @import("trainer");

test "Model with multiple Denselayers forward test" {
    std.debug.print("\n     test: Model with multiple layers forward test", .{});
    const allocator = std.testing.allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();
    defer model.deinit();

    var dense_layer1 = denselayer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .input = undefined,
        .bias = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
    };
    var layer1_ = denselayer.DenseLayer(f64, &allocator).create(&dense_layer1);
    try layer1_.init(3, 2);
    try model.addLayer(layer1_);

    var dense_layer2 = denselayer.DenseLayer(f64, &allocator){
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
    var layer2_ = denselayer.DenseLayer(f64, &allocator).create(&dense_layer2);
    try layer2_.init(2, 3);
    try model.addLayer(layer2_);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer {
        input_tensor.deinit();
        std.debug.print("\n -.-.-> input_tensor deinitialized", .{});
    }

    _ = try model.forward(&input_tensor);
}
