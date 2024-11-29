const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layers");
const Model = @import("model").Model;
const ActivationType = @import("activation_function").ActivationType;
const Trainer = @import("trainer");

test "Multiple layers training test" {
    std.debug.print("\n     test: Multiple layers training test", .{});
    const allocator = std.testing.allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    //layer 1: 3 inputs, 2 neurons
    var layer1 = layer.DenseLayer(f64, &allocator){
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
    var layer1_ = layer.DenseLayer(f64, &allocator).create(&layer1);
    try layer1_.init(3, 2);
    try model.addLayer(layer1_);

    //layer 1: 3 inputs, 2 neurons
    var layer1Activ = layer.ActivationLayer(f64, &allocator){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var layer1Activ_ = layer.ActivationLayer(f64, &allocator).create(&layer1Activ);
    try layer1Activ_.init(2, 2);
    try model.addLayer(layer1Activ_);

    //layer 2: 2 inputs, 5 neurons
    var layer2 = layer.DenseLayer(f64, &allocator){
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
    var layer2_ = layer.DenseLayer(f64, &allocator).create(&layer2);
    try layer2_.init(2, 5);
    try model.addLayer(layer2_);

    var layer2Activ = layer.ActivationLayer(f64, &allocator){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.Softmax,
        .allocator = &allocator,
    };
    var layer2Activ_ = layer.ActivationLayer(f64, &allocator).create(&layer2Activ);
    try layer2Activ_.init(2, 5);
    try model.addLayer(layer2Activ_);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var targetArray: [2][5]f64 = [_][5]f64{
        [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 },
        [_]f64{ 4.0, 5.0, 6.0, 4.0, 5.0 },
    };
    var targetShape: [2]usize = [_]usize{ 2, 5 };

    var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer {
        input_tensor.deinit();
        std.debug.print("\n -.-.-> input_tensor deinitialized", .{});
    }

    var target_tensor = try tensor.Tensor(f64).fromArray(&allocator, &targetArray, &targetShape);
    defer {
        target_tensor.deinit();
        std.debug.print("\n -.-.-> target_tensor deinitialized", .{});
    }

    try Trainer.trainTensors(
        f64, //type
        &allocator, //allocator
        &model, //model
        &input_tensor, //input
        &target_tensor, //target
        10, //epochs
        0.5, //learning rate
    );

    model.deinit();
}
