const std = @import("std");
const model_import_export = @import("model_import_export");
const Model = @import("model").Model;
const layer = @import("layers");
const Tensor = @import("tensor").Tensor;
const ActivationType = @import("activation_function").ActivationType;

test "Export of a complex model" {
    std.debug.print("\n     test: Export of a 2D Tensor", .{});

    const allocator = std.testing.allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    const layer1 = layer.DenseLayer(f64, &allocator){
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
    //layer 1: 784 inputs, 64 neurons
    var layer1_ = layer.DenseLayer(f64, &allocator).create(layer1);
    try layer1_.init(784, 64);
    try model.addLayer(layer1_);

    const layer1Activ = layer.ActivationLayer(f64, &allocator){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var layer1_act = layer.ActivationLayer(f64, &allocator).create(layer1Activ);
    try layer1_act.init(64, 64);
    try model.addLayer(layer1_act);

    const layer2 = layer.DenseLayer(f64, &allocator){
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
    //layer 2: 64 inputs, 64 neurons
    var layer2_ = layer.DenseLayer(f64, &allocator).create(layer2);
    try layer2_.init(64, 64);
    try model.addLayer(layer2_);

    const layer2Activ = layer.ActivationLayer(f64, &allocator){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var layer2_act = layer.ActivationLayer(f64, &allocator).create(layer2Activ);
    try layer2_act.init(64, 64);
    try model.addLayer(layer2_act);

    const layer3 = layer.DenseLayer(f64, &allocator){
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
    //layer 3: 64 inputs, 10 neurons
    var layer3_ = layer.DenseLayer(f64, &allocator).create(layer3);
    try layer3_.init(64, 10);
    try model.addLayer(layer3_);

    const layer3Activ = layer.ActivationLayer(f64, &allocator){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.Softmax,
        .allocator = &allocator,
    };
    var layer3_act = layer.ActivationLayer(f64, &allocator).create(layer3Activ);
    try layer3_act.init(10, 10);
    try model.addLayer(layer3_act);

    try model_import_export.exportModel(f64, &allocator, model, "exportTryModel.bin");

    var model_imported: Model(f64, &allocator) = try model_import_export.importModel(f64, &allocator, "exportTryModel.bin");
    defer model_imported.deinit();
}
