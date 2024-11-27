const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layers");
const Model = @import("model").Model;
const loader = @import("dataloader");
const ActivationType = @import("activation_function").ActivationType;
const LossType = @import("loss").LossType;
const Trainer = @import("trainer");

pub fn main() !void {
    const allocator = std.heap.raw_c_allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

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
    //layer 1: 784 inputs, 64 neurons
    var layer1_ = layer.DenseLayer(f64, &allocator).create(&layer1);
    try layer1_.init(784, 64);
    try model.addLayer(&layer1_);

    var layer1Activ = layer.ActivationLayer(f64, &allocator){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var layer1_act = layer.ActivationLayer(f64, &allocator).create(&layer1Activ);
    try layer1_act.init(64, 64);
    try model.addLayer(&layer1_act);

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
    //layer 2: 64 inputs, 64 neurons
    var layer2_ = layer.DenseLayer(f64, &allocator).create(&layer2);
    try layer2_.init(64, 64);
    try model.addLayer(&layer2_);

    var layer2Activ = layer.ActivationLayer(f64, &allocator){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.ReLU,
        .allocator = &allocator,
    };
    var layer2_act = layer.ActivationLayer(f64, &allocator).create(&layer2Activ);
    try layer2_act.init(64, 64);
    try model.addLayer(&layer2_act);

    var layer3 = layer.DenseLayer(f64, &allocator){
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
    var layer3_ = layer.DenseLayer(f64, &allocator).create(&layer3);
    try layer3_.init(64, 10);
    try model.addLayer(&layer3_);

    var layer3Activ = layer.ActivationLayer(f64, &allocator){
        .input = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .activationFunction = ActivationType.Softmax,
        .allocator = &allocator,
    };
    var layer3_act = layer.ActivationLayer(f64, &allocator).create(&layer3Activ);
    try layer3_act.init(10, 10);
    try model.addLayer(&layer3_act);

    var load = loader.DataLoader(f64, u8, u8, 10){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const image_file_name: []const u8 = "t10k-images-idx3-ubyte";
    const label_file_name: []const u8 = "t10k-labels-idx1-ubyte";

    try load.loadMNISTDataParallel(&allocator, image_file_name, label_file_name);

    try Trainer.TrainDataLoader(
        f64, //The data type for the tensor elements in the model
        u8, //The data type for the input tensor (X)
        u8, //The data type for the output tensor (Y)
        &allocator, //Memory allocator for dynamic allocations during training
        10, //The number of samples in each batch
        784, //The number of features in each input sample
        &model, //A pointer to the model to be trained
        &load, //A pointer to the `DataLoader` that provides data batches
        1, //The total number of epochs to train for
        LossType.CCE, //The type of loss function used during training
        0.005, //The learning rate for model optimization
    );

    model.deinit();
}
