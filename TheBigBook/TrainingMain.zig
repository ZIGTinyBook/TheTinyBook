const std = @import("std");
const tensor = @import("tensor.zig");
const layer = @import("layers.zig");
const Model = @import("model.zig").Model;
const loader = @import("dataLoader.zig");

pub fn main() !void {
    std.debug.print("\n     test: Model with multiple layers training test", .{});
    const allocator = std.heap.page_allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();
    var rng = std.Random.Xoshiro256.init(12345);

    var layer1 = layer.DenseLayer(f64, &allocator){
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
    try layer1.init(5, 16, &rng, "ReLU");
    try model.addLayer(&layer1);

    var layer2 = layer.DenseLayer(f64, &allocator){
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
    //layer 2: 2 inputs, 5 neurons
    try layer2.init(16, 32, &rng, "ReLU");
    try model.addLayer(&layer2);

    var layer3 = layer.DenseLayer(f64, &allocator){
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
    //layer 2: 2 inputs, 5 neurons
    try layer3.init(32, 1, &rng, "ReLU");
    try model.addLayer(&layer3);

    var load = loader.DataLoader(f64, f64, 100){
        .X = undefined,
        .y = undefined,
        .xTensor = undefined,
        .yTensor = undefined,
        .XBatch = undefined,
        .yBatch = undefined,
    };

    const file_name: []const u8 = "dataset_regressione.csv";
    const features = [_]usize{ 0, 1, 2, 3, 4 };
    const featureCols: []const usize = &features;
    const labelCol: usize = 5;
    try load.fromCSV(&allocator, file_name, featureCols, labelCol);

    try model.TrainDataLoader(&load, 10);

    //std.debug.print("Output tensor shape: {any}\n", .{output.shape});
    //std.debug.print("Output tensor data: {any}\n", .{output.data});

    model.deinit();
}
