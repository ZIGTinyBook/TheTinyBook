const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layers");
const Model = @import("model").Model;
const ActivationType = @import("activation_function").ActivationType;

test "Model with multiple layers forward test" {
    std.debug.print("\n     test: Model with multiple layers forward test", .{});
    const allocator = std.heap.page_allocator;

    var model = Model(f64, f64, f64, &allocator, 0.05){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    var rng = std.Random.Xoshiro256.init(12345);

    var layer1 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .input = undefined,
        .bias = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
        .activationFunction = ActivationType.ReLU,
    };
    var layer1_ = layer.Layer(f64, &allocator){
        .denseLayer = &layer1,
    };
    try layer1_.init(3, 2, &rng);
    try model.addLayer(&layer1_);

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
        .activationFunction = ActivationType.ReLU,
    };
    var layer2_ = layer.Layer(f64, &allocator){
        .denseLayer = &layer2,
    };
    try layer2_.init(2, 3, &rng);
    try model.addLayer(&layer2_);

    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    var output = try model.forward(&input_tensor);

    //std.debug.print("Output tensor shape: {}\n", .{output.shape});
    //std.debug.print("Output tensor data: {}\n", .{output.data});
    output.info();

    model.deinit();
}

// test "Model with multiple layers training test" {
//     std.debug.print("\n     test: Model with multiple layers training test", .{});
//     const allocator = std.heap.page_allocator;

//     var model = Model(f64, f64, f64, &allocator, 0.05){
//         .layers = undefined,
//         .allocator = &allocator,
//         .input_tensor = undefined,
//     };
//     try model.init();

//     var rng = std.Random.Xoshiro256.init(12345);

//     var layer1 = layer.DenseLayer(f64, &allocator){
//         .weights = undefined,
//         .bias = undefined,
//         .input = undefined,
//         .output = undefined,
//         .outputActivation = undefined,
//         .n_inputs = 0,
//         .n_neurons = 0,
//         .w_gradients = undefined,
//         .b_gradients = undefined,
//         .allocator = undefined,
//         .activationFunction = ActivationType.ReLU,
//     };
//     //layer 1: 3 inputs, 2 neurons
//     var layer1_ = layer.Layer(f64, &allocator){
//         .denseLayer = &layer1,
//     };
//     try layer1_.init(3, 2, &rng);
//     try model.addLayer(&layer1_);

//     var layer2 = layer.DenseLayer(f64, &allocator){
//         .weights = undefined,
//         .bias = undefined,
//         .input = undefined,
//         .output = undefined,
//         .outputActivation = undefined,
//         .n_inputs = 0,
//         .n_neurons = 0,
//         .w_gradients = undefined,
//         .b_gradients = undefined,
//         .allocator = undefined,
//         .activationFunction = ActivationType.ReLU,
//     };
//     //layer 2: 2 inputs, 5 neurons
//     var layer2_ = layer.Layer(f64, &allocator){
//         .denseLayer = &layer2,
//     };
//     try layer2_.init(2, 5, &rng);
//     try model.addLayer(&layer2_);

//     var inputArray: [2][3]f64 = [_][3]f64{
//         [_]f64{ 1.0, 2.0, 3.0 },
//         [_]f64{ 4.0, 5.0, 6.0 },
//     };
//     var shape: [2]usize = [_]usize{ 2, 3 };

//     var targetArray: [2][5]f64 = [_][5]f64{
//         [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 },
//         [_]f64{ 4.0, 5.0, 6.0, 4.0, 5.0 },
//     };
//     var targetShape: [2]usize = [_]usize{ 2, 5 };

//     var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
//     defer input_tensor.deinit();

//     var target_tensor = try tensor.Tensor(f64).fromArray(&allocator, &targetArray, &targetShape);
//     defer target_tensor.deinit();

//     try model.train(&input_tensor, &target_tensor, 100);

//     //std.debug.print("Output tensor shape: {any}\n", .{output.shape});
//     //std.debug.print("Output tensor data: {any}\n", .{output.data});

//     model.deinit();
// }
