const std = @import("std");
const layer = @import("layers");
const Model = @import("model").Model;
const ActivationType = @import("activation_function").ActivationType;

pub fn getDenseLayer(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    n_inputs: usize,
    n_neurons: usize,
    rng: *std.Random.Xoshiro256,
    activation: ActivationType
) *layer.Layer {
    var innerLayer = layer.DenseLayer(T, &allocator){
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
        .activationFunction = activation,
    };

    var layer = layer.Layer(T, &allocator){
        .denseLayer = &innerLayer,
    };

    try layer1_.init(n_inputs, n_neurons, rng);

    return layer;
}

pub fn getSequentialModel(
    allocator: *const std.mem.Allocator,
    layers: []layer.Layer
) *Model {
    var model = Model(f64, &allocator) {
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    for (layers) |layer| {
        model.addLayer(&layer);
    }

    return model;
}
