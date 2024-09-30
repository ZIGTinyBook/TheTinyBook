const std = @import("std");
const tensor = @import("tensor.zig");
const layer = @import("layers.zig");
const Model = @import("model.zig");

pub fn optimizer_SGD(T: type, lr: f64, allocator: *const std.mem.Allocator) type {
    return struct {
        learning_rate: f64 = lr,

        pub fn step(self: *@This(), model: *Model.Model(T, allocator)) !void {
            for (model.layers) |*dense_layer| {
                const weight_gradients = &dense_layer.w_gradients;
                const bias_gradients = &dense_layer.b_gradients;

                try self.update_tensor(&dense_layer.weights, weight_gradients);
                try self.update_tensor(&dense_layer.bias, bias_gradients);
            }
        }

        fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void {
            if (t.size != gradients.size) return error.InputTensorDifferentSize;

            for (t.data, 0..) |*value, i| {
                value.* -= gradients.data[i] * self.learning_rate;
            }
        }
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

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

    std.debug.print("Weights before:\n", .{});
    dense_layer.weights.info();

    var optimizer = optimizer_SGD(f64, 0.01, &allocator){};

    try optimizer.step(&model);

    std.debug.print("\nWeights afters:\n", .{});
    dense_layer.weights.info();

    model.deinit();
}
