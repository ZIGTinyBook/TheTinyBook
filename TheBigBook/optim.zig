const std = @import("std");
const tensor = @import("tensor.zig");
const layer = @import("layers.zig");
const Model = @import("model.zig");

const Optimizers = enum {
    SGD,
    Adam,
    RMSprop,
};

const errors = error{
    UnsupportedType,
    InputTensorDifferentSize,
};

pub fn Optimizer(comptime T: type, func: fn (comptime type, f64, *const std.mem.Allocator) type, lr: f64, allocator: *const std.mem.Allocator) type {
    return struct {
        optimizer: func(T, lr, allocator), // Qui stai costruendo effettivamente l'istanza dell'ottimizzatore
        optType: Optimizers,

        pub fn step(self: *@This(), model: *Model.Model(T, allocator)) !void {
            switch (self.optType) {
                Optimizers.SGD => {
                    try self.optimizer.step(model);
                },
                else => {
                    return errors.UnsupportedType;
                },
            }
        }
    };
}

pub fn optimizer_SGD(T: type, lr: f64, allocator: *const std.mem.Allocator) type {
    return struct {
        learning_rate: f64 = lr,
        allocator: *const std.mem.Allocator = allocator,

        pub fn step(self: *@This(), model: *Model.Model(T, allocator)) !void {
            for (model.layers) |*dense_layer| {
                const weight_gradients = &dense_layer.w_gradients;
                const bias_gradients = &dense_layer.b_gradients;

                try self.update_tensor(&dense_layer.weights, weight_gradients);
                try self.update_tensor(&dense_layer.bias, bias_gradients);
            }
        }

        fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void {
            if (t.size != gradients.size) return errors.InputTensorDifferentSize;

            for (t.data, 0..) |*value, i| {
                value.* -= gradients.data[i] * self.learning_rate;
            }
        }
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator; // Usa un puntatore costante all'allocatore

    var model = Model.Model(f64, &allocator){ // Inizializza il modello con il corretto allocatore
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
        .allocator = &allocator,
    };
    try dense_layer.init(3, 2, &rng);
    try model.addLayer(&dense_layer);

    std.debug.print("Weights before:\n", .{});
    dense_layer.weights.info();

    // Inizializzazione dell'ottimizzatore SGD con allocatore corretto
    var optimizer = optimizer_SGD(f64, 0.01, &allocator){};

    // Inizializzazione di Optimizer e set del campo optimizer con allocatore corretto
    var optimizer1 = Optimizer(f64, optimizer_SGD, 0.01, &allocator){
        .optType = Optimizers.SGD,
        .optimizer = optimizer, // Qui assegni l'istanza dell'ottimizzatore
    };

    try optimizer.step(&model);
    std.debug.print("AFTERRRRRR", .{});
    try optimizer1.step(&model);

    std.debug.print("\nWeights after:\n", .{});
    dense_layer.weights.info();

    model.deinit();
}
