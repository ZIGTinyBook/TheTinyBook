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

// Define the Optimizer struct with the optimizer function, learning rate, and allocator
pub fn Optimizer(comptime T: type, func: fn (comptime type, f64, *const std.mem.Allocator) type, lr: f64, allocator: *const std.mem.Allocator) type {
    const optim = func(T, lr, allocator){};
    return struct {
        optimizer: func(T, lr, allocator) = optim, // Instantiation of the optimizer (e.g., SGD, Adam)

        pub fn step(self: *@This(), model: *Model.Model(T, allocator)) !void {
            // Directly call the optimizer's step function
            try self.optimizer.step(model);
        }
    };
}

// Define the SGD optimizer
pub fn optimizer_SGD(T: type, lr: f64, allocator: *const std.mem.Allocator) type {
    return struct {
        learning_rate: f64 = lr,
        allocator: *const std.mem.Allocator = allocator,

        // Step function to update weights and biases using gradients
        pub fn step(self: *@This(), model: *Model.Model(T, allocator)) !void {
            var counter: u32 = 0;
            for (model.layers) |*dense_layer| {
                const weight_gradients = &dense_layer.w_gradients;
                const bias_gradients = &dense_layer.b_gradients;
                const weight = &dense_layer.weights;
                const bias = &dense_layer.bias;

                std.debug.print("\n ------ step {}", .{counter});
                counter += 1;

                try self.update_tensor(weight, weight_gradients);
                try self.update_tensor(bias, bias_gradients);
            }
        }

        // Helper function to update tensors
        fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void {
            if (t.size != gradients.size) return errors.InputTensorDifferentSize;

            for (t.data, 0..) |*value, i| {
                value.* -= gradients.data[i] * self.learning_rate;
            }
        }
    };
}

pub fn optimizer_ADAMTEST(T: type, lr: f64, allocator: *const std.mem.Allocator) type {
    return struct {
        learning_rate: f64 = lr,
        allocator: *const std.mem.Allocator = allocator,

        // Step function to update weights and biases using gradients
        pub fn step(self: *@This(), model: *Model.Model(T, allocator)) !void {
            for (model.layers) |*dense_layer| {
                const weight_gradients = &dense_layer.w_gradients;
                try self.update_tensor(&dense_layer.weights, weight_gradients);
            }
        }

        // Helper function to update tensors
        fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void {
            if (t.size != gradients.size) return errors.InputTensorDifferentSize;

            for (t.data, 0..) |*value, i| {
                value.* -= gradients.data[i] * self.learning_rate;
            }
        }
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator; // Constant pointer to the allocator

    var model = Model.Model(f64, &allocator){ // Initialize the model with the correct allocator
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
    try dense_layer.init(3, 2, &rng); // Layer with 3 inputs and 2 neurons
    try model.addLayer(&dense_layer);

    std.debug.print("Weights before:\n", .{});
    dense_layer.weights.info();

    // Create an instance of the optimizer_SGD

    // Initialize the Optimizer struct, passing the sgd_optimizer instance
    var optimizer1 = Optimizer(f64, optimizer_SGD, 0.01, &allocator){ // Here we pass the actual instance of the optimizer
    };

    std.debug.print("AFTERRRRRR\n", .{});
    try optimizer1.step(&model);

    std.debug.print("\nWeights after:\n", .{});
    dense_layer.weights.info();

    model.deinit();
}
