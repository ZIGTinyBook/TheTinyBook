const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layer");
const Model = @import("model");
const TensorMathError = @import("errorHandler").TensorMathError;
const DenseLayer = @import("denselayer");

pub const Optimizers = enum {
    SGD,
    Adam,
    RMSprop,
};

// Define the Optimizer struct with the optimizer function, learning rate
pub fn Optimizer(comptime T: type, comptime Xtype: type, comptime YType: type, func: fn (comptime type, comptime type, comptime type, f64) type, lr: f64) type {
    const optim = func(T, Xtype, YType, lr){};
    return struct {
        optimizer: func(T, Xtype, YType, lr) = optim, // Instantiation of the optimizer (e.g., SGD, Adam)

        pub fn step(self: *@This(), model: *Model.Model(T)) !void {
            // Directly call the optimizer's step function
            try self.optimizer.step(model);
        }
    };
}

// Define the SGD optimizer
// NEED TO BE MODIFIED IF NEW LAYERS ARE ADDED
pub fn optimizer_SGD(T: type, XType: type, YType: type, lr: f64) type {
    //Are XType and YType really necessary?
    _ = XType;
    _ = YType;

    return struct {
        learning_rate: f64 = lr,

        // Step function to update weights and biases using gradients
        pub fn step(self: *@This(), model: *Model.Model(T)) !void {
            var counter: u32 = 0;
            for (model.layers.items) |layer_| {
                if (layer_.layer_type == layer.LayerType.DenseLayer) {
                    const myDense: *DenseLayer.DenseLayer(T) = @ptrCast(@alignCast(layer_.layer_ptr));
                    const weight_gradients = &myDense.w_gradients;
                    const bias_gradients = &myDense.b_gradients;
                    const weight = &myDense.weights;
                    const bias = &myDense.bias;

                    std.debug.print("\n ------ step {}", .{counter});

                    try self.update_tensor(weight, weight_gradients);
                    try self.update_tensor(bias, bias_gradients);
                }
                counter += 1;
            }
        }

        // Helper function to update tensors
        fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void {
            if (t.size != gradients.size) return TensorMathError.InputTensorDifferentSize;
            //we move in the opposite direction of the gradient
            for (t.data, 0..) |*value, i| {
                value.* -= gradients.data[i] * self.learning_rate;
            }
        }
    };
}

pub fn optimizer_ADAMTEST(T: type, lr: f64) type {
    return struct {
        learning_rate: f64 = lr,

        // Step function to update weights and biases using gradients
        pub fn step(self: *@This(), model: *Model.Model(T)) !void {
            for (model.layers) |*dense_layer| {
                const weight_gradients = &dense_layer.w_gradients;
                try self.update_tensor(&dense_layer.weights, weight_gradients);
            }
        }

        // Helper function to update tensors
        fn update_tensor(self: *@This(), t: *tensor.Tensor(T), gradients: *tensor.Tensor(T)) !void {
            if (t.size != gradients.size) return TensorMathError.InputTensorDifferentSize;

            for (t.data, 0..) |*value, i| {
                value.* -= gradients.data[i] * self.learning_rate;
            }
        }
    };
}
