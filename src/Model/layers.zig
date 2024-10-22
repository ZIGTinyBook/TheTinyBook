//! This file contains the definition of the layers that can be used in the neural network.
//! There are function to initiialize random weigths, initialization right now is completely random but in the future
//! it will possible to use proper initialization techniques.
//! Layer can be stacked in a model and they implement proper forward and backward methods.

const std = @import("std");
const tensor = @import("tensor");
const TensMath = @import("tensor_m");
const Architectures = @import("architectures").Architectures;
const TensorError = @import("tensor_m").TensorError;
const ArchitectureError = @import("tensor_m").ArchitectureError;
const ActivLib = @import("activation_function");
const ActivationType = @import("activation_function").ActivationType;

pub const errors = error{
    NullLayer,
};

pub const LayerTypes = enum {
    DenseLayer,
    DefaultLayer,
    null,
};

//------------------------------------------------------------------------------------------------------
/// UTILS
/// Initialize a matrix of random values with a normal distribution
pub fn randn(comptime T: type, n_inputs: usize, n_neurons: usize, rng: *std.Random.Xoshiro256) ![][]T {
    const matrix = try std.heap.page_allocator.alloc([]T, n_inputs);
    for (matrix) |*row| {
        row.* = try std.heap.page_allocator.alloc(T, n_neurons);
        for (row.*) |*value| {
            value.* = rng.random().floatNorm(T) + 1; // fix me!! why +1 ??
        }
    }
    return matrix;
}
///Function used to initialize a matrix of zeros used for bias
pub fn zeros(comptime T: type, n_inputs: usize, n_neurons: usize) ![][]T {
    const matrix = try std.heap.page_allocator.alloc([]T, n_inputs);
    for (matrix) |*row| {
        row.* = try std.heap.page_allocator.alloc(T, n_neurons);
        for (row.*) |*value| {
            value.* = 0;
        }
    }
    return matrix;
}

//------------------------------------------------------------------------------------------------------
// INTERFACE LAYER
pub fn Layer(comptime T: type, allocator: *const std.mem.Allocator) type {
    const LayerType = union(enum) {
        denseLayer: *DenseLayer(T, allocator),
        null: void,

        pub fn init(self: @This(), n_inputs: usize, n_neurons: usize, rng: *std.Random.Xoshiro256) !void {
            switch (self) {
                .null => return error.NullLayer,
                inline else => |layer| {
                    try layer.init(n_inputs, n_neurons, rng);
                    return;
                },
            }
        }
        pub fn deinit(self: @This()) void {
            switch (self) {
                .null => {},
                inline else => |layer| return layer.deinit(),
            }
        }
        pub fn forward(self: @This(), input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            switch (self) {
                .null => return error.NullLayer,
                inline else => |layer| {
                    // layer.weights.info();
                    return layer.forward(input);
                },
            }
        }

        pub fn backward(self: @This(), dValues: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            switch (self) {
                .null => return error.NullLayer,
                inline else => |layer| return layer.backward(dValues),
            }
        }

        pub fn printLayer(self: @This(), choice: u8) void {
            switch (self) {
                .null => {},
                inline else => |layer| return layer.printLayer(choice),
            }
        }
    };
    return LayerType;
}

/// Function to create a DenseLayer struct in future it will be possible to create other types of layers like convolutional, LSTM etc.
/// The DenseLayer is a fully connected layer, it has a weight matrix and a bias vector.
/// It has also an activation function that can be applied to the output, it can even be none.
pub fn DenseLayer(comptime T: type, alloc: *const std.mem.Allocator) type {
    return struct {
        //          | w11   w12  w13 |
        // weight = | w21   w22  w23 | , where Wij, i= neuron i-th and j=input j-th
        //          | w31   w32  w33 |
        weights: tensor.Tensor(T), //each row represent a neuron, where each weight is associated to an input
        bias: tensor.Tensor(T), //a bias for each neuron
        input: tensor.Tensor(T), //is saved for semplicity, it can be sobstituted
        output: tensor.Tensor(T), // output = dot(input, weight.transposed) + bias
        outputActivation: tensor.Tensor(T), // outputActivation = activationFunction(output)
        //layer shape --------------------
        n_inputs: usize,
        n_neurons: usize,
        //activation function-----------------------
        activationFunction: ActivationType,
        //gradients-----------------------
        w_gradients: tensor.Tensor(T),
        b_gradients: tensor.Tensor(T),
        //utils---------------------------
        allocator: *const std.mem.Allocator,

        ///Initilize the layer with random weights and biases
        /// also for the gradients
        pub fn init(self: *@This(), n_inputs: usize, n_neurons: usize, rng: *std.Random.Xoshiro256) !void {
            std.debug.print("Init DenseLayer: n_inputs = {}, n_neurons = {}, Type = {}\n", .{ n_inputs, n_neurons, @TypeOf(T) });

            //check on parameters
            if (n_inputs <= 0 or n_neurons <= 0) return error.InvalidParameters;

            //initializing number of neurons and inputs----------------------------------
            self.n_inputs = n_inputs;
            self.n_neurons = n_neurons;

            var weight_shape: [2]usize = [_]usize{ n_inputs, n_neurons };
            var bias_shape: [1]usize = [_]usize{n_neurons};
            self.allocator = alloc;

            //std.debug.print("Generating random weights...\n", .{});
            const weight_matrix = try randn(T, n_inputs, n_neurons, rng);
            const bias_matrix = try randn(T, 1, n_neurons, rng);

            //initializing weights and biases--------------------------------------------
            self.weights = try tensor.Tensor(T).fromArray(alloc, weight_matrix, &weight_shape);
            self.bias = try tensor.Tensor(T).fromArray(alloc, bias_matrix, &bias_shape);

            //initializing gradients to all zeros----------------------------------------
            self.w_gradients = try tensor.Tensor(T).fromShape(self.allocator, &weight_shape);
            self.b_gradients = try tensor.Tensor(T).fromShape(self.allocator, &bias_shape);
        }
        ///Deallocate the layer
        pub fn deinit(self: *@This()) void {
            //std.debug.print("Deallocating DenseLayer resources...\n", .{});

            // Dealloc tensors of weights, bias and output if allocated
            if (self.weights.data.len > 0) {
                self.weights.deinit();
            }

            if (self.bias.data.len > 0) {
                self.bias.deinit();
            }

            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            // Dealloca i tensori di gradients se alsizelocati
            if (self.w_gradients.data.len > 0) {
                self.w_gradients.deinit();
            }

            if (self.b_gradients.data.len > 0) {
                self.b_gradients.deinit();
            }

            if (self.input.data.len > 0) {
                self.input.deinit();
            }

            std.debug.print("\nDenseLayer resources deallocated.", .{});
        }
        ///Forward pass of the layer if present it applies the activation function
        /// We can improve it removing as much as possibile all the copy operations
        pub fn forward(self: *@This(), input: *tensor.Tensor(T)) !tensor.Tensor(T) {

            //this copy is necessary for the backward
            if (self.input.data.len >= 0) {
                self.input.deinit();
            }
            self.input = try input.copy();

            //self.weights.info();

            // 1. Check if self.output is already allocated, deallocate if necessary
            // if (self.output.data.len > 0) {
            //     self.output.deinit();
            // }

            // 2. Perform multiplication between inputs and weights (dot product)
            self.output = try TensMath.compute_dot_product(T, &self.input, &self.weights);

            // 3. Add bias to the dot product
            try TensMath.add_bias(T, &self.output, &self.bias);

            // 4. copy the output in to outputActivation so to be modified in the activation function
            self.outputActivation = try self.output.copy();

            // 5. Apply activation function
            // I was gettig crazy with this.activation initialization since ActivLib.ActivationFunction( something ) is
            //dynamic and we are trying to do everything at comtime, no excuses
            if (self.activationFunction == ActivationType.ReLU) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.ReLU);
                var activation = act_type{};
                try activation.forward(&self.outputActivation);
            } else if (self.activationFunction == ActivationType.Softmax) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.Softmax);
                var activation = act_type{};
                try activation.forward(&self.outputActivation);
            } else if (self.activationFunction == ActivationType.Sigmoid) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.Sigmoid);
                var activation = act_type{};
                try activation.forward(&self.outputActivation);
            }

            //PAY ATTENTION: here we return the outputActivation, so the already activated output
            return self.outputActivation;
        }
        /// Backward pass of the layer It takes the dValues from the next layer and computes the gradients
        pub fn backward(self: *@This(), dValues: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            //---- Key Steps: -----

            // 1. Apply the derivative of the activation function to dValues
            if (self.activationFunction == ActivationType.ReLU) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.ReLU);
                var activation = act_type{};
                try activation.derivate(dValues);
            } else if (self.activationFunction == ActivationType.Softmax) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.Softmax);
                var activation = act_type{};
                try activation.derivate(dValues, &self.outputActivation);
            } else if (self.activationFunction == ActivationType.Sigmoid) {
                const act_type = ActivLib.ActivationFunction(T, ActivationType.Sigmoid);
                var activation = act_type{};
                try activation.derivate(dValues);
            }

            // 2. Compute weight gradients (w_gradients)
            var input_transposed = try self.input.transpose2D();
            defer input_transposed.deinit();

            self.w_gradients.deinit();
            self.w_gradients = try TensMath.dot_product_tensor(Architectures.CPU, T, T, &input_transposed, dValues);

            // 3. Compute bias gradients (b_gradients)
            // Equivalent of np.sum(dL_dOutput, axis=0, keepdims=True)
            var sum: T = 0;
            //summing all the values in each column(neuron) of dValue and putting it into b_gradint[neuron]
            for (0..dValues.shape[1]) |neuron| {
                //scanning all the inputs
                sum = 0;
                for (0..dValues.shape[0]) |input| {
                    sum += dValues.data[input * self.n_neurons + neuron];
                }
                self.b_gradients.data[neuron] = sum;
            }

            // 4. Compute input gradients: dL/dInput = dot(dValues, weights.T)
            var dValues_transposed = try dValues.transpose2D();
            defer dValues_transposed.deinit();

            var weights_transposed = try self.weights.transpose2D();
            defer weights_transposed.deinit();

            var dL_dInput = try TensMath.dot_product_tensor(Architectures.CPU, T, T, dValues, &weights_transposed);

            return &dL_dInput;
        }
        ///Print the layer used for debug purposes it has 2 different verbosity levels
        pub fn printLayer(self: *@This(), choice: u8) void {
            //MENU choice:
            // 0 -> full details layer
            // 1 -> shape schema
            if (choice == 0) {
                std.debug.print("\n ************************layer*********************", .{});
                std.debug.print("\n neurons:{}  inputs:{}", .{ self.n_neurons, self.n_inputs });
                std.debug.print("\n \n************input", .{});
                self.input.printMultidim();

                std.debug.print("\n \n************weights", .{});
                self.weights.printMultidim();
                std.debug.print("\n \n************bias", .{});
                std.debug.print("\n {any}", .{self.bias.data});
                std.debug.print("\n \n************output", .{});
                self.output.printMultidim();
                std.debug.print("\n \n************outputActivation", .{});
                self.outputActivation.printMultidim();
                std.debug.print("\n \n************w_gradients", .{});
                self.w_gradients.printMultidim();
                std.debug.print("\n \n************b_gradients", .{});
                std.debug.print("\n {any}", .{self.b_gradients.data});
            }
            if (choice == 1) {
                std.debug.print("\n ************************layer*********************", .{});
                std.debug.print("\n   input       weight   bias  output", .{});
                std.debug.print("\n [{} x {}] * [{} x {}] + {} = [{} x {}] ", .{
                    self.input.shape[0],
                    self.input.shape[1],
                    self.weights.shape[0],
                    self.weights.shape[1],
                    self.bias.shape[0],
                    self.output.shape[0],
                    self.output.shape[1],
                });
                std.debug.print("\n ", .{});
            }
        }
    };
}
