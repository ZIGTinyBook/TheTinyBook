const std = @import("std");
const tensor = @import("tensor.zig");
const TensMath = @import("./tensor_math.zig");
const Architectures = @import("./architectures.zig").Architectures;
const TensorError = @import("./tensor_math.zig").TensorError;
const ArchitectureError = @import("./tensor_math.zig").ArchitectureError;
const ActivLib = @import("./activation_function.zig");

pub fn randn(comptime T: type, n_inputs: usize, n_neurons: usize, rng: *std.Random.Xoshiro256) ![][]T {
    const matrix = try std.heap.page_allocator.alloc([]T, n_inputs);
    for (matrix) |*row| {
        row.* = try std.heap.page_allocator.alloc(T, n_neurons);
        for (row.*) |*value| {
            value.* = rng.random().floatNorm(T);
        }
    }
    return matrix;
}

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
        //activation function-------------
        activation: []const u8,
        //gradients-----------------------
        w_gradients: tensor.Tensor(T),
        b_gradients: tensor.Tensor(T),
        //utils---------------------------
        allocator: *const std.mem.Allocator,

        pub fn init(self: *@This(), n_inputs: usize, n_neurons: usize, rng: *std.Random.Xoshiro256, activationFunction: []const u8) !void {
            //std.debug.print("Init DenseLayer: n_inputs = {}, n_neurons = {}, Type = {}\n", .{ n_inputs, n_neurons, @TypeOf(T) });

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

            //initializing activation----------------------------------------------------
            self.activation = activationFunction;
            //just see sep 7 of forward()

            //initializing gradients to all zeros----------------------------------------
            self.w_gradients = try tensor.Tensor(T).fromShape(self.allocator, &weight_shape);
            self.b_gradients = try tensor.Tensor(T).fromShape(self.allocator, &bias_shape);
        }

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

            std.debug.print("DenseLayer resources deallocated.\n", .{});
        }

        pub fn forward(self: *@This(), input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            // std.debug.print("\n >>>>>>>>>>> input: ", .{});
            // input.info();
            self.input = try input.copy();

            // 1. Check if self.output is already allocated, deallocate if necessary
            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            // 2. Perform multiplication between inputs and weights (dot product)
            self.output = try TensMath.compute_dot_product(T, &self.input, &self.weights);

            // std.debug.print("\n >>>>>>>>>>> output pre-bias: ", .{});
            // self.output.info();

            // // 3. Print debug information for dot_product and bias
            // std.debug.print("\n >>>>>>>>>>> self.bias: ", .{});
            // self.bias.info();

            // 4. Add bias to the dot product
            try TensMath.add_bias(T, &self.output, &self.bias);

            // std.debug.print("\n >>>>>>>>>>> output post-bias: ", .{});
            // self.output.info();

            //5. copy the output in to outputActivation so to be modified in the activation function
            self.outputActivation = try self.output.copy();

            // 6. Apply activation function
            // I was gettig crazy with this.activation initialization since ActivLib.ActivationFunction( something ) is
            //dynamic and we are trying to do everything at comtime, no excuses

            if (std.mem.eql(u8, self.activation, "ReLU")) {
                var activation = ActivLib.ActivationFunction(ActivLib.ReLU){};
                try activation.forward(T, &self.outputActivation);
            } else if (std.mem.eql(u8, self.activation, "Softmax")) {
                var activation = ActivLib.ActivationFunction(ActivLib.Softmax){};
                try activation.forward(T, &self.outputActivation);
            }

            self.printLayer();

            //PAY ATTENTION: here we return the outputActivation, so the altrady activated output
            return self.outputActivation;
        }

        pub fn backward(self: *@This(), dValues: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            //---- Key Steps: -----
            // 1. Output Gradient: dL/dOutput, apply activation derivative
            std.debug.print("\n >>>>>>>>>>> dL_dOutput before applying activation derivative", .{});
            dValues.info();

            //--- Apply the derivative of the activation function
            if (std.mem.eql(u8, self.activation, "ReLU")) {
                var activ_grad = ActivLib.ActivationFunction(ActivLib.ReLU){};
                try activ_grad.derivate(T, dValues);
            } else if (std.mem.eql(u8, self.activation, "Softmax")) {
                var activ_grad = ActivLib.ActivationFunction(ActivLib.Softmax){};
                try activ_grad.derivate(T, dValues);
            }
            std.debug.print("\n >>>>>>>>>>> dL_dOutput after applying activation derivative", .{});
            dValues.info();

            // std.debug.print("\n >>>>>>>>>>> prec_layer_output aka: current_layer_input ", .{});
            // self.input.info();

            // 2. Compute weight gradients (w_gradients)
            var input_transposed = try self.input.transpose2D();
            defer input_transposed.deinit();

            self.w_gradients.deinit();
            self.w_gradients = try TensMath.dot_product_tensor(Architectures.CPU, T, T, &input_transposed, dValues);
            // std.debug.print("\n >>>>>>>>>>> w_gradients", .{});
            // self.w_gradients.info();

            // 3. Compute bias gradients (b_gradients)
            // Equivalent of np.sum(dL_dOutput, axis=0, keepdims=True)
            var sum: T = 0;
            std.debug.print("\n self.n_neurons: {}  self.n_input:{}", .{ self.n_neurons, self.n_inputs });

            for (0..dValues.shape[1]) |neuron| {
                //scanning all the inputs
                sum = 0;
                for (0..dValues.shape[0]) |input| {
                    sum += dValues.data[input * self.n_neurons + neuron];
                }
                self.b_gradients.data[neuron] = sum;
            }
            // std.debug.print("\n >>>>>>>>>>> b_gradients", .{});
            // self.b_gradients.info();

            // 4. Compute input gradients: dL/dInput = dot(dL_dOutput, weights.T)
            var dValues_transposed = try dValues.transpose2D();
            defer dValues_transposed.deinit();
            // std.debug.print("\n >>>>>>>>>>> dL_dOutput_transposed", .{});
            // dValues_transposed.info();

            var weights_transposed = try self.weights.transpose2D();
            defer weights_transposed.deinit();

            var dL_dInput = try TensMath.dot_product_tensor(Architectures.CPU, T, T, dValues, &weights_transposed);
            // std.debug.print("\n >>>>>>>>>>> dL_dInput", .{});
            // dL_dInput.info();

            self.printLayer();

            return &dL_dInput;
        }

        pub fn printLayer(self: *@This()) void {
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
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var rng = std.Random.Xoshiro256.init(12345);

    const n_inputs: usize = 4;
    const n_neurons: usize = 2;

    var dense_layer = DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .outputActivation = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .weightShape = undefined,
        .biasShape = undefined,
        .w_gradients = undefined,
        .b_gradients = undefined,
        .allocator = undefined,
        .activation = undefined,
    };

    try dense_layer.init(n_inputs, n_neurons, &rng, "ReLU");

    std.debug.print("Weights and bias initialized\n", .{});

    //std.debug.print("shapes after init main are {} x {} and {} x {}\n", .{ dense_layer.weights.shape[0], dense_layer.weights.shape[1], 1, dense_layer.bias.shape[0] });

    var inputArray: [2][4]f64 = [_][4]f64{
        [_]f64{ 1.0, 2.0, 3.0, 1 },
        [_]f64{ 4.0, 5.0, 6.0, 2 },
    };
    var shape: [2]usize = [_]usize{ 2, 4 };

    var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    _ = try dense_layer.forward(&input_tensor);
    dense_layer.output.info();

    dense_layer.output.deinit();
}
