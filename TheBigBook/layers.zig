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
        weights: tensor.Tensor(T),
        bias: tensor.Tensor(T),
        output: tensor.Tensor(T), // output = dot(input, weight) + bias
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

            var weight_shape: [2]usize = [_]usize{ n_neurons, n_inputs };
            var bias_shape: [1]usize = [_]usize{n_neurons};
            self.allocator = alloc;

            //std.debug.print("Generating random weights...\n", .{});
            const weight_matrix = try randn(T, n_inputs, n_neurons, rng);
            const bias_matrix = try randn(T, 1, n_neurons, rng);

            //std.debug.print("Initializing weights and bias...\n", .{});
            //initializing gradients-----------------------------------------------------
            self.w_gradients = try tensor.Tensor(T).init(alloc);
            try self.w_gradients.fill(try zeros(T, n_inputs, n_neurons), &weight_shape);
            self.b_gradients = try tensor.Tensor(T).init(alloc);
            try self.b_gradients.fill(try zeros(T, 1, n_neurons), &bias_shape);

            //initializing weights and biases--------------------------------------------
            self.weights = try tensor.Tensor(T).fromArray(alloc, weight_matrix, &weight_shape);
            self.weights = try self.weights.transpose2D();
            self.bias = try tensor.Tensor(T).fromArray(alloc, bias_matrix, &bias_shape);

            //initializing number of neurons and inputs----------------------------------
            self.n_inputs = n_inputs;
            self.n_neurons = n_neurons;

            //this showld be the correct way to implement it ...
            //self.activation = ActivLib.ActivationFunction(ActivLib.ReLU){};
            //It doesn't work and I'm stuck... please forgive me for what Imma do...
            self.activation = activationFunction;
            //just see sep 7 of forward()

            //std.debug.print("Weight shape: {d} x {d}\n", .{ weight_shape[0], weight_shape[1] });
            //std.debug.print("Bias shape: {d} x {d}\n", .{ 1, bias_shape[0] });

            //std.debug.print("shapes are {} x {} and {} x {}\n", .{ self.weights.shape[0], self.weights.shape[1], 1, self.bias.shape[0] });

            //std.debug.print("Weights and bias initialized.\n", .{});
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
            // std.debug.print("Forward pass: input tensor shape = {} x {}\n", .{ input.shape[0], input.shape[1] });
            // std.debug.print("shapes before forward pass are {} x {} and {} x {}\n", .{ self.weights.shape[0], self.weights.shape[1], 1, self.bias.shape[0] });
            std.debug.print("\n >>>>>>>>>>> input: ", .{});
            input.info();

            std.debug.print("\n >>>>>>>>>>> self.weights: ", .{});
            self.weights.info();

            // 4. Check if self.output is already allocated, deallocate if necessary
            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            // 1. Perform multiplication between inputs and weights (dot product)
            self.output = try TensMath.compute_dot_product(T, input, &self.weights);
            //defer dot_product.deinit(); // Defer per liberare il tensor alla fine

            std.debug.print("\n >>>>>>>>>>> output pre-bias: ", .{});
            self.weights.info();

            // 2. Print debug information for dot_product and bias
            std.debug.print("\n >>>>>>>>>>> self.bias: ", .{});
            self.bias.info();

            // 3. Add bias to the dot product
            try TensMath.add_bias(T, &self.output, &self.bias);

            std.debug.print("\n >>>>>>>>>>> output post-bias: ", .{});
            self.weights.info();

            // 5. Allocate memory for self.output with the same shape as dot_product
            //self.output = try dot_product.copy();

            //copy the output in to outputActivation so to be modified in the activation function
            self.outputActivation = try self.output.copy();

            // 7. Apply activation function
            // I was gettig crazy with this.activation initialization since ActivLib.ActivationFunction( something ) is
            //dynamic and we are trying to do everything at comtime, no excuses

            if (std.mem.eql(u8, self.activation, "ReLU")) {
                var activation = ActivLib.ActivationFunction(ActivLib.ReLU){};
                try activation.forward(T, &self.outputActivation);
            } else if (std.mem.eql(u8, self.activation, "Softmax")) {
                var activation = ActivLib.ActivationFunction(ActivLib.Softmax){};
                try activation.forward(T, &self.outputActivation);
            }

            // 8. Print information on the final output
            //self.output.info(); // print output using info()
            //self.outputActivation.info();

            //PAY ATTENTION: here we return the outputActivation, so the altrady activated output
            return self.outputActivation;
        }

        pub fn backward(self: *@This(), prec_layer_output: *tensor.Tensor(T), dL_dOutput: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            //---- Key Steps: -----
            // 1. Output Gradient: dL/dOutput, apply activation derivative
            std.debug.print("\n >>>>>>>>>>> dL_dOutput before applying activation derivative", .{});
            dL_dOutput.info();

            //--- Apply the derivative of the activation function
            if (std.mem.eql(u8, self.activation, "ReLU")) {
                var activ_grad = ActivLib.ActivationFunction(ActivLib.ReLU){};
                try activ_grad.derivate(T, dL_dOutput);
            } else if (std.mem.eql(u8, self.activation, "Softmax")) {
                var activ_grad = ActivLib.ActivationFunction(ActivLib.Softmax){};
                try activ_grad.derivate(T, dL_dOutput);
            }
            std.debug.print("\n >>>>>>>>>>> dL_dOutput after applying activation derivative", .{});
            dL_dOutput.info();

            std.debug.print("\n >>>>>>>>>>> prec_layer_output ", .{});
            prec_layer_output.info();

            // 2. Compute weight gradients (w_gradients)
            var prec_layer_output_transposed = try prec_layer_output.transpose2D();
            defer prec_layer_output_transposed.deinit();

            self.w_gradients.deinit();
            self.w_gradients = try TensMath.dot_product_tensor(Architectures.CPU, T, T, &prec_layer_output_transposed, dL_dOutput);
            std.debug.print("\n >>>>>>>>>>> w_gradients", .{});
            self.w_gradients.info();

            // 3. Compute bias gradients (b_gradients)
            // Equivalent of np.sum(dL_dOutput, axis=0, keepdims=True)
            self.b_gradients.deinit();
            self.b_gradients = try self.sumAlongAxis(dL_dOutput);
            std.debug.print("\n >>>>>>>>>>> b_gradients", .{});
            self.b_gradients.info();

            // 4. Compute input gradients: dL/dInput = dot(dL_dOutput, weights.T)
            var dL_dOutput_transposed = try dL_dOutput.transpose2D();
            defer dL_dOutput_transposed.deinit();

            var weights_transposed = try self.weights.transpose2D();
            defer weights_transposed.deinit();

            var dL_dInput = try TensMath.dot_product_tensor(Architectures.CPU, T, T, &dL_dOutput_transposed, &weights_transposed);
            return &dL_dInput;
        }

        fn sumAlongAxis(self: *@This(), ten: *tensor.Tensor(T)) !tensor.Tensor(T) {
            // Ensure axis is valid

            // Transpose the tensor and store it in a new variable
            var transposed_tensor = try ten.transpose2D(); // Use `try` to handle potential error

            const rows = transposed_tensor.shape[0];
            const cols = transposed_tensor.shape[1];

            // Create the shape for the resulting tensor (collapsed to one row)
            var result_shape: [2]usize = [_]usize{ 1, cols };

            // Allocating dynamic memory for the result data instead of static array
            var initial_result_data = try self.allocator.alloc(T, cols);

            // Create a new tensor for storing the summed result using `fromArray`
            var result = try tensor.Tensor(T).fromArray(self.allocator, initial_result_data, &result_shape);

            // Sum along the rows (axis 0), keeping the columns intact
            for (0..cols) |col| {
                var sum: T = 0;
                for (0..rows) |row| {
                    sum += transposed_tensor.data[row * cols + col]; // Access the transposed data
                }
                result.data[col] = sum; // Store the sum of the current column
            }

            transposed_tensor.deinit(); // Deallocate the transposed tensor
            initial_result_data = undefined; // Deallocate the initial result data

            return result;
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
