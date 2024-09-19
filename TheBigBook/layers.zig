const std = @import("std");
const tensor = @import("tensor.zig");
const TensMath = @import("./tensor_math.zig");
const Architectures = @import("./architectures.zig").Architectures;
const TensorError = @import("./tensor_math.zig").TensorError;
const ArchitectureError = @import("./tensor_math.zig").ArchitectureError;

pub fn randn(comptime T: type, n_inputs: usize, n_neurons: usize, rng: *std.rand.Random.Xoshiro256) ![][]T {
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

pub fn DenseLayer(comptime T: type) type {
    return struct {
        weights: *tensor.Tensor(T),
        bias: *tensor.Tensor(T),
        output: *tensor.Tensor(T),
        n_inputs: usize,
        n_neurons: usize,
        weightShape: []usize,
        biasShape: []usize,

        pub fn init(self: *@This(), n_inputs: usize, n_neurons: usize, rng: *std.rand.Random.Xoshiro256) !void {
            std.debug.print("Init DenseLayer: n_inputs = {}, n_neurons = {}, Type = {}\n", .{ n_inputs, n_neurons, @TypeOf(T) });

            var weight_shape: [2]usize = [_]usize{ n_inputs, n_neurons };
            var bias_shape: [2]usize = [_]usize{ 1, n_neurons };
            self.weightShape = &weight_shape;
            self.biasShape = &bias_shape;

            std.debug.print("Weight shape: {d} x {d}\n", .{ weight_shape[0], weight_shape[1] });
            std.debug.print("Bias shape: {d} x {d}\n", .{ bias_shape[0], bias_shape[1] });

            // Allocazione e inizializzazione di weights
            self.weights = try std.heap.page_allocator.create(tensor.Tensor(T));
            self.weights.* = try tensor.Tensor(T).init(&std.heap.page_allocator, weight_shape[0..]);

            // Allocazione e inizializzazione di bias
            self.bias = try std.heap.page_allocator.create(tensor.Tensor(T));
            self.bias.* = try tensor.Tensor(T).init(&std.heap.page_allocator, bias_shape[0..]);

            std.debug.print("shapes are {} x {} and {} x {}\n", .{ self.weights.shape[0], self.weights.shape[1], self.bias.shape[0], self.bias.shape[1] });

            std.debug.print("Generating random weights...\n", .{});
            var weight_matrix = try randn(T, n_inputs, n_neurons, rng);
            var bias_matrix = try zeros(T, 1, n_neurons);

            std.debug.print("Initializing weights and bias...\n", .{});
            std.debug.print("shapes after are {} x {} and {} x {}\n", .{ self.weights.shape[0], self.weights.shape[1], self.bias.shape[0], self.bias.shape[1] });

            _ = try self.weights.fromArray(weight_matrix[0..], weight_shape[0..]);
            _ = try self.bias.fromArray(bias_matrix[0..], bias_shape[0..]);
            self.n_inputs = n_inputs;
            self.n_neurons = n_neurons;

            std.debug.print("Weights and bias initialized.\n", .{});
        }

        pub fn forward(self: *@This(), input: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            std.debug.print("Forward pass: input tensor shape = {} x {}\n", .{ input.shape[0], input.shape[1] });
            std.debug.print("shapes before forw are {} x {} and {} x {}\n", .{ self.weights.shape[0], self.weights.shape[1], self.bias.shape[0], self.bias.shape[1] });

            // Passiamo i puntatori ai tensori
            self.output = try TensMath.compute_dot_product(T, input, self.weights);
            try TensMath.sum_tensors(Architectures.CPU, T, T, self.output, self.bias, self.output);

            // Stampiamo l'output
            std.debug.print("Output tensor: {any}\n", .{self.output});
            return self.output;
        }
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var rng = std.rand.Random.Xoshiro256.init(12345);

    const n_inputs: usize = 2;
    const n_neurons: usize = 2;

    var dense_layer = DenseLayer(f64){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .weightShape = undefined,
        .biasShape = undefined,
    };

    try dense_layer.init(n_inputs, n_neurons, &rng);

    std.debug.print("Pesi e bias inizializzati\n", .{});

    std.debug.print("shapes after init main are {} x {} and {} x {}\n", .{ dense_layer.weights.shape[0], dense_layer.weights.shape[1], dense_layer.bias.shape[0], dense_layer.bias.shape[1] });

    var inputArray: [2][2]f64 = [_][2]f64{
        [_]f64{ 1.0, 2.0 },
        [_]f64{ 3.0, 4.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 2 };

    var input_tensor = try tensor.Tensor(f64).init(&allocator, shape[0..]);
    _ = try input_tensor.fromArray(&inputArray, shape[0..]);

    _ = try dense_layer.forward(&input_tensor);

    //

    dense_layer.output.deinit();
    input_tensor.deinit();
}
