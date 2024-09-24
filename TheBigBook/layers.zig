const std = @import("std");
const tensor = @import("tensor.zig");
const TensMath = @import("./tensor_math.zig");
const Architectures = @import("./architectures.zig").Architectures;
const TensorError = @import("./tensor_math.zig").TensorError;
const ArchitectureError = @import("./tensor_math.zig").ArchitectureError;

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
        output: tensor.Tensor(T),
        n_inputs: usize,
        n_neurons: usize,
        weightShape: []usize,
        biasShape: []usize,
        allocator: *const std.mem.Allocator,

        pub fn init(self: *@This(), n_inputs: usize, n_neurons: usize, rng: *std.Random.Xoshiro256) !void {
            std.debug.print("Init DenseLayer: n_inputs = {}, n_neurons = {}, Type = {}\n", .{ n_inputs, n_neurons, @TypeOf(T) });

            var weight_shape: [2]usize = [_]usize{ n_inputs, n_neurons };
            var bias_shape: [1]usize = [_]usize{n_neurons};
            self.weightShape = &weight_shape;
            self.biasShape = &bias_shape;
            self.allocator = alloc;

            std.debug.print("Generating random weights...\n", .{});
            const weight_matrix = try randn(T, n_inputs, n_neurons, rng);
            const bias_matrix = try zeros(T, 1, n_neurons);

            std.debug.print("Initializing weights and bias...\n", .{});

            self.weights = try tensor.Tensor(T).fromArray(alloc, weight_matrix, &weight_shape);
            self.bias = try tensor.Tensor(T).fromArray(alloc, bias_matrix, &bias_shape);
            self.n_inputs = n_inputs;
            self.n_neurons = n_neurons;

            std.debug.print("Weight shape: {d} x {d}\n", .{ weight_shape[0], weight_shape[1] });
            std.debug.print("Bias shape: {d} x {d}\n", .{ 1, bias_shape[0] });

            std.debug.print("shapes are {} x {} and {} x {}\n", .{ self.weights.shape[0], self.weights.shape[1], 1, self.bias.shape[0] });

            std.debug.print("Weights and bias initialized.\n", .{});
        }
        pub fn forward(self: *@This(), input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            std.debug.print("Forward pass: input tensor shape = {} x {}\n", .{ input.shape[0], input.shape[1] });
            std.debug.print("shapes before forward pass are {} x {} and {} x {}\n", .{ self.weights.shape[0], self.weights.shape[1], 1, self.bias.shape[0] });

            var dot_product: tensor.Tensor(T) = try TensMath.compute_dot_product(T, input, &self.weights);
            defer dot_product.deinit();

            dot_product.info();
            self.bias.info();

            self.output = try TensMath.sum_tensors(Architectures.CPU, T, T, &dot_product, &self.bias);

            std.debug.print("Output tensor: {any}\n", .{self.output});

            return self.output;
        }
        pub fn deinit(self: *@This()) void {
            std.debug.print("Deallocating DenseLayer resources...\n", .{});

            // Dealloca i tensori di weights, bias e output se allocati
            if (self.weights.data.len > 0) {
                self.weights.deinit();
            }

            if (self.bias.data.len > 0) {
                self.bias.deinit();
            }

            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            std.debug.print("DenseLayer resources deallocated.\n", .{});
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
        .n_inputs = 0,
        .n_neurons = 0,
        .weightShape = undefined,
        .biasShape = undefined,
        .allocator = undefined,
    };

    try dense_layer.init(n_inputs, n_neurons, &rng);

    std.debug.print("Pesi e bias inizializzati\n", .{});

    //std.debug.print("shapes after init main are {} x {} and {} x {}\n", .{ dense_layer.weights.shape[0], dense_layer.weights.shape[1], 1, dense_layer.bias.shape[0] });

    var inputArray: [2][4]f64 = [_][4]f64{
        [_]f64{ 1.0, 2.0, 3.0, 1 },
        [_]f64{ 4.0, 5.0, 6.0, 2 },
    };
    var shape: [2]usize = [_]usize{ 2, 4 };

    var input_tensor = try tensor.Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer input_tensor.deinit();

    _ = try dense_layer.forward(&input_tensor);

    //

    dense_layer.output.deinit();
}
