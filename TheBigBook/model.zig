const std = @import("std");
const tensor = @import("tensor.zig");
const layer = @import("layers.zig");

pub fn Model(comptime T: type, allocator: *const std.mem.Allocator) type {
    return struct {
        layers: []layer.DenseLayer(T, allocator) = undefined,
        allocator: *const std.mem.Allocator,

        pub fn init(self: *@This()) !void {
            self.layers = try self.allocator.alloc(layer.DenseLayer(T, allocator), 0);
        }

        pub fn addLayer(self: *@This(), new_layer: *layer.DenseLayer(T, allocator)) !void {
            self.layers = try self.allocator.realloc(self.layers, self.layers.len + 1);
            self.layers[self.layers.len - 1] = new_layer.*;
        }

        pub fn forward(self: *@This(), input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            var output = input.*;
            for (self.layers) |*dense_layer| {
                output = try dense_layer.forward(&output);
            }
            return output;
        }

        pub fn deinit(self: *@This()) void {
            for (self.layers) |*dense_layer| {
                dense_layer.deinit();
            }
            self.allocator.free(self.layers);
        }
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
    };
    try model.init();

    var rng = std.rand.Random.Xoshiro256.init(12345);

    var layer1 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .weightShape = undefined,
        .biasShape = undefined,
        .allocator = undefined,
    };
    try layer1.init(3, 2, &rng);
    try model.addLayer(&layer1);

    var layer2 = layer.DenseLayer(f64, &allocator){
        .weights = undefined,
        .bias = undefined,
        .output = undefined,
        .n_inputs = 0,
        .n_neurons = 0,
        .weightShape = undefined,
        .biasShape = undefined,
        .allocator = undefined,
    };
    try layer2.init(2, 3, &rng);
    try model.addLayer(&layer2);

    // Creazione di un input tensor
    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var input_tensor = try tensor.Tensor(f64).init(&allocator);
    _ = try input_tensor.fill(&inputArray, shape[0..]);

    const output = try model.forward(&input_tensor);
    std.debug.print("Output finale: {any}\n", .{output});

    //output.deinit();
    model.deinit();
    input_tensor.deinit();
}
