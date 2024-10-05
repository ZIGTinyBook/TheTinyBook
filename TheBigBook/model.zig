const std = @import("std");
const tensor = @import("tensor.zig");
const layer = @import("layers.zig");
const Loss = @import("lossFunction.zig");
const TensMath = @import("tensor_math.zig");

pub fn Model(comptime T: type, allocator: *const std.mem.Allocator) type {
    return struct {
        layers: []layer.DenseLayer(T, allocator) = undefined,
        allocator: *const std.mem.Allocator,
        input_tensor: tensor.Tensor(T),

        pub fn init(self: *@This()) !void {
            self.layers = try self.allocator.alloc(layer.DenseLayer(T, allocator), 0);
            self.input_tensor = undefined;
        }

        pub fn deinit(self: *@This()) void {
            for (self.layers) |*dense_layer| {
                dense_layer.deinit();
            }
            self.allocator.free(self.layers);
        }

        pub fn addLayer(self: *@This(), new_layer: *layer.DenseLayer(T, allocator)) !void {
            self.layers = try self.allocator.realloc(self.layers, self.layers.len + 1);
            self.layers[self.layers.len - 1] = new_layer.*;
        }

        pub fn forward(self: *@This(), input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            var output = input.*;
            self.input_tensor = try input.copy();
            for (self.layers, 0..) |*dense_layer, i| {
                std.debug.print("\n----------------------------------------output layer {}", .{i});
                output = try dense_layer.forward(&output);
                output.info();
            }
            return output;
        }

        pub fn backward(self: *@This(), gradient: *tensor.Tensor(T)) !*tensor.Tensor(T) {

            //grad is always equal to dot(grad, weights)
            var grad = gradient;
            var counter = self.layers.len - 1;
            var current_layer_input: *tensor.Tensor(T) = undefined;
            while (counter >= 0) : (counter -= 1) {
                //getting precedent layer output, aka current layer input
                current_layer_input = self.get_current_layer_input(counter);
                std.debug.print("\n--------------------------------------backwarding layer {}", .{counter});
                grad = try self.layers[counter].backward(grad, current_layer_input);
            }

            return grad;
        }

        pub fn train(self: *@This(), input: *tensor.Tensor(T), targets: *tensor.Tensor(T), epochs: u32) !void {
            var LossMeanRecord: []f32 = try allocator.alloc(f32, epochs);

            for (0..epochs) |i| {
                std.debug.print("\n\n----------------------epoch:{}", .{i});

                //forwarding
                std.debug.print("\n-------------------------------forwarding", .{});
                var predictions = try self.forward(input);

                //compute loss
                std.debug.print("\n-------------------------------computing loss", .{});
                var loser = Loss.LossFunction(Loss.MSELoss){};
                var loss = try loser.computeLoss(T, &predictions, targets);
                loss.info();

                //compute accuracy
                LossMeanRecord[i] = TensMath.mean(T, &loss);
                std.debug.print("\n     loss:{}", .{LossMeanRecord[i]});

                //compute gradient of the loss
                std.debug.print("\n-------------------------------computing loss gradient", .{});
                var grad: tensor.Tensor(T) = try loser.computeGradient(T, &predictions, targets);
                grad.info();

                //backwarding
                std.debug.print("\n-------------------------------backwarding", .{});
                _ = try self.backward(&grad);

                //optimizing
                std.debug.print("\n-------------------------------backwarding", .{});
            }
        }

        fn get_current_layer_input(self: *@This(), layer_number: usize) *tensor.Tensor(T) {
            if (layer_number == 0) return &self.input_tensor;
            return &self.layers[layer_number - 1].outputActivation;
        }
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var model = Model(f64, &allocator){
        .layers = undefined,
        .allocator = &allocator,
        .loss = undefined,
        .input_tensor = undefined,
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
