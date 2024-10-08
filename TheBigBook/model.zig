const std = @import("std");
const tensor = @import("tensor.zig");
const layer = @import("layers.zig");
const Loss = @import("lossFunction.zig");
const TensMath = @import("tensor_math.zig");
const Optim = @import("optim.zig");
const loader = @import("dataLoader.zig").DataLoader;

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
                // std.debug.print("\n >>>>>>>>>>> output post-activation: ", .{});
                // output.info();
            }
            return output;
        }

        pub fn backward(self: *@This(), gradient: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            //grad is always equal to dot(grad, weights)
            var grad = gradient;
            var grad_duplicate = try grad.copy();
            var counter = (self.layers.len - 1);
            while (counter >= 0) : (counter -= 1) {
                std.debug.print("\n--------------------------------------backwarding layer {}", .{counter});
                grad = try self.layers[counter].backward(&grad_duplicate);
                grad_duplicate = try grad.copy();
                if (counter == 0) break;
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
                std.debug.print("\n-------------------------------Optimizer Step", .{});
                var optimizer = Optim.Optimizer(f64, Optim.optimizer_SGD, 0.01, allocator){ // Here we pass the actual instance of the optimizer
                };
                try optimizer.step(self);
            }

            std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
        }

        pub fn TrainDataLoader(self: *@This(), load: *loader(f64, f64, 100), ephocs: u32) !void {
            var LossMeanRecord: []f32 = try allocator.alloc(f32, ephocs);
            var shapeXArr = [_]usize{ 100, 5 };
            var shapeYArr = [_]usize{100};
            var shapeX: []usize = &shapeXArr;
            var shapeY: []usize = &shapeYArr;

            for (0..ephocs) |i| {
                std.debug.print("\n\n----------------------epoch:{}", .{i});
                for (0..10) |step| {
                    _ = load.xNextBatch(100);
                    _ = load.yNextBatch(100);
                    try load.toTensor(allocator, &shapeX, &shapeY);

                    //forwarding
                    std.debug.print("\n-------------------------------forwarding", .{});
                    var predictions = try self.forward(&load.xTensor);
                    var shape: [1]usize = [_]usize{100};
                    try predictions.reshape(&shape);
                    var pred_cpy = try predictions.copy();
                    try pred_cpy.reshape(&shape);
                    pred_cpy.shape.len = 1;

                    //compute loss
                    std.debug.print("\n-------------------------------computing loss", .{});
                    var loser = Loss.LossFunction(Loss.MSELoss){};
                    var loss = try loser.computeLoss(T, &pred_cpy, &load.yTensor);
                    loss.info();

                    //compute accuracy
                    LossMeanRecord[i] = TensMath.mean(T, &loss);
                    std.debug.print("\n     loss:{}", .{LossMeanRecord[i]});
                    //compute gradient of the loss
                    std.debug.print("\n-------------------------------computing loss gradient", .{});
                    var grad: tensor.Tensor(T) = try loser.computeGradient(T, &pred_cpy, &load.yTensor);
                    grad.info();
                    var shape2: [2]usize = [_]usize{ 100, 1 };
                    grad.shape.len = 2;
                    try grad.reshape(&shape2);

                    //backwarding
                    std.debug.print("\n-------------------------------backwarding", .{});
                    _ = try self.backward(&grad);

                    //optimizing
                    std.debug.print("\n-------------------------------Optimizer Step", .{});
                    var optimizer = Optim.Optimizer(f64, Optim.optimizer_SGD, 0.05, allocator){ // Here we pass the actual instance of the optimizer
                    };
                    try optimizer.step(self);
                    std.debug.print("Batch Bumber {}", .{step});
                }

                load.reset();
                std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
            }
        }
    };
}
