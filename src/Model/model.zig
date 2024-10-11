const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layers");
const Loss = @import("loss");
const LossType = @import("loss").LossType;
const TensMath = @import("tensor_m");
const Optim = @import("optim");
const loader = @import("dataloader").DataLoader;
const NormalizType = @import("dataprocessor").NormalizationType;
const DataProc = @import("dataprocessor");

pub fn Model(comptime T: type, comptime XType: type, comptime YType: type, comptime allocator: *const std.mem.Allocator, lr: f64) type {
    return struct {
        layers: []layer.Layer(T, allocator) = undefined,
        allocator: *const std.mem.Allocator,
        input_tensor: tensor.Tensor(T),

        pub fn init(self: *@This()) !void {
            self.layers = try self.allocator.alloc(layer.Layer(T, allocator), 0);
            self.input_tensor = undefined;
        }

        pub fn deinit(self: *@This()) void {
            for (self.layers) |*layer_| {
                layer_.deinit();
            }
            self.allocator.free(self.layers);
        }

        pub fn addLayer(self: *@This(), new_layer: *layer.Layer(T, allocator)) !void {
            self.layers = try self.allocator.realloc(self.layers, self.layers.len + 1);
            self.layers[self.layers.len - 1] = new_layer.*;
        }

        pub fn forward(self: *@This(), input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            var output = try input.copy();
            self.input_tensor = try input.copy();
            for (self.layers, 0..) |*layer_, i| {
                std.debug.print("\n----------------------------------------output layer {}", .{i});
                try DataProc.normalize(T, &output, NormalizType.UnityBasedNormalizartion);

                output = try layer_.forward(&output);
                std.debug.print("\n >>>>>>>>>>> output post-activation: ", .{});
                output.info();
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
        //TODO maybe wrap X and Y in dataloader
        pub fn train(self: *@This(), input: *tensor.Tensor(T), targets: *tensor.Tensor(T), epochs: u32) !void {
            var LossMeanRecord: []f32 = try allocator.alloc(f32, epochs);

            for (0..epochs) |i| {
                std.debug.print("\n\n----------------------epoch:{}", .{i});

                //forwarding
                std.debug.print("\n-------------------------------forwarding", .{});
                var predictions = try self.forward(input);

                //compute loss
                std.debug.print("\n-------------------------------computing loss", .{});
                const loser = Loss.LossFunction(LossType.MSE){};
                var loss = try loser.computeLoss(T, &predictions, targets);

                //compute accuracy
                LossMeanRecord[i] = TensMath.mean(T, &loss);
                std.debug.print("\n     loss:{}", .{LossMeanRecord[i]});

                //compute gradient of the loss
                std.debug.print("\n-------------------------------computing loss gradient", .{});
                var grad: tensor.Tensor(T) = try loser.computeGradient(T, &predictions, targets);

                //backwarding
                std.debug.print("\n-------------------------------backwarding", .{});
                _ = try self.backward(&grad);

                //optimizing
                std.debug.print("\n-------------------------------Optimizer Step", .{});
                var optimizer = Optim.Optimizer(T, XType, YType, Optim.optimizer_SGD, 0.05, allocator){ // Here we pass the actual instance of the optimizer
                };
                try optimizer.step(self);
            }

            std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
        }

        pub fn TrainDataLoader(self: *@This(), comptime batchSize: i16, features: usize, load: *loader(T, XType, YType, batchSize), ephocs: u32, classification: bool) !void {
            var LossMeanRecord: []f32 = try allocator.alloc(f32, ephocs);
            var shapeXArr = [_]usize{ batchSize, features };
            var shapeYArr = [_]usize{batchSize};
            var shapeX: []usize = &shapeXArr;
            var shapeY: []usize = &shapeYArr;
            var steps: u16 = 0;
            var max: T = 0;

            const len: u16 = @as(u16, @intCast(load.X.len));
            steps = @divFloor(len, batchSize);
            std.debug.print("\n\n----------------------len:{}", .{len});
            if (len % batchSize != 0) {
                steps += 1;
            }

            for (0..ephocs) |i| {
                std.debug.print("\n\n----------------------epoch:{}", .{i});
                for (0..steps) |step| {
                    _ = load.xNextBatch(batchSize);
                    _ = load.yNextBatch(batchSize);
                    try load.toTensor(allocator, &shapeX, &shapeY);

                    //forwarding
                    std.debug.print("\n-------------------------------forwarding", .{});
                    try DataProc.normalize(T, &load.xTensor, NormalizType.UnityBasedNormalizartion);
                    var predictions = try self.forward(&load.xTensor);
                    var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
                    try predictions.reshape(&shape);

                    //compute loss
                    std.debug.print("\n-------------------------------computing loss", .{});
                    const loser = Loss.LossFunction(LossType.CCE){};
                    if (classification) {
                        //Take the most likely prediction so the one with the highest value from tensor AN PUT IT AS THE PREDICTION
                        max = predictions.data[0];
                        var maxIndex: usize = 0;
                        for (0..predictions.size) |J| {
                            if (predictions.data[J] > max) {
                                max = predictions.data[J];
                                maxIndex = i;
                            }
                        }
                        var maxArray = [1]T{max};
                        var shape_: [1]usize = [_]usize{1};
                        try predictions.fill(&maxArray, &shape_);
                    }
                    var loss = try loser.computeLoss(T, &predictions, &load.yTensor);

                    //compute accuracy
                    LossMeanRecord[i] = TensMath.mean(T, &loss);
                    std.debug.print("\n     loss:{}", .{LossMeanRecord[i]});
                    //compute gradient of the loss
                    std.debug.print("\n-------------------------------computing loss gradient", .{});
                    var grad: tensor.Tensor(T) = try loser.computeGradient(T, &predictions, &load.yTensor);
                    var shape_: [1]usize = [_]usize{1};
                    try grad.reshape(&shape_);

                    //backwarding
                    std.debug.print("\n-------------------------------backwarding", .{});
                    _ = try self.backward(&grad);

                    //optimizing
                    std.debug.print("\n-------------------------------Optimizer Step", .{});
                    var optimizer = Optim.Optimizer(T, XType, YType, Optim.optimizer_SGD, lr, allocator){ // Here we pass the actual instance of the optimizer
                    };
                    try optimizer.step(self);
                    std.debug.print("Batch Bumber {}", .{step});
                }

                load.reset();
                std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
                std.debug.print("steps:{}", .{steps});
            }
        }
    };
}
