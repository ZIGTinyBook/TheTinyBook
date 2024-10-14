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
            for (0..self.layers.len) |i| {
                std.debug.print("\n-------------------------------pre-norm layer {}", .{i});
                try DataProc.normalize(T, &output, NormalizType.UnityBasedNormalizartion);
                std.debug.print("\n-------------------------------post-norm layer {}", .{i});

                output = try self.layers[i].forward(&output);
                std.debug.print("\n-------------------------------output layer {}", .{i});
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
                std.debug.print("\n     gradient:", .{});
                grad.info();

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

        pub fn TrainDataLoader(self: *@This(), comptime batchSize: i16, features: usize, load: *loader(T, XType, YType, batchSize), epochs: u32) !void {
            var LossMeanRecord: []f32 = try allocator.alloc(f32, epochs);
            var AccuracyRecord: []f32 = try allocator.alloc(f32, epochs); // Array per l'accuratezza
            var shapeXArr = [_]usize{ batchSize, features };
            var shapeYArr = [_]usize{ batchSize, 10 }; // Ora la forma di y è batchSize x 10
            var shapeX: []usize = &shapeXArr;
            var shapeY: []usize = &shapeYArr;
            var steps: u16 = 0;

            const len: u16 = @as(u16, @intCast(load.X.len));
            steps = @divFloor(len, batchSize);
            std.debug.print("\n\n----------------------len:{}", .{len});
            if (len % batchSize != 0) {
                steps += 1;
            }

            for (0..epochs) |i| {
                std.debug.print("\n\n----------------------epoch:{}", .{i});
                var totalCorrect: u16 = 0; // Per il calcolo dell'accuratezza
                var totalSamples: u16 = 0;
                for (0..steps) |step| {
                    _ = load.xNextBatch(batchSize);
                    _ = load.yNextBatch(batchSize);

                    // Converti y in formato "one-hot encoded"

                    // Carica in tensori
                    try load.toTensor(allocator, &shapeX, &shapeY);

                    try convertToOneHot(batchSize, &load.yTensor);

                    // Forwarding
                    std.debug.print("\n-------------------------------forwarding", .{});
                    try DataProc.normalize(T, &load.xTensor, NormalizType.UnityBasedNormalizartion);
                    var predictions = try self.forward(&load.xTensor);
                    var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
                    try predictions.reshape(&shape);

                    // Compute loss
                    std.debug.print("\n-------------------------------computing loss", .{});
                    const loser = Loss.LossFunction(LossType.CCE){};
                    try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);
                    var loss = try loser.computeLoss(T, &predictions, &load.yTensor);

                    // Compute accuracy
                    const correctPredictions: u16 = try self.computeAccuracy(&predictions, &load.yTensor);
                    totalCorrect += correctPredictions;
                    totalSamples += batchSize;

                    LossMeanRecord[i] = TensMath.mean(T, &loss);
                    // Converti totalCorrect e totalSamples in f32 per evitare errori di tipo
                    AccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrect)) / @as(f32, @floatFromInt(totalSamples)) * 100.0;
                    std.debug.print("\n     loss:{} accuracy:{}%", .{ LossMeanRecord[i], AccuracyRecord[i] });

                    // Compute gradient of the loss
                    std.debug.print("\n-------------------------------computing loss gradient", .{});
                    var grad: tensor.Tensor(T) = try loser.computeGradient(T, &predictions, &load.yTensor);

                    // Backwarding
                    std.debug.print("\n-------------------------------backwarding", .{});
                    _ = try self.backward(&grad);

                    // Optimizing
                    std.debug.print("\n-------------------------------Optimizer Step", .{});
                    var optimizer = Optim.Optimizer(T, XType, YType, Optim.optimizer_SGD, lr, allocator){};
                    try optimizer.step(self);
                    std.debug.print("Batch Number {}", .{step});
                }

                load.reset();
                std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
                std.debug.print("steps:{}", .{steps});
            }
        }

        fn computeAccuracy(self: *@This(), predictions: *tensor.Tensor(T), targets: *tensor.Tensor(T)) !u16 {
            _ = self;
            var correct: u16 = 0;
            const rows = predictions.shape[0];
            const cols = predictions.shape[1];

            for (0..rows) |i| {
                var predictedLabel: usize = 0;
                var maxVal: T = predictions.data[i * cols];

                // Trova la classe con il valore massimo nelle predizioni
                for (1..cols) |j| {
                    const val = predictions.data[i * cols + j];
                    if (val > maxVal) {
                        maxVal = val;
                        predictedLabel = j;
                    }
                }

                // Trova l'etichetta vera
                var actualLabel: usize = 0;
                var maxTargetVal: T = targets.data[i * cols];
                for (1..cols) |j| {
                    const val = targets.data[i * cols + j];
                    if (val > maxTargetVal) {
                        maxTargetVal = val;
                        actualLabel = j;
                    }
                }

                // Se la predizione è corretta
                if (predictedLabel == actualLabel) {
                    correct += 1;
                }
            }

            return correct;
        }

        fn convertToOneHot(batchSize: i16, yBatch: *tensor.Tensor(T)) !void {
            // Numero di classi
            const numClasses = 10;

            // Crea una forma per il tensore one-hot: batchSize x numClasses
            var shapeYArr = [_]usize{ @intCast(batchSize), numClasses };
            const oneHotShape = &shapeYArr;

            // Crea un nuovo tensore per yBatch in formato one-hot
            var oneHotYBatch = try tensor.Tensor(T).fromShape(yBatch.allocator, oneHotShape);

            // Per ogni esempio nel batch
            for (0..@intCast(batchSize)) |i| {
                // Ottieni l'etichetta corrente come f64 e convertila a usize
                const label: usize = (@intFromFloat(yBatch.data[i]));

                // Crea un vettore one-hot con 10 posizioni
                for (0..numClasses) |j| {
                    if (j == label) {
                        oneHotYBatch.data[i * numClasses + j] = 1;
                    } else {
                        oneHotYBatch.data[i * numClasses + j] = 0;
                    }
                }
            }

            // Dealloca il vecchio yBatch e sostituiscilo con il nuovo tensore one-hot
            yBatch.deinit();
            yBatch.* = oneHotYBatch;
        }
    };
}
