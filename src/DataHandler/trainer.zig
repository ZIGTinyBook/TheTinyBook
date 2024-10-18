const std = @import("std");
const Tensor = @import("tensor");
const TensMath = @import("tensor_m");
const Model = @import("model").Model;

const Loss = @import("loss");
const Optim = @import("optim");

const DataLoader = @import("dataloader").DataLoader;
const DataProc = @import("dataprocessor");

const LossType = @import("loss").LossType;
const NormalizType = @import("dataprocessor").NormalizationType;

pub const TrainerType = enum {
    DataLoaderTrainer,
    TensorTrainer,
};

pub fn TrainDataLoader(
    comptime T: type,
    comptime XType: type, //input types
    comptime YType: type, //output type
    comptime allocator: *const std.mem.Allocator,
    comptime batchSize: i16,
    features: usize,
    model: *Model(T, allocator),
    load: *DataLoader(T, XType, YType, batchSize),
    ephocs: u32,
    comptime lossType: LossType,
    comptime lr: f64,
) !void {
    var LossMeanRecord: []f32 = try allocator.alloc(f32, ephocs);
    defer allocator.free(LossMeanRecord);
    var AccuracyRecord: []f32 = try allocator.alloc(f32, ephocs); // Array per
    defer allocator.free(AccuracyRecord);
    var shapeXArr = [_]usize{ batchSize, features };
    var shapeYArr = [_]usize{batchSize};
    var shapeX: []usize = &shapeXArr;
    var shapeY: []usize = &shapeYArr;
    var steps: u16 = 0;

    const len: u16 = @as(u16, @intCast(load.X.len));
    steps = @divFloor(len, batchSize);
    std.debug.print("\n\n----------------------len:{}", .{len});
    if (len % batchSize != 0) {
        steps += 1;
    }

    for (0..ephocs) |i| {
        std.debug.print("\n\n----------------------epoch:{}", .{i});
        var totalCorrect: u16 = 0; // Per il calcolo dell'accuratezza
        var totalSamples: u16 = 0;
        for (0..steps) |step| {
            _ = load.xNextBatch(batchSize);
            _ = load.yNextBatch(batchSize);
            try load.toTensor(allocator, &shapeX, &shapeY);
            try convertToOneHot(T, batchSize, &load.yTensor);

            //forwarding
            std.debug.print("\n-------------------------------forwarding", .{});
            //try DataProc.normalize(T, &load.xTensor, NormalizType.UnityBasedNormalizartion);
            var predictions = try model.forward(&load.xTensor);
            defer predictions.deinit();
            defer predictions.deinit();
            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            try predictions.reshape(&shape);

            // Compute loss
            std.debug.print("\n-------------------------------computing loss", .{});
            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);
            var loss = try loser.computeLoss(T, &predictions, &load.yTensor);
            //compute accuracy
            // Compute accuracy
            const correctPredictions: u16 = try computeAccuracy(T, &predictions, &load.yTensor);
            totalCorrect += correctPredictions;
            totalSamples += batchSize;

            LossMeanRecord[i] = TensMath.mean(T, &loss);
            // Converti totalCorrect e totalSamples in f32 per evitare errori di tipo
            AccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrect)) / @as(f32, @floatFromInt(totalSamples)) * 100.0;
            std.debug.print("\n     loss:{} accuracy:{}%", .{ LossMeanRecord[i], AccuracyRecord[i] });

            //compute gradient of the loss
            std.debug.print("\n-------------------------------computing loss gradient", .{});
            var grad: Tensor.Tensor(T) = try loser.computeGradient(T, &predictions, &load.yTensor);

            //backwarding
            std.debug.print("\n-------------------------------backwarding", .{});
            _ = try model.backward(&grad);

            //optimizing
            std.debug.print("\n-------------------------------Optimizer Step", .{});
            var optimizer = Optim.Optimizer(T, XType, YType, Optim.optimizer_SGD, lr, allocator){ // Here we pass the actual instance of the optimizer
            };
            try optimizer.step(model);
            std.debug.print("\n Batch Bumber {}\n", .{step});
        }

        load.reset();
        std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
        std.debug.print("steps:{}", .{steps});
    }
}

fn computeAccuracy(comptime T: type, predictions: *Tensor.Tensor(T), targets: *Tensor.Tensor(T)) !u16 {
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

fn convertToOneHot(comptime T: type, batchSize: i16, yBatch: *Tensor.Tensor(T)) !void {
    // Numero di classi
    const numClasses = 10;

    // Crea una forma per il tensore one-hot: batchSize x numClasses
    var shapeYArr = [_]usize{ @intCast(batchSize), numClasses };
    const oneHotShape = &shapeYArr;

    // Crea un nuovo tensore per yBatch in formato one-hot
    var oneHotYBatch = try Tensor.Tensor(T).fromShape(yBatch.allocator, oneHotShape);

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

pub fn trainTensors(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    model: *Model(T, allocator),
    input: *Tensor.Tensor(T),
    targets: *Tensor.Tensor(T),
    epochs: u32,
    comptime lr: f64,
) !void {
    var LossMeanRecord: []f32 = try allocator.alloc(f32, epochs);

    for (0..epochs) |i| {
        std.debug.print("\n\n----------------------epoch:{}", .{i});

        //forwarding
        std.debug.print("\n-------------------------------forwarding", .{});
        var predictions = try model.forward(input);
        defer predictions.deinit();

        //compute loss
        std.debug.print("\n-------------------------------computing loss", .{});
        const loser = Loss.LossFunction(LossType.MSE){};
        var loss = try loser.computeLoss(T, &predictions, targets);

        //compute accuracy
        LossMeanRecord[i] = TensMath.mean(T, &loss);
        std.debug.print("\n     loss:{}", .{LossMeanRecord[i]});

        //compute gradient of the loss
        std.debug.print("\n-------------------------------computing loss gradient", .{});
        var grad: Tensor.Tensor(T) = try loser.computeGradient(T, &predictions, targets);
        std.debug.print("\n     gradient:", .{});
        grad.info();

        //backwarding
        std.debug.print("\n-------------------------------backwarding", .{});
        _ = try model.backward(&grad);

        //optimizing
        std.debug.print("\n-------------------------------Optimizer Step", .{});
        var optimizer = Optim.Optimizer(T, T, T, Optim.optimizer_SGD, lr, allocator){ // Here we pass the actual instance of the optimizer
        };
        try optimizer.step(model);
    }

    std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
}
