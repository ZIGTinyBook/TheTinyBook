//! This file contains at the moment all the available options to train a model.
//! Dependng on your intentions you can use trainTensors(), a general trainer for tensors,
//! or TrainDataLoader(), more specific for training data loaded from a file. This last one has been well tested for MNIST.

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

/// Defines the type of trainer used for model training.
///
/// - `DataLoaderTrainer`: Uses a `DataLoader` to feed batches of data into the model during training.
/// - `TensorTrainer`: Uses direct tensor inputs for training.
pub const TrainerType = enum {
    DataLoaderTrainer,
    TensorTrainer,
};

pub fn TrainDataLoader(
    comptime T: type,
    comptime XType: type, // Input types
    comptime YType: type, // Output type
    comptime allocator: *const std.mem.Allocator,
    comptime batchSize: i16,
    features: usize,
    model: *Model(T, allocator),
    load: *DataLoader(T, XType, YType, batchSize),
    epochs: u32,
    comptime lossType: LossType,
    comptime lr: f64,
    training_size: f32,
) !void {
    var LossMeanRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(LossMeanRecord);

    var AccuracyRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(AccuracyRecord);

    var ValidationLossRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(ValidationLossRecord);

    var ValidationAccuracyRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(ValidationAccuracyRecord);

    var shapeXArr = [_]usize{ batchSize, features };
    var shapeYArr = [_]usize{batchSize};
    var shapeX: []usize = &shapeXArr;
    var shapeY: []usize = &shapeYArr;

    var steps: u16 = 0;
    try load.trainTestSplit(allocator, training_size);

    const train_len: u16 = @as(u16, @intCast(load.X_train.?.len));
    steps = @divFloor(train_len, batchSize);
    if (train_len % batchSize != 0) {
        steps += 1;
    }

    std.debug.print("Number of training steps: {}\n", .{steps});

    for (0..epochs) |i| {
        var totalCorrect: u16 = 0;
        var totalSamples: u16 = 0;

        var totalCorrectVal: u16 = 0;
        var totalSamplesVal: u16 = 0;

        var optimizer = Optim.Optimizer(T, XType, YType, Optim.optimizer_SGD, lr, allocator){};

        for (0..steps) |step| {
            _ = load.xTrainNextBatch(batchSize);
            _ = load.yTrainNextBatch(batchSize);
            try load.toTensor(allocator, &shapeX, &shapeY);
            try convertToOneHot(T, batchSize, &load.yTensor);

            var predictions = try model.forward(&load.xTensor);
            defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            try predictions.reshape(&shape);

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);

            var loss = try loser.computeLoss(T, &predictions, &load.yTensor);
            defer loss.deinit();

            const correctPredictions: u16 = try computeAccuracy(T, &predictions, &load.yTensor);
            totalCorrect += correctPredictions;
            totalSamples += batchSize;

            LossMeanRecord[i] = TensMath.mean(T, &loss);
            AccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrect)) / @as(f32, @floatFromInt(totalSamples)) * 100.0;

            var grad: Tensor.Tensor(T) = try loser.computeGradient(T, &predictions, &load.yTensor);
            defer grad.deinit();
            _ = try model.backward(&grad);

            try optimizer.step(model);

            std.debug.print("Training - Epoch: {}, Step: {}, Loss: {}, Accuracy: {} \n", .{ i + 1, step + 1, LossMeanRecord[i], AccuracyRecord[i] });
        }

        load.reset();

        const val_len: u16 = @as(u16, @intCast(load.X_test.?.len));
        var val_steps: u16 = @divFloor(val_len, batchSize);
        if (val_len % batchSize != 0) {
            val_steps += 1;
        }

        std.debug.print("\nNumber of validation steps: {}\n", .{val_steps});

        for (0..val_steps) |step| {
            _ = load.xTestNextBatch(batchSize);
            _ = load.yTestNextBatch(batchSize);
            try load.toTensor(allocator, &shapeX, &shapeY);
            try convertToOneHot(T, batchSize, &load.yTensor);

            var predictions = try model.forward(&load.xTensor);
            defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            try predictions.reshape(&shape);

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);

            var loss = try loser.computeLoss(T, &predictions, &load.yTensor);
            defer loss.deinit();

            const correctPredictions: u16 = try computeAccuracy(T, &predictions, &load.yTensor);
            totalCorrectVal += correctPredictions;
            totalSamplesVal += batchSize;

            ValidationLossRecord[i] = TensMath.mean(T, &loss);
            ValidationAccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrectVal)) / @as(f32, @floatFromInt(totalSamplesVal)) * 100.0;

            std.debug.print("\nValidation - Epoch: {}, Step: {}", .{ i + 1, step + 1 });
        }

        load.reset();

        std.debug.print("\nEpoch {}: Training Loss = {}, Training Accuracy = {}%", .{ i + 1, LossMeanRecord[i], AccuracyRecord[i] });
        std.debug.print("\nEpoch {}: Validation Loss = {}, Validation Accuracy = {}%", .{ i + 1, ValidationLossRecord[i], ValidationAccuracyRecord[i] });
    }
}

pub fn TrainDataLoader2D(
    comptime T: type,
    comptime XType: type, // Input types
    comptime YType: type, // Output type
    allocator: *const std.mem.Allocator,
    comptime batchSize: i16,
    features: usize,
    model: *Model(T),
    load: *DataLoader(T, XType, YType, batchSize, 3),
    epochs: u32,
    comptime lossType: LossType,
    comptime lr: f64,
    training_size: f32,
) !void {
    var LossMeanRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(LossMeanRecord);

    var AccuracyRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(AccuracyRecord);

    var ValidationLossRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(ValidationLossRecord);

    var ValidationAccuracyRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(ValidationAccuracyRecord);

    _ = features;
    var shapeXArr = [_]usize{ batchSize, 1, 28, 28 };
    var shapeYArr = [_]usize{batchSize};
    var shapeX: []usize = &shapeXArr;
    var shapeY: []usize = &shapeYArr;

    var steps: u16 = 0;
    try load.trainTestSplit(allocator, training_size);

    const train_len: u16 = @as(u16, @intCast(load.X_train.?.len));
    steps = @divFloor(train_len, batchSize);
    if (train_len % batchSize != 0) {
        steps += 1;
    }

    std.debug.print("Number of training steps: {}\n", .{steps});

    for (0..epochs) |i| {
        var totalCorrect: u16 = 0;
        var totalSamples: u16 = 0;

        var totalCorrectVal: u16 = 0;
        var totalSamplesVal: u16 = 0;

        for (0..steps) |step| {
            _ = load.xTrainNextBatch(batchSize);
            _ = load.yTrainNextBatch(batchSize);
            try load.toTensor(allocator, &shapeX, &shapeY);

            try convertToOneHot(T, batchSize, &load.yTensor);

            var predictions = try model.forward(&load.xTensor);
            defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            try predictions.reshape(&shape);

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);

            var loss = try loser.computeLoss(T, &predictions, &load.yTensor);
            defer loss.deinit();

            const correctPredictions: u16 = try computeAccuracy(T, &predictions, &load.yTensor);
            totalCorrect += correctPredictions;
            totalSamples += batchSize;

            LossMeanRecord[i] = TensMath.mean(T, &loss);
            AccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrect)) / @as(f32, @floatFromInt(totalSamples)) * 100.0;

            var grad: Tensor.Tensor(T) = try loser.computeGradient(T, &predictions, &load.yTensor);
            defer grad.deinit();
            _ = try model.backward(&grad);

            var optimizer = Optim.Optimizer(T, XType, YType, Optim.optimizer_SGD, lr){};
            try optimizer.step(model);

            std.debug.print("Training - Epoch: {}, Step: {}\n", .{ i + 1, step + 1 });
        }

        load.reset();

        const val_len: u16 = @as(u16, @intCast(load.X_test.?.len));
        var val_steps: u16 = @divFloor(val_len, batchSize);
        if (val_len % batchSize != 0) {
            val_steps += 1;
        }

        std.debug.print("Number of validation steps: {}\n", .{val_steps});

        for (0..val_steps) |step| {
            _ = load.xTestNextBatch(batchSize);
            _ = load.yTestNextBatch(batchSize);
            try load.toTensor(allocator, &shapeX, &shapeY);
            try convertToOneHot(T, batchSize, &load.yTensor);

            var predictions = try model.forward(&load.xTensor);
            defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            try predictions.reshape(&shape);

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);

            var loss = try loser.computeLoss(T, &predictions, &load.yTensor);
            defer loss.deinit();

            const correctPredictions: u16 = try computeAccuracy(T, &predictions, &load.yTensor);
            totalCorrectVal += correctPredictions;
            totalSamplesVal += batchSize;

            ValidationLossRecord[i] = TensMath.mean(T, &loss);
            ValidationAccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrectVal)) / @as(f32, @floatFromInt(totalSamplesVal)) * 100.0;

            std.debug.print("Validation - Epoch: {}, Step: {}\n", .{ i + 1, step + 1 });
        }

        load.reset();

        std.debug.print("Epoch {}: Training Loss = {}, Training Accuracy = {}%\n", .{ i + 1, LossMeanRecord[i], AccuracyRecord[i] });
        std.debug.print("Epoch {}: Validation Loss = {}, Validation Accuracy = {}%\n", .{ i + 1, ValidationLossRecord[i], ValidationAccuracyRecord[i] });
    }
}

/// Computes the accuracy of model predictions by comparing predicted and actual labels.
fn computeAccuracy(comptime T: type, predictions: *Tensor.Tensor(T), targets: *Tensor.Tensor(T)) !u16 {
    var correct: u16 = 0;
    const rows = predictions.shape[0];
    const cols = predictions.shape[1];

    for (0..rows) |i| {
        var predictedLabel: usize = 0;
        var maxVal: T = predictions.data[i * cols];

        // Find the class with the highest value in predictions
        for (1..cols) |j| {
            const val = predictions.data[i * cols + j];
            if (val > maxVal) {
                maxVal = val;
                predictedLabel = j;
            }
        }

        // Find the actual label
        var actualLabel: usize = 0;
        var maxTargetVal: T = targets.data[i * cols];
        for (1..cols) |j| {
            const val = targets.data[i * cols + j];
            if (val > maxTargetVal) {
                maxTargetVal = val;
                actualLabel = j;
            }
        }

        // Check if the prediction is correct
        if (predictedLabel == actualLabel) {
            correct += 1;
        }
    }

    return correct;
}

fn convertToOneHot(comptime T: type, batchSize: i16, yBatch: *Tensor.Tensor(T)) !void {
    const numClasses = 10;

    var shapeYArr = [_]usize{ @intCast(batchSize), numClasses };
    const oneHotShape = &shapeYArr;

    var oneHotYBatch = try Tensor.Tensor(T).fromShape(yBatch.allocator, oneHotShape);

    for (0..@intCast(batchSize)) |i| {
        const label: usize = (@intFromFloat(yBatch.data[i]));
        for (0..numClasses) |j| {
            if (j == label) {
                oneHotYBatch.data[i * numClasses + j] = 1;
            } else {
                oneHotYBatch.data[i * numClasses + j] = 0;
            }
        }
    }

    yBatch.deinit();
    yBatch.* = oneHotYBatch;
}

pub fn trainTensors(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    model: *Model(T),
    input: *Tensor.Tensor(T),
    targets: *Tensor.Tensor(T),
    epochs: u32,
    comptime lr: f64,
) !void {
    var LossMeanRecord: []f32 = try allocator.alloc(f32, epochs);
    defer allocator.free(LossMeanRecord);

    for (0..epochs) |i| {
        std.debug.print("\n\n----------------------epoch:{}", .{i});

        // Forward pass
        std.debug.print("\n-------------------------------forwarding", .{});
        var predictions = try model.forward(input);
        defer predictions.deinit();

        // Loss computation
        std.debug.print("\n-------------------------------computing loss", .{});
        const loser = Loss.LossFunction(LossType.MSE){};
        var loss = try loser.computeLoss(T, &predictions, targets);
        defer loss.deinit();

        LossMeanRecord[i] = TensMath.mean(T, &loss);
        std.debug.print("\n     loss:{}", .{LossMeanRecord[i]});

        // Gradient computation
        std.debug.print("\n-------------------------------computing loss gradient", .{});
        var grad: Tensor.Tensor(T) = try loser.computeGradient(T, &predictions, targets);
        defer grad.deinit();
        std.debug.print("\n     gradient:", .{});

        // Backpropagation
        std.debug.print("\n-------------------------------backwarding", .{});
        _ = try model.backward(&grad);

        // Optimization
        std.debug.print("\n-------------------------------Optimizer Step", .{});
        var optimizer = Optim.Optimizer(T, T, T, Optim.optimizer_SGD, lr){};
        try optimizer.step(model);
    }

    std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
}
