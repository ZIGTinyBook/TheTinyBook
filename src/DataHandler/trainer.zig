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

/// Trains a model using data from a `DataLoader` over multiple epochs with specified
/// batch size and learning rate.
///
/// # Type Parameters
/// - `T`: The data type for the tensor elements in the model.
/// - `XType`: The data type for the input tensor (X).
/// - `YType`: The data type for the output tensor (Y).
///
/// # Parameters
/// - `allocator`: Memory allocator for dynamic allocations during training.
/// - `batchSize`: The number of samples in each batch.
/// - `features`: The number of features in each input sample.
/// - `model`: A pointer to the model to be trained.
/// - `load`: A pointer to the `DataLoader` that provides data batches.
/// - `ephocs`: The total number of epochs to train for.
/// - `lossType`: The type of loss function used during training.
/// - `lr`: The learning rate for model optimization.
/// - `training_size` : the percentage to use as training
///
/// # Returns
/// - `!void`: Returns an error if any allocation or training step fails.
///
/// # Description
/// This function allocates memory to track loss and accuracy across epochs and initializes
/// tensors for the input (X) and output (Y) shapes based on `batchSize` and `features`.
/// `steps` is used to count training steps within each epoch. Resources are freed automatically
/// after each epoch, ensuring efficient memory use.
///
/// # Notes
/// - `LossMeanRecord` and `AccuracyRecord` are arrays that store mean loss and accuracy
///   values over `ephocs`, providing performance metrics across the training process.
/// - `shapeX` and `shapeY` define the shape of input and output tensors for each batch, look at
///   load.xNextBatch() and load.yNextBatch()
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
            //defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            try predictions.reshape(&shape);

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);
            var loss = try loser.computeLoss(T, &predictions, &load.yTensor);

            const correctPredictions: u16 = try computeAccuracy(T, &predictions, &load.yTensor);
            totalCorrect += correctPredictions;
            totalSamples += batchSize;

            LossMeanRecord[i] = TensMath.mean(T, &loss);
            AccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrect)) / @as(f32, @floatFromInt(totalSamples)) * 100.0;

            var grad: Tensor.Tensor(T) = try loser.computeGradient(T, &predictions, &load.yTensor);
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
            //defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            try predictions.reshape(&shape);

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);
            var loss = try loser.computeLoss(T, &predictions, &load.yTensor);

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
    //TODO need to be changed for not 2D images
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
            //defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            try predictions.reshape(&shape);

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);
            var loss = try loser.computeLoss(T, &predictions, &load.yTensor);

            const correctPredictions: u16 = try computeAccuracy(T, &predictions, &load.yTensor);
            totalCorrect += correctPredictions;
            totalSamples += batchSize;

            LossMeanRecord[i] = TensMath.mean(T, &loss);
            AccuracyRecord[i] = @as(f32, @floatFromInt(totalCorrect)) / @as(f32, @floatFromInt(totalSamples)) * 100.0;

            var grad: Tensor.Tensor(T) = try loser.computeGradient(T, &predictions, &load.yTensor);
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
            //defer predictions.deinit();

            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            try predictions.reshape(&shape);

            const loser = Loss.LossFunction(lossType){};
            try DataProc.normalize(T, &load.yTensor, NormalizType.UnityBasedNormalizartion);
            var loss = try loser.computeLoss(T, &predictions, &load.yTensor);

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
///
/// This function iterates through a batch of predictions and targets, identifying the
/// label with the highest value for each row (representing a sample) and counting the
/// number of correct predictions.
///
/// # Type Parameters
/// - `T`: The data type of tensor elements (e.g., `f32` or `f64`).
///
/// # Parameters
/// - `predictions`: A pointer to a tensor containing predicted values. Each row represents
///   a sample, and each column represents the probability or score for a class.
/// - `targets`: A pointer to a tensor containing the true labels, in a one-hot encoded format.
///   Each row represents a sample, and each column indicates the presence (1) or absence (0)
///   of a specific class.
///
/// # Returns
/// - `!u16`: The number of correctly predicted samples. Returns an error if accessing
///   tensor data fails.
///
/// # Errors
/// Returns an error if issues occur when accessing tensor data (e.g., if the tensors
/// are of incompatible shapes).
///
/// # Algorithm
/// - For each sample (row in `predictions` and `targets`):
///   1. Find the class with the maximum value in `predictions` as the predicted label.
///   2. Find the class with the maximum value in `targets` as the actual label.
///   3. If the predicted label matches the actual label, increment the correct count.
/// - Finally, the function returns the count of correctly predicted samples.
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

/// Trains a model using tensors for input and target data over a specified number of epochs.
///
/// This function performs a training loop that includes forward propagation, loss computation,
/// accuracy calculation, gradient computation, backpropagation, and model optimization for each epoch.
///
/// # Type Parameters
/// - `T`: The data type of tensor elements (e.g., `f32` or `f64`).
///
/// # Parameters
/// - `allocator`: Memory allocator used for dynamic memory allocations during training.
/// - `model`: A pointer to the model to be trained.
/// - `input`: A tensor containing input data for the model.
/// - `targets`: A tensor containing the target data the model will be trained to predict.
/// - `epochs`: The number of training epochs.
/// - `lr`: The learning rate for the optimizer.
///
/// # Returns
/// - `!void`: Returns an error if any allocation, forward pass, backward pass, or optimization step fails.
///
/// # Algorithm
/// - `LossMeanRecord`: Records the mean loss for each epoch, which is dynamically allocated at the start
///   of the function and freed after training completes.
/// - For each epoch:
///   1. **Forward Propagation**: Computes model predictions based on the current `input`.
///   2. **Loss Computation**: Calculates loss using the specified loss function (`LossType.MSE`) and records the mean loss.
///   3. **Gradient Calculation**: Computes the gradient of the loss with respect to model predictions.
///   4. **Backpropagation**: Adjusts model parameters based on the computed gradient.
///   5. **Optimization**: Updates model weights using Stochastic Gradient Descent (SGD) with the specified `lr`.
///
/// # Notes
/// - Loss and gradient values are printed at each epoch for monitoring training progress.
/// - `LossMeanRecord` is printed at the end to provide a summary of the training loss trend over epochs.
///
/// # Errors
/// This function returns an error if memory allocation, tensor operations, or optimizer steps fail at any point.
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
    defer allocator.free(LossMeanRecord);

    for (0..epochs) |i| {
        std.debug.print("\n\n----------------------epoch:{}", .{i});

        // Forward pass
        std.debug.print("\n-------------------------------forwarding", .{});
        var predictions = try model.forward(input);
        //predictions.info();
        //defer predictions.deinit();

        // Loss computation
        std.debug.print("\n-------------------------------computing loss", .{});
        const loser = Loss.LossFunction(LossType.MSE){};
        var loss = try loser.computeLoss(T, &predictions, targets);
        defer loss.deinit();

        // Accuracy calculation
        LossMeanRecord[i] = TensMath.mean(T, &loss);
        std.debug.print("\n     loss:{}", .{LossMeanRecord[i]});

        // Gradient computation
        std.debug.print("\n-------------------------------computing loss gradient", .{});
        var grad: Tensor.Tensor(T) = try loser.computeGradient(T, &predictions, targets);
        std.debug.print("\n     gradient:", .{});
        // grad.info();

        // Backpropagation
        std.debug.print("\n-------------------------------backwarding", .{});
        _ = try model.backward(&grad);

        // Optimization
        std.debug.print("\n-------------------------------Optimizer Step", .{});
        var optimizer = Optim.Optimizer(T, T, T, Optim.optimizer_SGD, lr, allocator){};
        try optimizer.step(model);
    }

    std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
}
