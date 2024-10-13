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
    classification: bool,
    comptime lossType: LossType,
    comptime lr: f64,
) !void {
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
            var predictions = try model.forward(&load.xTensor);
            var shape: [2]usize = [_]usize{ load.yTensor.shape[0], 10 };
            try predictions.reshape(&shape);

            //compute loss
            std.debug.print("\n-------------------------------computing loss", .{});
            const loser = Loss.LossFunction(lossType){};
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
            var grad: Tensor.Tensor(T) = try loser.computeGradient(T, &predictions, &load.yTensor);
            var shape_: [1]usize = [_]usize{1};
            try grad.reshape(&shape_);

            //backwarding
            std.debug.print("\n-------------------------------backwarding", .{});
            _ = try model.backward(&grad);

            //optimizing
            std.debug.print("\n-------------------------------Optimizer Step", .{});
            var optimizer = Optim.Optimizer(T, XType, YType, Optim.optimizer_SGD, lr, allocator){ // Here we pass the actual instance of the optimizer
            };
            try optimizer.step(model);
            std.debug.print("Batch Bumber {}", .{step});
        }

        load.reset();
        std.debug.print("\n>>>>>>>>>>>> loss record:{any}", .{LossMeanRecord});
        std.debug.print("steps:{}", .{steps});
    }
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
