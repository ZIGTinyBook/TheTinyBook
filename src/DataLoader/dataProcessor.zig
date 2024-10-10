const std = @import("std");
const Tensor = @import("tensor").Tensor;

pub const NormalizationType = enum {
    null,
    UnityBasedNormalizartion,
    StandardDeviationNormalizartion,
};

pub fn normalize(comptime T: anytype, tensor: *Tensor(T), normalizationType: NormalizationType) !void {
    switch (normalizationType) {
        NormalizationType.UnityBasedNormalizartion => try normalizeUnityBased2D(T, tensor),
        else => try normalizeUnityBased2D(T, tensor),
    }
}

// implements unity-based normalization
fn normalizeUnityBased2D(comptime T: anytype, tensor: *Tensor(T)) !void {

    // --- Checks ---
    //2D only
    if (tensor.shape.len > 2) return error.TooManyDimensions;
    //T must be float necessary
    if (@typeInfo(T) != .Float) return error.NotFloatType;

    const rows = tensor.shape[0];
    const cols = tensor.shape[1];

    //normalise
    var max: T = tensor.data[0];
    var min: T = tensor.data[0];
    var delta: T = 0;

    for (0..rows) |i| {
        max = tensor.data[i * cols];
        min = tensor.data[i * cols];
        //find max and min
        for (0..cols) |j| {
            if (tensor.data[i * cols + j] > max) max = tensor.data[i * cols + j];
            if (tensor.data[i * cols + j] < min) min = tensor.data[i * cols + j];
        }
        delta = max - min;
        std.debug.print("\n min:{} max:{} delta:{}", .{ min, max, delta });

        //update tensor
        for (0..cols) |j| {
            tensor.data[i * cols + j] = (tensor.data[i * cols + j] - min) / delta;
        }
    }
}

// fn normalizeStandardDeviation2D(comptime T: anytype, tensor: *Tensor(T)) !void {

//     // --- Checks ---
//     //2D only
//     if (tensor.shape.len > 2) return error.TooManyDimensions;
//     //T must be float necessary
//     if (@typeInfo(T) != .Float) return error.NotFloatType;

//     const rows = tensor.shape[0];
//     const cols = tensor.shape[1];

//     //normalize
//     var sum: T;
//     var mean: T;

//     for (0..rows) |i| {
//         sum = 0.0;
//         //find mean
//         for (0..cols) |j| {
//             sum += tensor.data[i * cols + j];
//         }
//         mean = sum /
//         delta = max - min;
//         //update tensor
//         for (0..cols) |j| {
//             tensor.data[i * cols + j] = tensor.data[i * cols + j] - min / delta;
//         }
//     }

// }
