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
    // T must be float
    if (@typeInfo(T) != .Float) return error.NotFloatType;

    const rows = tensor.shape[0];
    const cols = if (tensor.shape.len == 2) tensor.shape[1] else 1;

    // Normalise
    var max: T = tensor.data[0];
    var min: T = tensor.data[0];
    var delta: T = 0;

    // If tensor is 1D
    if (tensor.shape.len == 1) {
        // Find max and min for 1D tensor
        for (0..rows) |i| {
            if (tensor.data[i] > max) max = tensor.data[i];
            if (tensor.data[i] < min) min = tensor.data[i];
        }
        delta = max - min;
        //std.debug.print("\n 1D min:{} max:{} delta:{}", .{ min, max, delta });

        // Update tensor for 1D normalization
        for (0..rows) |i| {
            tensor.data[i] = if (delta == 0) ((tensor.data[i] - min)) else ((tensor.data[i] - min) / delta);
        }
    } else {
        // 2D tensor case
        for (0..rows) |i| {
            max = tensor.data[i * cols];
            min = tensor.data[i * cols];
            // Find max and min for each row
            for (0..cols) |j| {
                if (tensor.data[i * cols + j] > max) max = tensor.data[i * cols + j];
                if (tensor.data[i * cols + j] < min) min = tensor.data[i * cols + j];
            }
            delta = max - min;
            //std.debug.print("\n 2D min:{} max:{} delta:{}", .{ min, max, delta });

            // Update tensor for 2D normalization
            for (0..cols) |j| {
                std.debug.print("\n {}-{} / {}", .{ tensor.data[i * cols + j], min, delta });
                tensor.data[i * cols + j] = if (delta == 0) ((tensor.data[i] - min)) else ((tensor.data[i] - min) / delta);
            }
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
