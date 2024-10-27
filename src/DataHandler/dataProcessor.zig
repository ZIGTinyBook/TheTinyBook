const std = @import("std");
const Tensor = @import("tensor").Tensor;

pub const NormalizationType = enum {
    null,
    UnityBasedNormalizartion,
    StandardDeviationNormalizartion,
};

pub fn normalize(comptime T: anytype, tensor: *Tensor(T), normalizationType: NormalizationType) !void {
    switch (normalizationType) {
        NormalizationType.UnityBasedNormalizartion => try multidimNormalizeUnityBased(T, tensor),
        else => try multidimNormalizeUnityBased(T, tensor),
    }
}

/// Normalize each row in a multidimensional tensor
fn multidimNormalizeUnityBased(comptime T: anytype, tensor: *Tensor(T)) !void {
    const epsilon: T = 1e-10;
    // --- Checks ---
    // T must be float
    if (@typeInfo(T) != .Float) return error.NotFloatType;

    var counter: usize = 0; //counter counts the rows in all the tensor, indipendently of the shape
    const cols: usize = tensor.shape[tensor.shape.len - 1]; //aka: elements per row
    const numb_of_rows = tensor.data.len / cols;

    // Normalise
    var max: T = tensor.data[0];
    var min: T = tensor.data[0];
    var delta: T = 0;

    while (counter < numb_of_rows) {

        // Find max and min for each row
        for (0..cols) |i| {
            if (tensor.data[counter * cols + i] > max) max = tensor.data[i];
            if (tensor.data[counter * cols + i] < min) min = tensor.data[i];
        }
        delta = max - min + epsilon;

        // Update tensor for 1D normalization
        for (0..cols) |i| {
            tensor.data[counter * cols + i] = if (delta == 0) 0 else (tensor.data[counter * cols + i] - min) / delta;
        }

        counter += 1;
    }
}
