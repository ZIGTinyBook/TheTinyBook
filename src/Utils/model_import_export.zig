const std = @import("std");
const cwd = std.fs.cwd();
const Model = @import("model");
const Tensor = @import("tensor").Tensor;

pub fn exportTensor(comptime T: type, tensor: Tensor(T), file_path: []const u8) !void {
    var file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();

    const writer = file.writer();

    // Write tensor size and shape
    try writer.writeInt(usize, tensor.size, std.builtin.Endian.little);
    try writer.writeInt(usize, tensor.shape.len, std.builtin.Endian.little);
    for (tensor.shape) |dim| {
        try writer.writeInt(usize, dim, std.builtin.Endian.little);
    }

    // Write tensor data
    for (tensor.data) |value| {
        try writer.writeInt(T, value, std.builtin.Endian.little);
    }
}

pub fn importTensor(allocator: *const std.mem.Allocator, comptime T: type, file_path: []const u8) !Tensor(T) {
    var file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const reader = file.reader();

    // Read tensor size
    const tensor_size: usize = try reader.readInt(usize, std.builtin.Endian.little);

    // Read tensor shape lenght
    const tensor_shapeLen: usize = try reader.readInt(usize, std.builtin.Endian.little);

    // Read tensor shape
    const tensor_shape = try allocator.alloc(usize, tensor_shapeLen);
    for (0..tensor_shapeLen) |i| {
        tensor_shape[i] = try reader.readInt(usize, std.builtin.Endian.little);
    }

    const tensor_data = try allocator.alloc(T, tensor_size);
    // Read tensor data
    for (0..tensor_size) |i| {
        tensor_data[i] = try reader.readInt(T, std.builtin.Endian.little);
    }

    return Tensor(T){
        .data = tensor_data,
        .size = tensor_size,
        .shape = tensor_shape,
        .allocator = allocator,
    };
}
