const std = @import("std");
const model_import_export = @import("model_import_export");
const Model = @import("model").Model;
const layer = @import("layers");
const Tensor = @import("tensor").Tensor;

test "Export of a 2D Tensor" {
    std.debug.print("\n     test: Export of a 2D Tensor", .{});

    const allocator = std.testing.allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 10, 2, 30 },
        [_]u8{ 4, 50, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try model_import_export.exportTensor(u8, tensor, "exportTryTensor.bin");

    var tensor_imported: Tensor(u8) = try model_import_export.importTensor(&allocator, u8, "exportTryTensor.bin");
    defer tensor_imported.deinit();

    for (0..tensor.data.len) |i| {
        try std.testing.expectEqual(tensor.data[i], tensor_imported.data[i]);
    }

    for (0..tensor.shape.len) |i| {
        try std.testing.expectEqual(tensor.shape[i], tensor_imported.shape[i]);
    }

    try std.testing.expectEqual(tensor.size, tensor_imported.size);
}
