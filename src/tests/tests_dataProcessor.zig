const std = @import("std");
const DataProc = @import("dataprocessor");
const NormalizType = @import("dataprocessor").NormalizationType;
const Tensor = @import("tensor").Tensor;

test "normalize float" {
    std.debug.print("\n     test: normalize float", .{});

    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try DataProc.normalize(f32, &t1, NormalizType.UnityBasedNormalizartion);
}

test "normalize float all different" {
    std.debug.print("\n     test: normalize float", .{});

    const allocator = std.heap.page_allocator;

    var inputArray: [2][4]f32 = [_][4]f32{
        [_]f32{ 1.0, 2.0, 3.0, 10.0 },
        [_]f32{ 4.0, 5.0, 1.0, 2.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 4 }; // 2x4 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try DataProc.normalize(f32, &t1, NormalizType.UnityBasedNormalizartion);

    try std.testing.expect(1.0 == t1.data[3]);
    try std.testing.expect(1.0 == t1.data[5]);

    for (t1.data) |*val| {
        try std.testing.expect(1.0 >= val.*);
        try std.testing.expect(0.0 <= val.*);
    }
}
