const std = @import("std");
const Tensor = @import("./tensors.zig").Tensor;

test "Sizetest" {
    const allocator = std.heap.page_allocator;
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f64).init(&allocator, &shape);
    const size = tensor.getSize();
    try std.testing.expect(size == 6);
}

test "GetSetTest" {
    const allocator = std.heap.page_allocator;
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f64).init(&allocator, &shape);
    try tensor.set(0, 1.0);
    const value = try tensor.get(0);
    try std.testing.expect(value == 1.0);
}

test "FlattenIndexTest" {
    const allocator = std.heap.page_allocator;
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f64).init(&allocator, &shape);
    var indices = [_]usize{ 1, 2 };
    const flatIndex = try tensor.flatten_index(&indices);
    std.debug.print("\nflatIndex: {}\n", .{flatIndex});
    try std.testing.expect(flatIndex == 5);
    indices = [_]usize{ 0, 0 };
    const flatIndex2 = try tensor.flatten_index(&indices);
    std.debug.print("\nflatIndex2: {}\n", .{flatIndex2});
    try std.testing.expect(flatIndex2 == 0);
}

test "GetSetAtTest" {
    const allocator = std.heap.page_allocator;
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f64).init(&allocator, &shape);
    var indices = [_]usize{ 1, 2 };
    try tensor.set_at(&indices, 1.0);
    const value = try tensor.get_at(&indices);
    try std.testing.expect(value == 1.0);
}

test "SetAtTest" {
    const allocator = std.heap.page_allocator;
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f64).init(&allocator, &shape);
    var indices = [_]usize{ 1, 2 };
    try tensor.set_at(&indices, 1.0);
    const value = try tensor.get_at(&indices);
    try std.testing.expect(value == 1.0);
}

test "test Tensor fromArray" {
    const allocator = std.testing.allocator;

    // Definisco un array multidimensionale
    var inputArray: [2][3]f64 = [_][3]f64{
        [_]f64{ 1.0, 2.0, 3.0 },
        [_]f64{ 4.0, 5.0, 6.0 },
    };

    // Definisco la forma desiderata
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(f64).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try std.testing.expectEqual(6, tensor.size);

    try std.testing.expectEqual(2, tensor.shape[0]);
    try std.testing.expectEqual(3, tensor.shape[1]);

    try std.testing.expectEqual(1.0, tensor.data[0]);
    try std.testing.expectEqual(2.0, tensor.data[1]);
    try std.testing.expectEqual(3.0, tensor.data[2]);
    try std.testing.expectEqual(4.0, tensor.data[3]);
    try std.testing.expectEqual(5.0, tensor.data[4]);
    try std.testing.expectEqual(6.0, tensor.data[5]);

    const value = try tensor.get_at(&[_]usize{ 1, 2 });
    try std.testing.expectEqual(6.0, value);

    const tensorSize = tensor.getSize();
    try std.testing.expectEqual(6, tensorSize);
}
