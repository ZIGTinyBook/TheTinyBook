const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;

test "Tensor test description" {
    std.debug.print("\n--- Running tensor tests\n", .{});
}

test "Sizetest" {
    std.debug.print("\n     test: Size ", .{});
    const allocator = std.heap.page_allocator;
    var tensor = try Tensor(f64).init(&allocator);
    const size = tensor.getSize();
    try std.testing.expect(size == 0);
}

test "initialization fromShape" {
    std.debug.print("\n     test:initialization fromShape", .{});
    const allocator = std.heap.page_allocator;
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f64).fromShape(&allocator, &shape);
    const size = tensor.getSize();
    try std.testing.expect(size == 6);
    for (0..tensor.size) |i| {
        const val = try tensor.get(i);
        try std.testing.expect(val == 0);
    }
}

test "Get_Set_Test" {
    std.debug.print("\n     test:Get_Set_Test", .{});
    const allocator = std.heap.page_allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    try tensor.set(5, 33);
    const val = try tensor.get(5);

    try std.testing.expect(val == 33);
}

test "Flatten Index Test" {
    std.debug.print("\n     test:Flatten Index Test", .{});
    const allocator = std.heap.page_allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var indices = [_]usize{ 1, 2 };
    const flatIndex = try tensor.flatten_index(&indices);

    //std.debug.print("\nflatIndex: {}\n", .{flatIndex});
    try std.testing.expect(flatIndex == 5);
    indices = [_]usize{ 0, 0 };
    const flatIndex2 = try tensor.flatten_index(&indices);
    //std.debug.print("\nflatIndex2: {}\n", .{flatIndex2});
    try std.testing.expect(flatIndex2 == 0);
}

test "Get_at Set_at Test" {
    std.debug.print("\n     test:Get_at Set_at Test", .{});
    const allocator = std.heap.page_allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();
    var indices = [_]usize{ 1, 1 };
    var value = try tensor.get_at(&indices);
    try std.testing.expect(value == 5.0);
    //try tensor.set_at(&indices, 1.0);
    //tensor.info();

    for (0..2) |i| {
        for (0..3) |j| {
            indices[0] = i;
            indices[1] = j;
            value = try tensor.get_at(&indices);
            //std.debug.print(" ({},{}):{} ", .{ i, j, value });
            try std.testing.expect(value == i * 3 + j + 1);
        }
    }

    try tensor.set_at(&indices, 1.0);
    value = try tensor.get_at(&indices);
    try std.testing.expect(value == 1.0);
}

test "init than fill " {
    std.debug.print("\n     test:init than fill ", .{});
    const allocator = std.heap.page_allocator;
    var tensor = try Tensor(u8).init(&allocator);
    defer tensor.deinit();

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    try tensor.fill(&inputArray, &shape);

    try std.testing.expect(tensor.data[0] == 1);
    try std.testing.expect(tensor.data[1] == 2);
    try std.testing.expect(tensor.data[2] == 3);
    try std.testing.expect(tensor.data[3] == 4);
    try std.testing.expect(tensor.data[4] == 5);
    try std.testing.expect(tensor.data[5] == 6);
}

test "fromArray than fill " {
    std.debug.print("\n     test:fromArray than fill ", .{});
    const allocator = std.heap.page_allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 10, 20, 30 },
        [_]u8{ 40, 50, 60 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var inputArray2: [3][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
        [_]u8{ 7, 8, 9 },
    };
    var shape2: [2]usize = [_]usize{ 3, 3 };

    try tensor.fill(&inputArray2, &shape2);

    try std.testing.expect(tensor.data[0] == 1);
    try std.testing.expect(tensor.data[1] == 2);
    try std.testing.expect(tensor.data[2] == 3);
    try std.testing.expect(tensor.data[3] == 4);
    try std.testing.expect(tensor.data[4] == 5);
    try std.testing.expect(tensor.data[5] == 6);
    try std.testing.expect(tensor.data[6] == 7);
    try std.testing.expect(tensor.data[7] == 8);
    try std.testing.expect(tensor.data[8] == 9);
}

test " copy() method" {
    std.debug.print("\n     test:copy() method ", .{});
    const allocator = std.heap.page_allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 10, 20, 30 },
        [_]u8{ 40, 50, 60 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var tensorCopy = try tensor.copy();
    defer tensorCopy.deinit();

    for (0..tensor.data.len) |i| {
        try std.testing.expect(tensor.data[i] == tensorCopy.data[i]);
    }

    for (0..tensor.shape.len) |i| {
        try std.testing.expect(tensor.shape[i] == tensorCopy.shape[i]);
    }

    // tensorCopy.info();
    // tensor.info();
}

test "to array " {
    std.debug.print("\n     test:to array ", .{});

    const allocator = std.heap.page_allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();
    std.debug.print("Tensor 1 Info:\n", .{});
    tensor.info();
    std.debug.print("\nTesting toArray on Tensor 1:\n", .{});
    const array_from_tensor = tensor.toArray(2);
    std.debug.print("Array extracted from tensor: {any}\n", .{array_from_tensor});
}

test "Reshape" {
    std.debug.print("\n     test: Reshape ", .{});
    const allocator = std.heap.page_allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();
    std.debug.print("Tensor 1 Info:\n", .{});
    tensor.info();
    var new_shape: [2]usize = [_]usize{ 3, 2 };
    try tensor.reshape(&new_shape);
    std.debug.print("Tensor 1 Info after reshape:\n", .{});
    tensor.info();
}

test "transpose" {
    std.debug.print("\n     test: transpose ", .{});
    const allocator = std.heap.page_allocator;

    // Inizializzazione degli array di input
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    var tensor_transposed = try tensor.transpose2D();
    defer tensor_transposed.deinit();
    tensor.info();
    tensor_transposed.info();

    try std.testing.expect(tensor_transposed.data[0] == 1);
    try std.testing.expect(tensor_transposed.data[1] == 4);
    try std.testing.expect(tensor_transposed.data[2] == 2);
    try std.testing.expect(tensor_transposed.data[3] == 5);
    try std.testing.expect(tensor_transposed.data[4] == 3);
    try std.testing.expect(tensor_transposed.data[5] == 6);
}
