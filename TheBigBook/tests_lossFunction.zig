const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;
const MSELoss = @import("./lossFunction.zig").MSELoss;
const CCELoss = @import("./lossFunction.zig").CCELoss;

test "tests description" {
    std.debug.print("\n--- Running loss_function tests\n", .{});
}

test " MSE target==predictor, 2 x 2" {

    std.debug.print("\n     test: MSE target==predictor, 2 x 2 ", .{});
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2_PREDICTION.deinit();

    //LOSS SHOULD RESULT ALL ZEROS
    var loss: Tensor(f32) = try MSELoss.lossFn(f32, &t2_PREDICTION, &t1_TARGET);

    defer loss.deinit();
    //loss.info();

    for (0..loss.size) |i| {
        try std.testing.expect(0.0 == loss.data[i]);
    }
}

test " MSE target==predictor, 2 x 3 X 2" {
    std.debug.print("\n     test: MSE target==predictor, 2 x 3 X 2 ", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var inputArray: [2][3][2]u32 = [_][3][2]u32{
        [_][2]u32{
            [_]u32{ 1, 2 },
            [_]u32{ 4, 5 },
            [_]u32{ 6, 7 },
        },
        [_][2]u32{
            [_]u32{ 10, 20 },
            [_]u32{ 40, 50 },
            [_]u32{ 60, 70 },
        },
    };

    var shape: [3]usize = [_]usize{ 2, 3, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(u32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(u32).fromArray(&allocator, &inputArray, &shape);
    defer t2_PREDICTION.deinit();

    //std.debug.print("\n creating a new MSE loss function", .{});

    //LOSS SHOULD RESULT ALL ZEROS
    var loss: Tensor(u32) = try MSELoss.lossFn(u32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();

    for (0..loss.size) |i| {
        try std.testing.expect(0.0 == loss.data[i]);
    }
}

test " MSE target!=predictor, 2 x 3 X 2" {
    std.debug.print("\n     test: MSE target!=predictor, 2 x 3 X 2", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var inputArray: [2][3][2]i32 = [_][3][2]i32{
        [_][2]i32{
            [_]i32{ 1, 2 },
            [_]i32{ 4, 5 },
            [_]i32{ 6, 7 },
        },
        [_][2]i32{
            [_]i32{ 10, 20 },
            [_]i32{ 40, 50 },
            [_]i32{ 60, 70 },
        },
    };

    var inputArray2: [2][3][2]i32 = [_][3][2]i32{
        [_][2]i32{
            [_]i32{ 10, 20 },
            [_]i32{ 40, 50 },
            [_]i32{ 60, 70 },
        },
        [_][2]i32{
            [_]i32{ 1, 2 },
            [_]i32{ 4, 5 },
            [_]i32{ 6, 7 },
        },
    };

    var shape: [3]usize = [_]usize{ 2, 3, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(i32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(i32).fromArray(&allocator, &inputArray2, &shape);
    defer t2_PREDICTION.deinit();

    //std.debug.print("\n creating a new MSE loss function", .{});

    //LOSS SHOULD RESULT ALL ZEROS
    var loss = try MSELoss.lossFn(i32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();

    for (0..loss.size) |i| {
        try std.testing.expect(0.0 != loss.data[i]);
    }
}

test " CCE target==predictor, 2 x 2" {
    std.debug.print("\n     test:CCE target==predictor, 2 x 2 \n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2_PREDICTION.deinit();

    var loss: Tensor(f32) = try CCELoss.lossFn(f32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();
    //loss.info();
}
