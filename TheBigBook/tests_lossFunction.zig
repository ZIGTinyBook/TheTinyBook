const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;
const Loss = @import("./lossFunction.zig");
const MSELoss = @import("./lossFunction.zig").MSELoss;
const CCELoss = @import("./lossFunction.zig").CCELoss;
const LossError = @import("./lossFunction.zig").LossError;

test "tests description" {
    std.debug.print("\n--- Running loss_function tests\n", .{});
}

// INTERFACE LOSS FUNCTION TESTS--------------------------------------------------------------------------------------------------

test " Loss Function MSE using Interface, target==predictor" {
    std.debug.print("\n     test: Loss Function MSE using Interface, target==predictor ", .{});
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
    var mse = Loss.LossFunction(Loss.MSELoss){};
    var loss: Tensor(f32) = try mse.computeLoss(f32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();
    //loss.info();

    for (0..loss.size) |i| {
        try std.testing.expect(0.0 == loss.data[i]);
    }
}

test " Loss Function CCE using Interface, target==predictor" {
    std.debug.print("\n     test: Loss Function CCE using Interface, target==predictor", .{});
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

    var cce = Loss.LossFunction(Loss.CCELoss){};
    var loss: Tensor(f32) = try cce.computeLoss(f32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();
}

// LOSS FUNCTION TESTS--------------------------------------------------------------------------------------------------

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
    var mse = Loss.LossFunction(Loss.MSELoss){};
    var loss: Tensor(f32) = try mse.computeLoss(f32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();

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
    var mse = Loss.LossFunction(Loss.MSELoss){};
    var loss: Tensor(u32) = try mse.computeLoss(u32, &t2_PREDICTION, &t1_TARGET);
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
    var mse = Loss.LossFunction(Loss.MSELoss){};
    var loss: Tensor(i32) = try mse.computeLoss(i32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();

    for (0..loss.size) |i| {
        try std.testing.expect(0.0 != loss.data[i]);
    }
}

test " CCE target==predictor, 2 x 2" {
    std.debug.print("\n     test:CCE target==predictor, 2 x 2", .{});
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

    var cce = Loss.LossFunction(Loss.CCELoss){};
    var loss: Tensor(f32) = try cce.computeLoss(f32, &t2_PREDICTION, &t1_TARGET);
    defer loss.deinit();
    //loss.info();
}

// GRADIENT TESTS-------------------------------------------------------------------------------------------------------

test " GRADIENT MSE target==predictor, 2 x 2" {
    std.debug.print("\n     test: GRADIENT MSE target==predictor, 2 x 2 ", .{});
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
    var t3_GRAD = try Tensor(f64).fromShape(&allocator, &shape);
    defer t3_GRAD.deinit();

    //gradient SHOULD RESULT ALL ZEROS
    var mse = Loss.LossFunction(Loss.MSELoss){};
    try mse.computeGradient(f32, &t2_PREDICTION, &t1_TARGET, &t3_GRAD);
}

test " GRADIENT MSE error on shape (dimensions)" {
    std.debug.print("\n     test: GRADIENT MSE error on shape (dimensions)", .{});
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape1);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape2);
    defer t2_PREDICTION.deinit();
    var t3_GRAD = try Tensor(f64).fromShape(&allocator, &shape1);
    defer t3_GRAD.deinit();

    var mse = Loss.LossFunction(Loss.MSELoss){};

    //gradient SHOULD RESULT ALL ZEROS
    try std.testing.expectError(LossError.ShapeMismatch, mse.computeGradient(f32, &t2_PREDICTION, &t1_TARGET, &t3_GRAD));
}

test " GRADIENT MSE error on shape (len)" {
    std.debug.print("\n     test: GRADIENT MSE error on shape (len)", .{});
    const allocator = std.heap.page_allocator;

    var shape1: [2]usize = [_]usize{ 2, 4 }; // 2x2 matrix
    var shape2: [3]usize = [_]usize{ 2, 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromShape(&allocator, &shape1);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromShape(&allocator, &shape2);
    defer t2_PREDICTION.deinit();
    var t3_GRAD = try Tensor(f64).fromShape(&allocator, &shape1);
    defer t3_GRAD.deinit();

    var mse = Loss.LossFunction(Loss.MSELoss){};
    //gradient SHOULD RESULT ALL ZEROS
    try std.testing.expectError(LossError.ShapeMismatch, mse.computeGradient(f32, &t2_PREDICTION, &t1_TARGET, &t3_GRAD));
}

test " GRADIENT CCE error on shape (dimensions)" {
    std.debug.print("\n     test: GRADIENT CCE error on shape (dimensions)", .{});
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromArray(&allocator, &inputArray, &shape1);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromArray(&allocator, &inputArray, &shape2);
    defer t2_PREDICTION.deinit();
    var t3_GRAD = try Tensor(f64).fromShape(&allocator, &shape1);
    defer t3_GRAD.deinit();

    var cce = Loss.LossFunction(Loss.CCELoss){};

    //gradient SHOULD RESULT ALL ZEROS
    try std.testing.expectError(LossError.ShapeMismatch, cce.computeGradient(f32, &t2_PREDICTION, &t1_TARGET, &t3_GRAD));
}

test " GRADIENT CCE error on shape (len)" {
    std.debug.print("\n     test: GRADIENT CCE error on shape (len)", .{});
    const allocator = std.heap.page_allocator;

    var shape1: [2]usize = [_]usize{ 2, 4 }; // 2x2 matrix
    var shape2: [3]usize = [_]usize{ 2, 2, 2 }; // 2x2 matrix

    var t1_TARGET = try Tensor(f32).fromShape(&allocator, &shape1);
    defer t1_TARGET.deinit();
    var t2_PREDICTION = try Tensor(f32).fromShape(&allocator, &shape2);
    defer t2_PREDICTION.deinit();
    var t3_GRAD = try Tensor(f64).fromShape(&allocator, &shape1);
    defer t3_GRAD.deinit();

    var cce = Loss.LossFunction(Loss.CCELoss){};

    //gradient SHOULD RESULT ALL ZEROS
    try std.testing.expectError(LossError.ShapeMismatch, cce.computeGradient(f32, &t2_PREDICTION, &t1_TARGET, &t3_GRAD));
}
