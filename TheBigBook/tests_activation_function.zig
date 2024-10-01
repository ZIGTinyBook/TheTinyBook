const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;
const ActivFun = @import("./activation_function.zig");
const ReLU = @import("./activation_function.zig").ReLU;
const Softmax = @import("./activation_function.zig").Softmax;

// initialization from activationFunction()------------------------------------------------
test "ReLU from ActivationFunction()" {
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -1.0, -2.0 },
        [_]f32{ -4.0, -5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var relu = ActivFun.ActivationFunction(ActivFun.ReLU){};

    try relu.forward(f32, &t1);

    for (t1.data) |*val| {
        try std.testing.expect(0.0 == val.*);
    }
}

test "Softmax from ActivationFunction()" {
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 0, 0 },
        [_]f32{ 0, 0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    var soft = ActivFun.ActivationFunction(ActivFun.Softmax){};
    try soft.forward(f32, &t1);
    //now data is:
    //{ 0.2689414,  0.7310586  }
    //{ 0.2689414,  0.73105854 }

    try std.testing.expect(t1.data[0] == t1.data[1]);
    try std.testing.expect(t1.data[2] == t1.data[3]);
}

test "ReLU all negative" {
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -1.0, -2.0 },
        [_]f32{ -4.0, -5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try ReLU.forward(f32, 0, &t1);

    for (t1.data) |*val| {
        try std.testing.expect(0.0 == val.*);
    }
}

test "ReLU all positive" {
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try ReLU.forward(f32, 0, &t1);

    for (t1.data) |*val| {
        try std.testing.expect(val.* >= 0);
    }
}

test "Softmax all positive" {
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try Softmax.forward(f32, &t1);
    //now data is:
    //{ 0.2689414,  0.7310586  }
    //{ 0.2689414,  0.73105854 }

    try std.testing.expect(t1.data[0] + t1.data[1] > 0.9);
    try std.testing.expect(t1.data[0] + t1.data[1] < 1.1);

    try std.testing.expect(t1.data[2] + t1.data[3] > 0.9);
    try std.testing.expect(t1.data[2] + t1.data[3] < 1.1);
}

test "Softmax all 0" {
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 0, 0 },
        [_]f32{ 0, 0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    try Softmax.forward(f32, &t1);
    //now data is:
    //{ 0.2689414,  0.7310586  }
    //{ 0.2689414,  0.73105854 }

    try std.testing.expect(t1.data[0] == t1.data[1]);
    try std.testing.expect(t1.data[2] == t1.data[3]);
}
