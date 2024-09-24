const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;
const TensMath = @import("./tensor_math.zig");
const Architectures = @import("./architectures.zig").Architectures;
const TensorMathError = @import("./tensor_math.zig").TensorMathError;
const ArchitectureError = @import("./tensor_math.zig").ArchitectureError;

test "Sum two tensors on CPU architecture" {
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t2.deinit();

    var t3 = try TensMath.sum_tensors(Architectures.CPU, f32, f64, &t1, &t2); // Output tensor with larger type
    defer t3.deinit();

    // Check if the values in t3 are as expected
    try std.testing.expect(2.0 == t3.data[0]);
    try std.testing.expect(4.0 == t3.data[1]);
}

test "Error when input tensors have different sizes" {
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };
    var inputArray2: [3][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
        [_]f32{ 14.0, 15.0 },
    };

    var shape1 = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2 = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape1);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray2, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorDifferentSize, TensMath.sum_tensors(Architectures.CPU, f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "Dot product 2x2" {
    const allocator = std.heap.page_allocator;

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    var t2 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);

    var result_tensor = try TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &t1, &t2);

    try std.testing.expect(9.0 == result_tensor.data[0]);
    try std.testing.expect(12.0 == result_tensor.data[1]);

    result_tensor.deinit();
    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible sizes for dot product" {
    const allocator = std.heap.page_allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorDifferentSize, TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible shapes for dot product" {
    const allocator = std.heap.page_allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 4x1 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "GPU architecture under development error" {
    const allocator = std.heap.page_allocator;

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape);
    var t3 = try Tensor(f64).fromShape(&allocator, &shape);

    try std.testing.expectError(ArchitectureError.UnderDevelopementArchitecture, TensMath.sum_tensors(Architectures.GPU, f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
    t3.deinit();
}

test "add bias" {
    const allocator = std.heap.page_allocator;

    var shape_tensor: [2]usize = [_]usize{ 2, 3 }; // 2x3 matrix
    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };

    var bias_array: [3]f32 = [_]f32{ 1.0, 1.0, 1.0 };
    var shape_bias: [1]usize = [_]usize{3};

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &shape_bias);

    try TensMath.add_bias(f32, &t1, &bias);

    t1.info();
}
