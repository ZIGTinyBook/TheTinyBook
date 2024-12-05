const std = @import("std");
const Tensor = @import("tensor").Tensor;
const TensMath = @import("tensor_m");
const Architectures = @import("architectures").Architectures;
const TensorMathError = @import("errorHandler").TensorMathError;
const ArchitectureError = @import("errorHandler").ArchitectureError;
const ErrorHandler = @import("errorHandler");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

test "Sum two tensors on CPU architecture" {
    std.debug.print("\n     test: Sum two tensors on CPU architecture", .{});
    const allocator = std.testing.allocator;

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
    std.debug.print("\n     test: Error when input tensors have different sizes", .{});
    const allocator = std.testing.allocator;

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
    std.debug.print("\n     test:Dot product 2x2", .{});

    const allocator = std.testing.allocator;

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
    const allocator = std.testing.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 3, 2 }; // 3x2 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &t1, &t2));

    _ = TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &t1, &t2) catch |err| {
        std.debug.print("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
    };
    t1.deinit();
    t2.deinit();
}

test "Error when input tensors have incompatible shapes for dot product" {
    std.debug.print("\n     test: Error when input tensors have incompatible shapes for dot product", .{});
    const allocator = std.testing.allocator;

    var shape1: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix
    var shape2: [2]usize = [_]usize{ 4, 1 }; // 4x1 matrix
    var t1 = try Tensor(f32).fromShape(&allocator, &shape1);
    var t2 = try Tensor(f32).fromShape(&allocator, &shape2);

    try std.testing.expectError(TensorMathError.InputTensorsWrongShape, TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &t1, &t2));

    t1.deinit();
    t2.deinit();
}

test "GPU architecture under development error" {
    std.debug.print("\n     test: GPU architecture under development error\n", .{});
    const allocator = std.testing.allocator;

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
    std.debug.print("\n     test:add bias", .{});
    const allocator = std.testing.allocator;

    var shape_tensor: [2]usize = [_]usize{ 2, 3 }; // 2x3 matrix
    var inputArray: [2][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
    };
    const flatArr: [6]f32 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    var shape_bias: [1]usize = [_]usize{3};
    var bias_array: [3]f32 = [_]f32{ 1.0, 1.0, 1.0 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    var bias = try Tensor(f32).fromArray(&allocator, &bias_array, &shape_bias);

    try TensMath.add_bias(f32, &t1, &bias);

    for (t1.data, 0..) |*data, i| {
        try std.testing.expect(data.* == flatArr[i] + 1);
    }

    t1.deinit();
    bias.deinit();
}

test "mean" {
    std.debug.print("\n     test:mean", .{});
    const allocator = std.testing.allocator;

    var shape_tensor: [1]usize = [_]usize{3}; // 2x3 matrix
    var inputArray: [3]f32 = [_]f32{ 1.0, 2.0, 3.0 };

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);

    try std.testing.expect(2.0 == TensMath.mean(f32, &t1));

    t1.deinit();
}

test "Convolution 3x3 Input with 2x2 Kernel" {
    std.debug.print("\n     test: Convolution 3x3 Input with 2x2 Kernel", .{});

    const allocator = std.testing.allocator;

    var input_shape: [2]usize = [_]usize{ 3, 3 }; // 3x3 input tensor
    var kernel_shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 kernel tensor

    var inputArray: [3][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
        [_]f32{ 7.0, 8.0, 9.0 },
    };

    var kernelArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -1.0, 0.0 },
        [_]f32{ 0.0, 1.0 },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);

    var result_tensor = try TensMath.convolve_tensor(Architectures.CPU, f32, f32, &input_tensor, &kernel_tensor);

    var expected_result: [2][2]f32 = [_][2]f32{
        [_]f32{ 0.0, 0.0 },
        [_]f32{ 0.0, 0.0 },
    };

    // (1*-1) + (2*0) + (4*0) + (5*1) = -1 + 0 + 0 + 5 = 4
    expected_result[0][0] = 4.0;

    // (2*-1) + (3*0) + (5*0) + (6*1) = -2 + 0 + 0 + 6 = 4
    expected_result[0][1] = 4.0;

    // (4*-1) + (5*0) + (7*0) + (8*1) = -4 + 0 + 0 + 8 = 4
    expected_result[1][0] = 4.0;

    // (5*-1) + (6*0) + (8*0) + (9*1) = -5 + 0 + 0 + 9 = 4
    expected_result[1][1] = 4.0;

    for (0..2) |i| {
        for (0..2) |j| {
            const idx = i * 2 + j;
            try std.testing.expectEqual(expected_result[i][j], result_tensor.data[idx]);
        }
    }

    result_tensor.deinit();
    input_tensor.deinit();
    kernel_tensor.deinit();
}

test "Convolution 5x5 Input with 3x3 Kernel" {
    std.debug.print("\n     test: Convolution 5x5 Input with 3x3 Kernel", .{});

    const allocator = std.testing.allocator;

    var input_shape: [2]usize = [_]usize{ 5, 5 };
    var kernel_shape: [2]usize = [_]usize{ 3, 3 };

    var inputArray: [5][5]f32 = [_][5]f32{
        [_]f32{ 1, 2, 3, 4, 5 },
        [_]f32{ 6, 7, 8, 9, 10 },
        [_]f32{ 11, 12, 13, 14, 15 },
        [_]f32{ 16, 17, 18, 19, 20 },
        [_]f32{ 21, 22, 23, 24, 25 },
    };

    var kernelArray: [3][3]f32 = [_][3]f32{
        [_]f32{ 1, 0, -1 },
        [_]f32{ 1, 0, -1 },
        [_]f32{ 1, 0, -1 },
    };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);

    var result_tensor = try TensMath.convolve_tensor(Architectures.CPU, f32, f32, &input_tensor, &kernel_tensor);

    // Expected result is a 3x3 matrix
    var expected_result: [3][3]f32 = [_][3]f32{
        [_]f32{ 0.0, 0.0, 0.0 },
        [_]f32{ 0.0, 0.0, 0.0 },
        [_]f32{ 0.0, 0.0, 0.0 },
    };

    // CManually calculate the expected result
    for (0..3) |i| {
        for (0..3) |j| {
            var sum: f32 = 0.0;
            for (0..3) |m| {
                for (0..3) |n| {
                    const input_value = inputArray[i + m][j + n];
                    const kernel_value = kernelArray[m][n];
                    sum += input_value * kernel_value;
                }
            }
            expected_result[i][j] = sum;
        }
    }

    std.debug.print("\nCalcolo del risultato atteso:\n", .{});
    for (0..3) |i| {
        for (0..3) |j| {
            std.debug.print("expected_result[{}][{}] = {}\n", .{ i, j, expected_result[i][j] });
        }
    }

    for (0..3) |i| {
        for (0..3) |j| {
            const idx = i * 3 + j;
            try std.testing.expectEqual(expected_result[i][j], result_tensor.data[idx]);
        }
    }

    // Deinit tensors
    result_tensor.deinit();
    input_tensor.deinit();
    kernel_tensor.deinit();
}
test "Convolution with Bias 3x3 Input with 2x2 Kernel" {
    std.debug.print("\nTest: Convolution with Bias 3x3 Input with 2x2 Kernel\n", .{});

    const allocator = std.testing.allocator;

    var input_shape: [2]usize = [_]usize{ 3, 3 }; // 3x3 input tensor
    var kernel_shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 kernel tensor

    var inputArray: [3][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
        [_]f32{ 7.0, 8.0, 9.0 },
    };

    var kernelArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -1.0, 0.0 },
        [_]f32{ 0.0, 1.0 },
    };

    const bias: f32 = 1.0;

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();

    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();

    var result_tensor = try TensMath.CPU_convolve_tensors_with_bias(f32, f32, &input_tensor, &kernel_tensor, bias);
    defer result_tensor.deinit();

    // Expected result after convolution and adding bias
    const expected_result: [2][2]f32 = [_][2]f32{
        [_]f32{ 5.0, 5.0 },
        [_]f32{ 5.0, 5.0 },
    };

    // Verification of the results
    for (0..2) |i| {
        for (0..2) |j| {
            const idx = i * 2 + j;
            try std.testing.expectEqual(expected_result[i][j], result_tensor.data[idx]);
        }
    }
}
