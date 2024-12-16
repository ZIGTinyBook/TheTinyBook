const std = @import("std");
const Tensor = @import("tensor").Tensor;
const TensMath = @import("tensor_m");
const Architectures = @import("architectures").Architectures;
const TensorMathError = @import("errorHandler").TensorMathError;
const ArchitectureError = @import("errorHandler").ArchitectureError;
const ErrorHandler = @import("errorHandler");
const PoolingType = @import("poolingLayer").PoolingType;

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

test "Convolution 4D Input with 2x2 Kernel" {
    std.debug.print("\n     test: Convolution 4D Input with 2x2 Kernel\n", .{});

    const allocator = std.testing.allocator;

    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var kernel_shape: [4]usize = [_]usize{ 1, 2, 2, 2 };

    var inputArray: [2][2][3][3]f32 = [_][2][3][3]f32{
        //Firsr Batch
        [_][3][3]f32{
            // First Channel
            [_][3]f32{
                [_]f32{ 2.0, 2.0, 3.0 },
                [_]f32{ 4.0, 5.0, 6.0 },
                [_]f32{ 7.0, 8.0, 9.0 },
            },
            // Second Channel
            [_][3]f32{
                [_]f32{ 8.0, 8.0, 7.0 },
                [_]f32{ 6.0, 5.0, 4.0 },
                [_]f32{ 3.0, 2.0, 1.0 },
            },
        },
        // Second batch
        [_][3][3]f32{
            // First channel
            [_][3]f32{
                [_]f32{ 2.0, 3.0, 4.0 },
                [_]f32{ 5.0, 6.0, 7.0 },
                [_]f32{ 8.0, 9.0, 10.0 },
            },
            // Second channel
            [_][3]f32{
                [_]f32{ 10.0, 9.0, 8.0 },
                [_]f32{ 7.0, 6.0, 5.0 },
                [_]f32{ 4.0, 3.0, 2.0 },
            },
        },
    };

    // Kernel tensor
    var kernelArray: [1][2][2][2]f32 = [_][2][2][2]f32{
        [_][2][2]f32{
            [_][2]f32{
                [_]f32{ -1.0, 0.0 },
                [_]f32{ 0.0, 1.0 },
            },
            [_][2]f32{
                [_]f32{ 1.0, -1.0 },
                [_]f32{ -1.0, 1.0 },
            },
        },
    };

    var inputbias: [2][1]f32 = [_][1]f32{
        [_]f32{1},
        [_]f32{
            1,
        },
    };
    var shape: [2]usize = [_]usize{ 2, 1 };
    var bias = try Tensor(f32).fromArray(&allocator, &inputbias, &shape);

    var input_tensor = try Tensor(f32).fromArray(&allocator, &inputArray, &input_shape);
    var kernel_tensor = try Tensor(f32).fromArray(&allocator, &kernelArray, &kernel_shape);

    var result_tensor = try TensMath.CPU_convolve_tensors_with_bias(f32, f32, &input_tensor, &kernel_tensor, &bias);

    // Expected results with the correct dimensions
    const expected_result: [2][1][2][2]f32 = [_][1][2][2]f32{
        // Primo batch
        [_][2][2]f32{
            [_][2]f32{ [_]f32{ 3.0, 5.0 }, [_]f32{ 5.0, 5.0 } },
        },
        // Secondo batch
        [_][2][2]f32{
            [_][2]f32{ [_]f32{ 5.0, 5.0 }, [_]f32{ 5.0, 5.0 } },
        },
    };
    result_tensor.info();

    for (0..2) |batch| {
        for (0..1) |filter| {
            for (0..2) |i| {
                for (0..2) |j| {
                    const idx = batch * (1 * 2 * 2) + filter * (2 * 2) + i * 2 + j;
                    try std.testing.expectEqual(expected_result[batch][filter][i][j], result_tensor.data[idx]);
                }
            }
        }
    }

    result_tensor.deinit();
    input_tensor.deinit();
    kernel_tensor.deinit();
    bias.deinit();
}

test "Pooling 2D " {
    std.debug.print("\n     test: Pooling 2D\n", .{});

    const allocator = std.testing.allocator;

    // ------------
    var shape_tensor: [2]usize = [_]usize{ 3, 3 }; // 3x3 matrix
    var inputArray: [3][3]f32 = [_][3]f32{
        [_]f32{ 1.0, 2.0, 3.0 },
        [_]f32{ 4.0, 5.0, 6.0 },
        [_]f32{ 40.0, 50.0, 60.0 },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    var used_input1 = try Tensor(u1).fromShape(&allocator, &shape_tensor);
    defer used_input1.deinit();

    var output1: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output1.deinit();

    // input1.info();
    // output.info();
    // used_input1.info();

    try std.testing.expectEqual(output1.shape.len, input1.shape.len);
    try std.testing.expectEqual(output1.shape[0], 2);
    try std.testing.expectEqual(output1.shape[1], 2);

    try std.testing.expectEqual(output1.data[0], 5);
    try std.testing.expectEqual(output1.data[1], 6);
    try std.testing.expectEqual(output1.data[2], 50);
    try std.testing.expectEqual(output1.data[3], 60);

    try std.testing.expectEqual(used_input1.data[0], 0);
    try std.testing.expectEqual(used_input1.data[1], 0);
    try std.testing.expectEqual(used_input1.data[2], 0);
    try std.testing.expectEqual(used_input1.data[3], 0);
    try std.testing.expectEqual(used_input1.data[4], 1);
    try std.testing.expectEqual(used_input1.data[5], 1);
    try std.testing.expectEqual(used_input1.data[6], 0);
    try std.testing.expectEqual(used_input1.data[7], 1);
    try std.testing.expectEqual(used_input1.data[8], 1);

    var kernel2: [2]usize = [2]usize{ 2, 2 };
    var stride2: [2]usize = [2]usize{ 2, 2 };

    var used_input2 = try Tensor(u1).fromShape(&allocator, &shape_tensor);
    defer used_input2.deinit();

    var output2: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input2, &kernel2, &stride2, PoolingType.Max);
    defer output2.deinit();

    // input1.info();
    // output.info();
    // used_input1.info();

    try std.testing.expectEqual(output2.shape.len, input1.shape.len);
    try std.testing.expectEqual(output2.shape[0], 1);
    try std.testing.expectEqual(output2.shape[1], 1);
    try std.testing.expectEqual(output2.data[0], 5);
}

test "Pooling multidim" {
    std.debug.print("\n     test: Pooling multidim\n", .{});

    const allocator = std.testing.allocator;

    // ------------
    var shape_tensor: [3]usize = [_]usize{ 3, 3, 3 };
    var inputArray: [3][3][3]f32 = [_][3][3]f32{
        [_][3]f32{
            [_]f32{ 1.0, 2.0, 3.0 },
            [_]f32{ 4.0, 5.0, 6.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ 10.0, 20.0, 30.0 },
            [_]f32{ 40.0, 0.0, -10.0 },
            [_]f32{ 40.0, 50.0, 60.0 },
        },
        [_][3]f32{
            [_]f32{ -1.0, -2.0, -3.0 },
            [_]f32{ -4.0, -5.0, -6.0 },
            [_]f32{ -40.0, -50.0, -60.0 },
        },
    };

    var input1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape_tensor);
    defer input1.deinit();
    var kernel1: [2]usize = [2]usize{ 2, 2 };
    var stride1: [2]usize = [2]usize{ 1, 1 };

    var used_input1 = try Tensor(u1).fromShape(&allocator, &shape_tensor);
    defer used_input1.deinit();

    var output: Tensor(f32) = try TensMath.pool_tensor(f32, &input1, &used_input1, &kernel1, &stride1, PoolingType.Max);
    defer output.deinit();

    try std.testing.expectEqual(output.shape.len, input1.shape.len);
    try std.testing.expectEqual(output.shape[0], 3);
    try std.testing.expectEqual(output.shape[1], 2);
    try std.testing.expectEqual(output.shape[2], 2);
}
