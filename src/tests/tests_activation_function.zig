const std = @import("std");
const Tensor = @import("tensor").Tensor;
const ActivFun = @import("activation_function");
const ActivType = @import("activation_function").ActivationType;

test "tests description" {
    std.debug.print("\n--- Running activation_function tests\n", .{});
}

//*********************************************** ReLU ***********************************************
test "ReLU from ActivationFunction()" {
    std.debug.print("\n     test: ReLU from ActivationFunction()", .{});

    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -1.0, -2.0 },
        [_]f32{ -4.0, -5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    const act_type = ActivFun.ActivationFunction(f32, ActivType.ReLU);
    var relu = act_type{};

    try relu.forward(&t1);

    for (t1.data) |*val| {
        try std.testing.expect(0.0 == val.*);
    }
}

test "ReLU all negative" {
    std.debug.print("\n     test: ReLU all negative", .{});

    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -1.0, -2.0 },
        [_]f32{ -4.0, -5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    const act_type = ActivFun.ActivationFunction(f32, ActivType.ReLU);
    var relu = act_type{};

    try relu.forward(&t1);

    for (t1.data) |*val| {
        try std.testing.expect(0.0 == val.*);
    }
}

test "ReLU all positive" {
    std.debug.print("\n     test: ReLU all positive", .{});

    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    const relu_type = ActivFun.ActivationFunction(f32, ActivType.ReLU);
    var relu = relu_type{};
    try relu.forward(&t1);

    for (t1.data) |*val| {
        try std.testing.expect(val.* >= 0);
    }
}

//*********************************************** Softmax ***********************************************

test "Softmax from ActivationFunction()" {
    std.debug.print("\n     test: Softmax from ActivationFunction()", .{});
    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 0, 0 },
        [_]f32{ 0, 0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    const soft_type = ActivFun.ActivationFunction(f32, ActivType.Softmax);
    var soft = soft_type{};
    t1.info();
    //std.debug.print("\n \n soft:{any}", .{soft});

    try soft.forward(&t1);
    //now data is:forward
    //{ 0.2689414,  0.7310586  }
    //{ 0.2689414,  0.73105854 }

    try std.testing.expect(t1.data[0] == t1.data[1]);
    try std.testing.expect(t1.data[2] == t1.data[3]);
}

test "Softmax all positive" {
    std.debug.print("\n     test: Softmax all positive", .{});

    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    const soft_type = ActivFun.ActivationFunction(f32, ActivType.Softmax);
    var soft = soft_type{};
    try soft.forward(&t1);
    //now data is:
    //{ 0.2689414,  0.7310586  }
    //{ 0.2689414,  0.73105854 }

    try std.testing.expect(t1.data[0] + t1.data[1] > 0.9);
    try std.testing.expect(t1.data[0] + t1.data[1] < 1.1);

    try std.testing.expect(t1.data[2] + t1.data[3] > 0.9);
    try std.testing.expect(t1.data[2] + t1.data[3] < 1.1);
}

test "Softmax all 0" {
    std.debug.print("\n     test: Softmax all 0", .{});

    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 0, 0 },
        [_]f32{ 0, 0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    const soft_type = ActivFun.ActivationFunction(f32, ActivType.Softmax);
    var soft = soft_type{};
    try soft.forward(&t1);

    t1.info();

    try std.testing.expect(t1.data[0] == t1.data[1]);
    try std.testing.expect(t1.data[2] == t1.data[3]);
}

test "Softmax derivate" {
    std.debug.print("\n     test: Softmax derivate", .{});

    const allocator = std.heap.page_allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ 1.0, 2.0 },
        [_]f32{ 4.0, 5.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 2 }; // 2x2 matrix

    var t1 = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer t1.deinit();

    const soft_type = ActivFun.ActivationFunction(f32, ActivType.Softmax);
    var soft = soft_type{};
    try soft.forward(&t1);
    //now data is:
    //{ 0.2689414,  0.7310586  }
    //{ 0.2689414,  0.73105854 }
    //t1.info();

    try std.testing.expect(t1.data[0] + t1.data[1] > 0.9);
    try std.testing.expect(t1.data[0] + t1.data[1] < 1.1);

    try std.testing.expect(t1.data[2] + t1.data[3] > 0.9);
    try std.testing.expect(t1.data[2] + t1.data[3] < 1.1);

    try soft.derivate(&t1);
    //now data is:
    //{0.196, −0.196}
    //{​−0.196, 0.196​}
    t1.info();
}

//*********************************************** Sigmoid ***********************************************

test "Sigmoid forward" {
    std.debug.print("\n     test: Sigmoid forward ", .{});

    const allocator = std.heap.page_allocator;

    const input_data = [_]f64{ 0.0, 2.0, -2.0 }; // input data for the tensor
    var shape: [1]usize = [_]usize{3};
    var input_tensor = try Tensor(f64).fromArray(&allocator, &input_data, &shape); // create tensor from input data
    defer input_tensor.deinit();

    const act_type = ActivFun.ActivationFunction(f64, ActivType.Sigmoid);
    var sigmoid = act_type{};
    // Test forward pass
    try sigmoid.forward(&input_tensor);
    const expected_forward_output = [_]f64{ 0.5, 0.880797, 0.119203 }; // expected sigmoid output for each input value
    for (input_tensor.data, 0..) |*data, i| {
        try std.testing.expect(@abs(data.* - expected_forward_output[i]) < 1e-6);
    }
}

test "Sigmoid derivate" {
    std.debug.print("\n     test: Sigmoid derivate ", .{});

    const allocator = std.heap.page_allocator;

    const input_data = [_]f64{ 0.0, 2.0, -2.0 }; // input data for the tensor
    var shape: [1]usize = [_]usize{3};
    var input_tensor = try Tensor(f64).fromArray(&allocator, &input_data, &shape); // create tensor from input data
    defer input_tensor.deinit();

    const act_type = ActivFun.ActivationFunction(f64, ActivType.Sigmoid);
    var sigmoid = act_type{};

    // Test forward pass
    try sigmoid.forward(&input_tensor);
    const expected_forward_output = [_]f64{ 0.5, 0.880797, 0.119203 }; // expected sigmoid output for each input value
    for (input_tensor.data, 0..) |*data, i| {
        try std.testing.expect(@abs(data.* - expected_forward_output[i]) < 1e-6);
    }
    // Test derivative
    try sigmoid.derivate(&input_tensor);
    const expected_derivative_output = [_]f64{ 0.25, 0.104994, 0.104994 }; // expected derivative output for each input
    for (input_tensor.data, 0..) |*data, i| {
        try std.testing.expect(@abs(data.* - expected_derivative_output[i]) < 1e-6);
    }
}
