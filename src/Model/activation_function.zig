const std = @import("std");
const Tensor = @import("tensor").Tensor;
const TensorError = @import("tensor").TensorError;

pub const ActivationType = enum {
    ReLU,
    Sigmoid,
    Softmax,
    None,
};

// activation function Interface
pub fn ActivationFunction(comptime T: anytype, activationType: ActivationType) type {
    //const act =
    return switch (activationType) {
        ActivationType.ReLU => ReLU(T),
        ActivationType.Sigmoid => Sigmoid(T),
        ActivationType.Softmax => Softmax(T),
        ActivationType.None => None(),
    };

    //LET THIS COMMENTED, COUL BE USEFULL IN FUTURE
    //const act = activationFunctionStruct(){};
    // return switch (activationType) {
    //     ActivationType.Softmax => struct {
    //         pub fn forward(self: *act, input: *Tensor(T)) !void {
    //             std.debug.print("\n 1111AAAAAAAAAAAAAAAAAAAAAAAA", .{});

    //             std.debug.print("\n {} \n soft:{any}", .{ @typeInfo(self.*), self });

    //             try act.forward(self, input);
    //             //act.softmax_output = try input.copy();
    //         }

    //         pub fn derivate(dVal: *Tensor(T), act_forward_out: *Tensor(T)) !void {
    //             try act.derivateWithPrevOutput(T, dVal, act_forward_out);
    //         }
    //     },
    //     else => struct {
    //         pub fn forward(input: *Tensor(T)) !void {
    //             try act.forward(T, input);
    //         }

    //         pub fn derivate(input: *Tensor(T)) !void {
    //             try act.derivate(T, input);
    //         }
    //     },
    // };
}
pub fn None() type {}

pub fn ReLU(comptime T: anytype) type {
    return struct {
        const Self = @This();
        //it directly modify the input tensor
        //threshold is usually set to zero
        pub fn forward(self: *Self, input: *Tensor(T)) !void {
            _ = self;

            //checks
            if (input.size <= 0) return TensorError.ZeroSizeTensor;

            //apply ReLU
            //OSS: can be improved, see how did I parallelized CPU Tensor Sum
            for (0..input.size) |i| {
                if (input.data[i] <= 0) input.data[i] = 0;
            }
        }

        pub fn derivate(self: *Self, input: *Tensor(T)) !void {
            _ = self;
            //checks
            if (input.size <= 0) return TensorError.ZeroSizeTensor;

            //apply ReLU
            //OSS: can be improved, see how did I parallelized CPU Tensor Sum
            for (0..(input.size - 1)) |i| {
                input.data[i] = if (input.data[i] <= 0) 0 else 1;
            }
        }
    };
}

pub fn Sigmoid(comptime T: anytype) type {
    return struct {
        const Self = @This();
        //it directly modify the input tensor
        pub fn forward(self: *Self, input: *Tensor(T)) !void {
            _ = self;
            //checks
            if (input.size <= 0) return TensorError.ZeroSizeTensor;

            //apply Sigmoid
            for (0..input.size) |i| {
                input.data[i] = 1.0 / (1.0 + @exp(-input.data[i]));
            }
        }

        pub fn derivate(self: *Self, input: *Tensor(T)) !void {
            _ = self;
            //checks
            if (input.size <= 0) return TensorError.ZeroSizeTensor;

            //apply Sigmoid
            for (0..input.size) |i| {
                input.data[i] = input.data[i] * (1.0 - input.data[i]);
            }
        }
    };
}

pub fn Softmax(comptime T: anytype) type {
    return struct {
        const Self = @This();
        //it directly modify the input tensor
        pub fn forward(self: *Self, input: *Tensor(T)) !void {
            _ = self;
            const allocator = std.heap.page_allocator;

            const location = try allocator.alloc(usize, input.shape.len);
            defer allocator.free(location);

            //fill starting location to all zeros
            for (0..input.shape.len) |i| {
                location[i] = 0;
            }
            // std.debug.print("\n Softmax forward input", .{});
            // input.info();

            try compute_2D_softmax(input);

            // std.debug.print("\n >>>>>>>>>>>>>>Softmax forward input post compute_2D_softmax", .{});
            input.info();
        }

        fn compute_2D_softmax(input: *Tensor(T)) !void {
            const rows = input.shape[0];
            const cols = input.shape[1];

            var sum_of_exp: T = 0.0;
            var val: T = undefined;

            // std.debug.print("\n Softmax input inside compute_2D_softmax", .{});
            // input.info();

            //calculating the value of the exponential for each element
            for (0..cols) |j| {
                sum_of_exp = 0.0;
                for (0..rows) |i| {
                    val = input.data[i * cols + j];
                    //std.debug.print("\nval: {}", .{val});
                    val = @exp(val);
                    input.data[i * cols + j] = val;
                    //std.debug.print(" exp: {}", .{val});
                    sum_of_exp += val;
                }
                for (0..rows) |i| {
                    input.data[i * cols + j] = input.data[i * cols + j] / sum_of_exp;
                }
            }
        }

        fn compute_mutidim_softmax(input: *Tensor(T), current_depth: usize, location: []usize) !void {
            if (current_depth == (input.shape.len - 1)) {
                //declaring res as the result of the sum of the MSE
                const allocator = std.heap.page_allocator;

                //get location is used just to manage the gets and sets relative to the current depth
                const get_location = try allocator.alloc(usize, location.len);
                defer allocator.free(get_location);
                //initializing get location to the same values of location
                for (0..get_location.len) |i| {
                    get_location[i] = location[i];
                }

                //input.info();

                //allocating space for the exponent of each value
                var sum_of_exp: T = 0.0;
                var val: T = undefined;
                var exp: T = undefined;

                //calculating the value of the exponential for each element
                for (0..input.shape[current_depth]) |i| {
                    get_location[current_depth] = i; //for each element of predictions vect and target vect
                    val = try input.get_at(get_location);
                    exp = @exp(val);
                    try input.set_at(get_location, exp);
                    sum_of_exp += exp;
                }

                //set the value of current_elem/sum_of_exp
                for (0..input.shape[current_depth]) |i| {
                    get_location[current_depth] = i; //for each element of predictions vect and target vect
                    val = try input.get_at(get_location);
                    val = val / sum_of_exp;
                    try input.set_at(get_location, val);
                }
            } else {
                for (0..input.shape[current_depth]) |element_at_current_depth| {
                    //print depth:
                    //std.debug.print("\n depth: {} element_at_current_depth: {}", .{ current_depth, element_at_current_depth });
                    location[current_depth] = element_at_current_depth;
                    //otherwise I have to go deeper
                    try compute_mutidim_softmax(
                        input,
                        current_depth + 1,
                        location,
                    );
                }
            }
        }

        pub fn derivate(self: *Self, dL_dX: *Tensor(T), act_forward_out: *Tensor(T)) !void {
            _ = self;
            // softmax_output: The output matrix from the Softmax forward pass.
            // dL_dS: The gradient of the loss with respect to the Softmax output (this is given to us during backpropagation).
            // dL_dX: The gradient of the loss with respect to the input matrix (this is what we are computing in the backward pass).

            //checks
            if (dL_dX.size <= 0) return TensorError.ZeroSizeTensor;

            const dim = dL_dX.shape.len;
            const rows = dL_dX.shape[dim - 2];
            const cols = dL_dX.shape[dim - 1];

            var dL_dS = try dL_dX.copy(); //the copy is necessary since we are going to modify dL_dX
            defer dL_dS.deinit();

            // Loop over each row (assuming we apply Softmax across rows)
            for (0..rows) |i| {
                // Loop over each element in the row
                for (0..cols) |j| {
                    var dL_dX_ij: T = 0;

                    // Calculate the gradient for input element x_ij
                    const softmax_j = act_forward_out.data[i * cols + j];

                    // Sum over all elements in the row to compute dL/dX_ij
                    for (0..cols) |k| {
                        const softmax_k = act_forward_out.data[i * cols + k];
                        const dL_dS_k = dL_dS.data[i * cols + k];

                        if (j == k) {
                            dL_dX_ij += dL_dS_k * softmax_k * (1 - softmax_j);
                        } else {
                            dL_dX_ij += dL_dS_k * -softmax_k * softmax_j;
                        }
                    }

                    // Store the computed gradient for input x_ij
                    dL_dX.data[i * cols + j] = dL_dX_ij;
                }
            }
        }
    };
}
