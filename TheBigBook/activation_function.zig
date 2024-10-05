const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;
const TensorError = @import("./tensor.zig").TensorError;

// activation function Interface

pub fn ActivationFunction(activationFunctionStruct: fn () type) type {
    const act = activationFunctionStruct(){};
    return struct {
        activation: activationFunctionStruct() = act,

        pub fn forward(self: *@This(), comptime T: anytype, input: *Tensor(T)) !void {
            try self.activation.forward(T, input);
        }

        pub fn derivate(self: *@This(), comptime T: anytype, input: *Tensor(T)) !void {
            try self.activation.derivate(T, input);
        }
    };
}

pub fn ReLU() type {
    return struct {
        //it directly modify the input tensor
        //threshold is usually set to zero
        pub fn forward(self: *@This(), comptime T: anytype, input: *Tensor(T)) !void {
            _ = self;
            //checks
            if (input.size <= 0) return TensorError.ZeroSizeTensor;

            //apply ReLU
            //OSS: can be improved, see how did I parallelized CPU Tensor Sum
            for (0..input.size) |i| {
                if (input.data[i] <= 0) input.data[i] = 0;
            }
        }

        pub fn derivate(self: *@This(), comptime T: anytype, input: *Tensor(T)) !void {
            _ = self;
            //checks
            if (input.size <= 0) return TensorError.ZeroSizeTensor;

            //apply ReLU
            //OSS: can be improved, see how did I parallelized CPU Tensor Sum
            for (0..(input.size - 1)) |i| {
                if (input.data[i] <= 0) input.data[i] = 0;
            }
        }
    };
}

pub fn Softmax() type {
    return struct {

        //it directly modify the input tensor
        pub fn forward(self: *@This(), comptime T: anytype, input: *Tensor(T)) !void {
            _ = self;
            const allocator = std.heap.page_allocator;

            const location = try allocator.alloc(usize, input.shape.len);
            defer allocator.free(location);

            //fill starting location to all zeros
            for (0..input.shape.len) |i| {
                location[i] = 0;
            }

            return compute_mutidim_softmax(T, input, 0, location);
        }

        fn compute_mutidim_softmax(comptime T: anytype, input: *Tensor(T), current_depth: usize, location: []usize) !void {
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
                        T,
                        input,
                        current_depth + 1,
                        location,
                    );
                }
            }
        }

        pub fn derivate(self: *@This(), comptime T: anytype, input: *Tensor(T)) !void {
            _ = self;
            //checks
            if (input.size <= 0) return TensorError.ZeroSizeTensor;
        }
    };
}
