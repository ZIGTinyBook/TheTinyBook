const std = @import("std");
const Tensor = @import("./tensor.zig").Tensor;
const TensorMathError = @import("./tensor_math.zig").TensorMathError;
const Convert = @import("./typeConverter.zig");

pub const LossError = error{
    SizeMismatch,
    ShapeMismatch,
    InvalidPrediction,
};

//LossFunction Interface
pub fn LossFunction(lossFunctionStruct: fn () type) type {
    const ls = lossFunctionStruct(){};
    return struct {
        loss: lossFunctionStruct() = ls,

        //return a rensor where the smallest element is the result of the loss function for each array of weights
        //ex:
        // PredictionTens =[ [ vect , vect ],
        //                   [ vect , vect ],
        //                   [ vect , vect ] ] -> 3 x 2 x vect.len
        // TargetTens = same of prediction
        // OutputTens = [ [ a, b],
        //                [ c, d],
        //                [ e, f] ] -> 3 x 2 where each letter is the result of the loss function applied to the "vect" of predictions and targets
        //
        pub fn computeLoss(self: *@This(), comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T) {
            return try self.loss.computeLoss(T, predictions, targets);
        }
        pub fn computeGradient(self: *@This(), comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !*Tensor(T) {
            return try self.loss.computeGradient(T, predictions, targets);
        }
    };
}

pub fn MSELoss() type {
    return struct {
        //return a rensor where the smallest element is the result of the loss function for each array of weights
        //         //ex:
        //         // PredictionTens =[ [ vect , vect ],
        //         //                   [ vect , vect ],
        //         //                   [ vect , vect ] ] -> 3 x 2 x vect.len
        //         // TargetTens = same of prediction
        //         // OutputTens = [ [ a, b],
        //         //                [ c, d],
        //         //                [ e, f] ] -> 3 x 2 where each letter is the result of the loss function applied to the "vect" of predictions and targets
        //         //
        //
        pub fn computeLoss(self: *@This(), comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T) {
            _ = self;
            //CHECKS :
            // -inputs size
            if (predictions.size != targets.size) return TensorMathError.InputTensorDifferentSize;

            //create the shape of the output tensor
            const allocator = std.heap.page_allocator;
            var out_shape = allocator.alloc(usize, (predictions.shape.len - 1)) catch {
                return TensorMathError.MemError;
            };

            for (0..out_shape.len) |i| {
                out_shape[i] = predictions.shape[i];
            }

            var out_tensor = Tensor(T).fromShape(&allocator, out_shape) catch {
                return TensorMathError.MemError;
            };

            //initialize the current location to all 0
            const location = allocator.alloc(usize, predictions.shape.len) catch {
                return TensorMathError.MemError;
            };

            for (location) |*loc| {
                loc.* = 0;
            }

            //call mutidim_mat_mul to handle multidimensionality
            try multidim_MSE(
                T,
                predictions,
                targets,
                &out_tensor,
                0,
                location,
            );

            //out_tensor.info();
            return out_tensor;
        }

        fn multidim_MSE(comptime T: type, predictions: *Tensor(T), targets: *Tensor(T), out_tensor: *Tensor(T), current_depth: usize, location: []usize) !void {
            //      0                  1
            if (current_depth == (predictions.shape.len - 1)) {
                //declaring res as the result of the sum of the MSE
                var res: T = 0;
                const allocator = std.heap.page_allocator;

                const get_location = try allocator.alloc(usize, location.len);
                defer allocator.free(get_location);
                //initializing get location to the same values of location
                for (0..get_location.len) |i| {
                    get_location[i] = location[i];
                }
                //calculating the loss
                for (0..predictions.shape[current_depth]) |i| {
                    get_location[current_depth] = i; //for each element of predictions vect and target vect

                    const target = try targets.get_at(location);
                    const prediction = try predictions.get_at(location);
                    const diff = target - prediction;
                    res += diff * diff;
                }
                const divisor: T = Convert.convert(usize, T, predictions.shape[current_depth]);
                switch (@typeInfo(T)) {
                    .Int => res = @divFloor(res, divisor),
                    else => res = res / divisor,
                }

                //declaring and initializing the landing location of the sum
                const out_location = try allocator.alloc(usize, predictions.shape.len - 1);
                defer allocator.free(out_location);
                for (0..out_location.len) |i| {
                    out_location[i] = location[i];
                }

                //set the loss value into out_tensor
                try out_tensor.set_at(out_location, res);
            } else {
                // for 0,1
                for (0..predictions.shape[current_depth]) |element_at_current_depth| {
                    //print depth:
                    //std.debug.print("\n depth: {} element_at_current_depth: {}", .{ current_depth, element_at_current_depth });
                    location[current_depth] = element_at_current_depth;
                    //otherwise I have to go deeper
                    try multidim_MSE(
                        T,
                        predictions,
                        targets,
                        out_tensor,
                        current_depth + 1,
                        location,
                    );
                }
            }
        }

        pub fn computeGradient(self: *@This(), comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !*Tensor(T) {
            _ = self;

            //check on the size of predictions, targets and gradient
            if (predictions.size != targets.size) {
                return LossError.SizeMismatch;
            }
            //checks on the shape of predictions, targets and gradient
            if (predictions.shape.len != targets.shape.len) {
                return LossError.ShapeMismatch;
            }
            for (predictions.shape, 0..) |*dim, i| {
                if (dim.* != targets.shape[i]) return LossError.ShapeMismatch;
            }

            var gradient = try Tensor(T).fromShape(predictions.allocator, predictions.shape);

            const n: f32 = @floatFromInt(predictions.size);

            for (0..predictions.size) |i| {
                gradient.data[i] = (2.0 / n) * (predictions.data[i] - targets.data[i]);
            }

            return &gradient;
        }
        // -inputs size
    };
}
//Categorical Cross-Entropy loss function
pub fn CCELoss() type {
    return struct {
        //return a rensor where the smallest element is the result of the loss function for each array of weights
        //         //ex:
        //         // PredictionTens =[ [ vect , vect ],
        //         //                   [ vect , vect ],
        //         //                   [ vect , vect ] ] -> 3 x 2 x vect.len
        //         // TargetTens = same of prediction
        //         // OutputTens = [ [ a, b],
        //         //                [ c, d],
        //         //                [ e, f] ] -> 3 x 2 where each letter is the result of the loss function applied to the "vect" of predictions and targets
        //         //
        //
        pub fn computeLoss(self: *@This(), comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !Tensor(T) {
            _ = self;

            //CHECKS :
            // -inputs size
            if (predictions.size != targets.size) return TensorMathError.InputTensorDifferentSize;

            //create the shape of the output tensor
            const allocator = std.heap.page_allocator;
            var out_shape = allocator.alloc(usize, (predictions.shape.len - 1)) catch {
                return TensorMathError.MemError;
            };

            for (0..out_shape.len) |i| {
                out_shape[i] = predictions.shape[i];
            }

            var out_tensor = Tensor(T).fromShape(&allocator, out_shape) catch {
                return TensorMathError.MemError;
            };

            //initialize the current location to all 0
            const location = allocator.alloc(usize, predictions.shape.len) catch {
                return TensorMathError.MemError;
            };

            for (location) |*loc| {
                loc.* = 0;
            }

            //call mutidim_mat_mul to handle multidimensionality
            try multidim_CCE(
                T,
                predictions,
                targets,
                &out_tensor,
                0,
                location,
            );

            //out_tensor.info();
            return out_tensor;
        }

        fn multidim_CCE(comptime T: type, predictions: *Tensor(T), targets: *Tensor(T), out_tensor: *Tensor(T), current_depth: usize, location: []usize) !void {
            if (current_depth == (predictions.shape.len - 1)) {
                //declaring res as the result of the sum of the MSE
                var res: f32 = 0.0;
                const allocator = std.heap.page_allocator;

                const get_location = try allocator.alloc(usize, location.len);
                defer allocator.free(get_location);
                //initializing get location to the same values of location
                for (0..get_location.len) |i| {
                    get_location[i] = location[i];
                }

                //predictions.info();
                //targets.info();
                //std.debug.print("\n predictions get_at [0, 1]:{} ", .{try predictions.get_at(&[2]usize{ 0, 1 })});
                //std.debug.print("\n predictions get 1:{} ", .{try predictions.get(1)});

                //calculating the loss
                for (0..predictions.shape[current_depth]) |i| {
                    get_location[current_depth] = i; //for each element of predictions vect and target vect
                    const target = try targets.get_at(get_location);
                    const prediction = try predictions.get_at(get_location);
                    const log = std.math.log(f32, std.math.e, prediction);
                    res -= (target * log);
                    //std.debug.print("\n CCE get_at pred:{} trg:{} log:{} at: ", .{ prediction, target, log });
                    // for (get_location) |*val| {
                    //     std.debug.print("{} ", .{val.*});
                    // }
                }

                //declaring and initializing the landing location of the sum
                const out_location = try allocator.alloc(usize, predictions.shape.len - 1);
                defer allocator.free(out_location);
                for (0..out_location.len) |i| {
                    out_location[i] = location[i];
                }

                const out_res: T = Convert.convert(f32, T, res);
                //set the loss value into out_tensortry
                //std.debug.print("\n CCE set val {} at: ", .{out_res});
                // for (out_location) |*val| {
                //     std.debug.print("{} ", .{val.*});
                // }
                try out_tensor.set_at(out_location, out_res);
            } else {
                // for 0,1
                for (0..predictions.shape[current_depth]) |element_at_current_depth| {
                    //print depth:
                    //std.debug.print("\n depth: {} element_at_current_depth: {}", .{ current_depth, element_at_current_depth });
                    location[current_depth] = element_at_current_depth;
                    //otherwise I have to go deeper
                    try multidim_CCE(
                        T,
                        predictions,
                        targets,
                        out_tensor,
                        current_depth + 1,
                        location,
                    );
                }
            }
        }

        pub fn computeGradient(self: *@This(), comptime T: type, predictions: *Tensor(T), targets: *Tensor(T)) !*Tensor(T) {
            _ = self;

            //check on the size of predictions, targets and gradient
            if (predictions.size != targets.size) {
                return LossError.SizeMismatch;
            }

            //checks on the shape of predictions, targets and gradient
            if (predictions.shape.len != targets.shape.len) {
                return LossError.ShapeMismatch;
            }
            for (predictions.shape, 0..) |*dim, i| {
                if (dim.* != targets.shape[i]) return LossError.ShapeMismatch;
            }

            const n = predictions.size;
            var gradient = try Tensor(T).fromShape(predictions.allocator, predictions.shape);

            for (0..n) |i| {
                if (predictions.data[i] == 0.0) {
                    return LossError.InvalidPrediction; // Avoid division by zero
                }
                gradient.data[i] = -targets.data[i] / predictions.data[i];
            }

            return &gradient;
        }
    };
}
