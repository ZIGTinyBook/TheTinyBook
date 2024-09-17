const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;


// LossFunction Interface
pub const LossFunction = struct {

    pub fn compute(self: *const LossFunction, predictions: *[]f64, targets: *[]f64) f64 {
        return self.lossFn(predictions, targets);
    }

    pub fn derivative(self: *const LossFunction, predictions: *[]f64, targets: *[]f64, out_gradient: *[]f64) void {
        return self.gradientFn(predictions, targets, out_gradient);
    }

    // Function pointers for dynamic behavior
    lossFn: fn(predictions: *[]f64, targets: *[]f64) f64, //
    gradientFn: fn(predictions: *[]f64, targets: *[]f64, out_gradient: []f64) void,

};

pub const MSELoss = struct {
    fn lossFn(predictions: []f64, targets: []f64) f64 {
        var sum: f64 = 0;
        for (predictions, 0..) |pred, i| {
            const diff = pred - targets[i];
            sum += diff * diff;
        }
        return sum / @intToFloat(f64, predictions.len);
    }

    fn gradientFn(predictions: []f64, targets: []f64, out_gradient: []f64) void {
        for (predictions) |pred, i| {
            out_gradient[i] = 2.0 * (pred - targets[i]) / @intToFloat(f64, predictions.len);
        }
    }

    pub fn new() LossFunction {
        return LossFunction{
            .computeFn = MSELoss.lossFn,
            .derivativeFn = MSELoss.gradientFn,
        };
    }
};

pub const CrossEntropyLoss = struct {
    fn lossFn(predictions: []f64, targets: []f64) f64 {
        var sum: f64 = 0;
        for (predictions) |pred, i| {
            sum += -targets[i] * std.math.log(pred) - (1.0 - targets[i]) * std.math.log(1.0 - pred);
        }
        return sum / @intToFloat(f64, predictions.len);
    }

    fn gradientFn(predictions: []f64, targets: []f64, out_gradient: []f64) void {
        for (predictions, 0..) |pred, i| {
            out_gradient[i] = (pred - targets[i]) / (pred * (1.0 - pred));
        }
    }

    pub fn new() LossFunction {
        return LossFunction{
            .computeFn = CrossEntropyLoss.lossFn,
            .derivativeFn = CrossEntropyLoss.gradientFn,
        };
    }
};
