const std = @import("std");
const tensor = @import("tensor");
const layer = @import("layers");
const Loss = @import("loss");
const LossType = @import("loss").LossType;
const TensMath = @import("tensor_m");
const Optim = @import("optim");
const loader = @import("dataloader").DataLoader;
const NormalizType = @import("dataprocessor").NormalizationType;
const DataProc = @import("dataprocessor");

pub fn Model(comptime T: type, comptime allocator: *const std.mem.Allocator) type {
    return struct {
        layers: []layer.Layer(T, allocator) = undefined,
        allocator: *const std.mem.Allocator,
        input_tensor: tensor.Tensor(T),

        pub fn init(self: *@This()) !void {
            self.layers = try self.allocator.alloc(layer.Layer(T, allocator), 0);
            self.input_tensor = undefined;
        }

        pub fn deinit(self: *@This()) void {
            for (self.layers) |*layer_| {
                layer_.deinit();
            }
            self.allocator.free(self.layers);
        }

        pub fn addLayer(self: *@This(), new_layer: *layer.Layer(T, allocator)) !void {
            self.layers = try self.allocator.realloc(self.layers, self.layers.len + 1);
            self.layers[self.layers.len - 1] = new_layer.*;
        }

        pub fn forward(self: *@This(), input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            var output = try input.copy();
            self.input_tensor = try input.copy();
            for (0..self.layers.len) |i| {
                std.debug.print("\n-------------------------------pre-norm layer {}", .{i});
                try DataProc.normalize(T, &output, NormalizType.UnityBasedNormalizartion);
                std.debug.print("\n-------------------------------post-norm layer {}", .{i});
                std.debug.print("\n>>>>>>>>>>>>>  input layer {} normalized  <<<<<<<<<<<<", .{i});
                output.info();
                output = try self.layers[i].forward(&output);
                std.debug.print("\n>>>>>>>>>>>>>  output layer {}  <<<<<<<<<<<<", .{i});
                output.info();
            }
            return output;
        }

        pub fn backward(self: *@This(), gradient: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            //grad is always equal to dot(grad, weights)
            var grad = gradient;
            var grad_duplicate = try grad.copy();
            var counter = (self.layers.len - 1);
            while (counter >= 0) : (counter -= 1) {
                std.debug.print("\n--------------------------------------backwarding layer {}", .{counter});
                grad = try self.layers[counter].backward(&grad_duplicate);
                grad_duplicate = try grad.copy();
                if (counter == 0) break;
            }

            return grad;
        }
    };
}
