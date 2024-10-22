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
            self.input_tensor = try tensor.Tensor(T).init(self.allocator);
        }

        pub fn deinit(self: *@This()) void {
            for (self.layers, 0..) |*layer_, i| {
                layer_.deinit();
                std.debug.print("\n -.-.-> dense layer {} deinitialized", .{i});
            }
            self.allocator.free(self.layers);
            std.debug.print("\n -.-.-> model layers deinitialized", .{});

            self.input_tensor.deinit(); // pay attention! input_tensor is initialised only if forward() is run at leas once. Sess self.forward()
            std.debug.print("\n -.-.-> model input_tensor deinitialized", .{});
        }

        pub fn addLayer(self: *@This(), new_layer: *layer.Layer(T, allocator)) !void {
            self.layers = try self.allocator.realloc(self.layers, self.layers.len + 1);
            self.layers[self.layers.len - 1] = new_layer.*;
        }

        pub fn forward(self: *@This(), input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            self.input_tensor.deinit(); //OSS: also if is the first time that forward() is run is not a problem since input_tensorhas been initialized in @This().init()
            self.input_tensor = try input.copy();
            for (0..self.layers.len) |i| {
                try DataProc.normalize(T, try self.getPrevOut(i), NormalizType.UnityBasedNormalizartion);

                _ = try self.layers[i].forward(try self.getPrevOut(i));
            }
            return (try self.getPrevOut(self.layers.len)).*;
        }

        pub fn backward(self: *@This(), gradient: *tensor.Tensor(T)) !*tensor.Tensor(T) {
            var grad = gradient;
            var grad_duplicate = try grad.copy();
            defer grad_duplicate.deinit(); // Assicura che grad_duplicate venga deallocato

            var counter = (self.layers.len - 1);
            while (counter >= 0) : (counter -= 1) {
                std.debug.print("\n--------------------------------------backwarding layer {}", .{counter});
                grad = try self.layers[counter].backward(&grad_duplicate);
                grad_duplicate = try grad.copy();
                if (counter == 0) break; // Uscita forzata quando si raggiunge il primo layer
            }

            return grad;
        }

        fn getPrevOut(self: *@This(), layer_numb: usize) !*tensor.Tensor(T) {
            if (layer_numb == 0) {
                return &self.input_tensor;
            } else {
                return &self.layers[layer_numb - 1].denseLayer.outputActivation;
            }
        }
    };
}
