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
            defer output.deinit(); // Deallochiamo la copia alla fine del forward

            self.input_tensor = try input.copy();
            defer self.input_tensor.deinit(); // Dealloca self.input_tensor alla fine

            for (0..self.layers.len) |i| {
                std.debug.print("\n-------------------------------pre-norm layer {}", .{i});
                std.debug.print("\n>>>>>>>>>>>>>  input layer {} NOT normalized  <<<<<<<<<<<<", .{i});
                output.info();

                try DataProc.normalize(T, &output, NormalizType.UnityBasedNormalizartion);

                std.debug.print("\n-------------------------------post-norm layer {}", .{i});
                std.debug.print("\n>>>>>>>>>>>>>  input layer {} normalized  <<<<<<<<<<<<", .{i});
                output.info();

                var next_output = try self.layers[i].forward(&output);
                defer next_output.deinit(); // Dealloca la memoria dopo l'uso
                output = try next_output.copy(); // Copia il risultato per il prossimo layer
            }

            return output.copy(); // Ritorniamo una copia sicura di output
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
    };
}
