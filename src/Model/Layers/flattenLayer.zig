const std = @import("std");
const Tensor = @import("Tensor");
const TensMath = @import("tensor_m");
const Layer = @import("Layer");
const Architectures = @import("architectures").Architectures;
const LayerError = @import("errorHandler").LayerError;

pub fn FlattenLayer(comptime T: type, alloc: *const std.mem.Allocator) type {
    return struct {
        // Flatten layer parameters
        input: Tensor.Tensor(T), // Stored input for backward pass
        output: Tensor.Tensor(T), // Flattened output
        allocator: *const std.mem.Allocator,

        // Placeholder struct for init arguments
        pub const FlattenInitArgs = struct {
            placeholder: bool,
        };

        pub fn create(self: *FlattenLayer(T, alloc)) Layer.Layer(T, alloc) {
            return Layer.Layer(T, alloc){
                .layer_type = Layer.LayerType.FlattenLayer,
                .layer_ptr = self,
                .layer_impl = &.{
                    .init = init,
                    .deinit = deinit,
                    .forward = forward,
                    .backward = backward,
                    .printLayer = printLayer,
                    .get_n_inputs = get_n_inputs,
                    .get_n_neurons = get_n_neurons,
                    .get_input = get_input,
                    .get_output = get_output,
                },
            };
        }

        /// Initialize the Flatten layer (just store allocator, no weights)
        pub fn init(ctx: *anyopaque, args: *anyopaque) !void {
            const self: *FlattenLayer(T, alloc) = @ptrCast(@alignCast(ctx));
            const argsStruct: *const FlattenInitArgs = @ptrCast(@alignCast(args));
            _ = argsStruct; // We don't really need the placeholder here

            self.allocator = alloc;

            return; // No errors expected
        }

        /// Deallocate the Flatten layer resources
        pub fn deinit(ctx: *anyopaque) void {
            const self: *FlattenLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            if (self.input.data.len > 0) {
                self.input.deinit();
            }

            if (self.output.data.len > 0) {
                self.output.deinit();
            }
        }

        /// Forward pass: Flatten the input tensor
        pub fn forward(ctx: *anyopaque, input: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *FlattenLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            // Save the input for backward pass
            if (self.input.data.len > 0) {
                self.input.deinit();
            }
            self.input = try input.copy();

            // Compute total size of the input
            var total_size: usize = 1;
            for (input.shape) |dim| {
                total_size *= dim;
            }

            var output_shape: [1]usize = .{total_size};

            if (self.output.data.len > 0) {
                self.output.deinit();
            }
            self.output = try Tensor.Tensor(T).fromArray(self.allocator, input.data, output_shape[0..]);

            return self.output;
        }

        /// Backward pass: Reshape the gradients to the original input shape
        pub fn backward(ctx: *anyopaque, dValues: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *FlattenLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            // `self.input.shape` might be []const usize, need a mutable slice
            const input_shape_const = self.input.shape;
            var input_shape = try self.allocator.alloc(usize, input_shape_const.len);
            _ = &input_shape;
            defer self.allocator.free(input_shape);
            @memcpy(input_shape, input_shape_const);

            var dInput = try Tensor.Tensor(T).fromArray(self.allocator, dValues.data, input_shape);
            _ = &dInput;
            return dInput;
        }

        /// Print the flatten layer information
        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            const self: *FlattenLayer(T, alloc) = @ptrCast(@alignCast(ctx));
            switch (choice) {
                0 => std.debug.print("Flatten Layer\n", .{}),
                1 => std.debug.print("Input shape: {any}, Output shape: {any}\n", .{ self.input.shape, self.output.shape }),
                else => {},
            }
        }

        //---------------------------------------------------------------
        //---------------------------- Getters --------------------------
        //---------------------------------------------------------------
        /// Get the number of inputs = product of input dimensions
        pub fn get_n_inputs(ctx: *anyopaque) usize {
            const self: *FlattenLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            var total: usize = 1;
            for (self.input.shape) |dim| {
                total *= dim;
            }
            return total;
        }

        /// For Flatten layer, number of neurons = number of inputs
        pub fn get_n_neurons(ctx: *anyopaque) usize {
            return get_n_inputs(ctx);
        }

        pub fn get_input(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *FlattenLayer(T, alloc) = @ptrCast(@alignCast(ctx));
            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *Tensor.Tensor(T) {
            const self: *FlattenLayer(T, alloc) = @ptrCast(@alignCast(ctx));
            return &self.output;
        }
    };
}
