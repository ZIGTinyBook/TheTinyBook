const std = @import("std");
const tensor = @import("tensor");
const TensMath = @import("tensor_m");
const Layer = @import("Layer");
const Architectures = @import("architectures").Architectures;
const LayerError = @import("errorHandler").LayerError;
const TensorError = @import("errorHandler").TensorError;

pub const PoolingType = enum {
    Max,
    Min,
    Avg,
};

/// TODO: implement padding
/// TODO: upgrade to multidim input Tensor
pub fn PoolingLayer(comptime T: type, allocator: *const std.mem.Allocator) type {
    return struct {
        input: tensor.Tensor(T), //is saved for semplicity, it can be sobstituted
        output: tensor.Tensor(T), // output = dot(input, weight.transposed) + bias
        used_input: tensor.Tensor(u1), //
        kernel: [2]usize, // kernerl.size=2 -> [rows, cols]
        stride: [2]usize, // stride.size=2 -> [rows, cols]
        poolingType: PoolingType,
        allocator: *const std.mem.Allocator,

        pub fn create(self: *PoolingLayer(T, allocator)) !Layer.Layer(T, allocator) {
            return Layer.Layer(T, allocator){
                .layer_type = Layer.LayerType.DenseLayer,
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

        ///Initilize the layer with random weights and biases
        /// also for the gradients
        pub fn init(ctx: *anyopaque, args: *anyopaque) !void {
            const self: *PoolingLayer(T, allocator) = @ptrCast(@alignCast(ctx));
            const argsStruct: *const struct { kernel: [2]usize, stride: [2]usize, poolingType: PoolingType } = @ptrCast(@alignCast(args));

            //initializing basic attributes
            @memcpy(self.kernel, argsStruct.kernel);
            @memcpy(self.stride, argsStruct.stride);
            self.poolingType = argsStruct.poolingType;

            std.debug.print("\nInit Pooling Layer", .{});

            //At the moment pooling is available only for 2D tensor
            if (self.kernel.size != 2 or self.stride.size != 2) {
                return LayerError.Only2DSupported;
            }
            //Check Kernel!=0
            for (self.kernel) |val| {
                if (val <= 0) return LayerError.ZeroValueKernel;
            }

            //impossible size kernel / stride
            if (self.kernel.size <= 0 or self.stride.size <= 0) {
                return TensorError.ZeroSizeTensor;
            }
            //Check stride!=0
            for (self.stride) |val| {
                if (val <= 0) return LayerError.ZeroValueStride;
            }
        }

        ///Deallocate the layer
        pub fn deinit(ctx: *anyopaque) void {
            const self: *PoolingLayer(T, allocator) = @ptrCast(@alignCast(ctx));

            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            if (self.input.data.len > 0) {
                self.input.deinit();
            }

            std.debug.print("\n Pooling layer resources deallocated.", .{});
        }

        /// Forward pass of the pooling layer
        /// We can improve it removing as much as possibile all the copy operations
        /// Keeps thrack of the used input valued for each kernel windows.
        /// Padding is not implemented jet.
        pub fn forward(ctx: *anyopaque, input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            const self: *PoolingLayer(T, allocator) = @ptrCast(@alignCast(ctx));

            //this copy is necessary for the backward
            if (self.input.data.len >= 0) {
                self.input.deinit();
            }
            self.input = try input.copy();

            // used_input remember wich values of the input went into the output, .fromShape() initialize all to zero
            self.used_input = try self.used_input.fromShape(allocator, input.shape);

            self.output = TensMath.pool_tensor(T, &self.input, &self.used_input, &self.kernel, &self.stride);

            return self.output;
        }

        /// Backward pass of the layer. It takes the dValues from the next layer and computes the gradients
        pub fn backward(ctx: *anyopaque, dValues: *tensor.Tensor(T)) !tensor.Tensor(T) {
            const self: *PoolingLayer(T, allocator) = @ptrCast(@alignCast(ctx));
            _ = dValues;
            _ = self;
        }

        ///Print the layer used for debug purposes it has 2 different verbosity levels
        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            const self: *PoolingLayer(T, allocator) = @ptrCast(@alignCast(ctx));

            std.debug.print("\n ************************Dense layer*********************", .{});
            //MENU choice:
            // 0 -> full details layer
            // 1 -> shape schema
            std.debug.print("\n kernel: {any}  stride:{any}", .{ self.kernel, self.stride });
            if (choice == 0) {
                std.debug.print("\n \n************input", .{});
                self.input.printMultidim();
                std.debug.print("\n \n************output", .{});
                self.output.printMultidim();
            }
            if (choice == 1) {
                std.debug.print("\n   input: [", .{});
                for (0..self.input.shape.len) |i| {
                    std.debug.print("{}", .{self.input.shape[i]});
                    if (i == self.input.shape.len - 1) {
                        std.debug.print("]", .{});
                    } else {
                        std.debug.print(" x ", .{});
                    }
                }
                std.debug.print("\n   output: [", .{});
                for (0..self.output.shape.len) |i| {
                    std.debug.print("{}", .{self.output.shape[i]});
                    if (i == self.output.shape.len - 1) {
                        std.debug.print("]", .{});
                    } else {
                        std.debug.print(" x ", .{});
                    }
                }
                std.debug.print("\n ", .{});
            }
        }

        //---------------------------------------------------------------
        //----------------------------getters----------------------------
        //---------------------------------------------------------------
        pub fn get_n_inputs(ctx: *anyopaque) usize {
            _ = ctx;
            return LayerError.FeatureNotSupported;
        }

        pub fn get_n_neurons(ctx: *anyopaque) usize {
            _ = ctx;
            return LayerError.FeatureNotSupported;
        }

        pub fn get_input(ctx: *anyopaque) *const tensor.Tensor(T) {
            const self: *PoolingLayer(T, allocator) = @ptrCast(@alignCast(ctx));

            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *tensor.Tensor(T) {
            const self: *PoolingLayer(T, allocator) = @ptrCast(@alignCast(ctx));

            return &self.output;
        }
    };
}
