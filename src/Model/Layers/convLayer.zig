const std = @import("std");
const Tensor = @import("Tensor");
const TensMath = @import("tensor_m");
const Layer = @import("Layer");
const Architectures = @import("architectures").Architectures;
const LayerError = @import("errorHandler").LayerError;

pub fn ConvolutionalLayer(comptime T: type, alloc: *const std.mem.Allocator) type {
    return struct {
        // Convolutional layer parameters
        weights: Tensor.Tensor(T), // Weights (kernels) of shape [out_channels, in_channels, kernel_height, kernel_width]
        bias: Tensor.Tensor(T), // Biases for each output channel
        input: Tensor.Tensor(T), // Input tensor
        output: Tensor.Tensor(T), // Output tensor after convolution
        // Layer configuration
        input_channels: usize,
        output_channels: usize,
        kernel_size: [2]usize, // [kernel_height, kernel_width]
        // Gradients
        w_gradients: Tensor.Tensor(T),
        b_gradients: Tensor.Tensor(T),
        // Utils
        allocator: *const std.mem.Allocator,

        pub fn create(self: *ConvolutionalLayer(T, alloc)) Layer.Layer(T, alloc) {
            return Layer.Layer(T, alloc){
                .layer_type = Layer.LayerType.ConvolutionalLayer,
                .layer_ptr = self,
                .layer_impl = &.{
                    .init = init,
                    .convInit = convInit,
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

        pub fn init(ctx: *anyopaque, n_inputs: usize, n_neurons: usize) !void {
            _ = ctx;
            _ = n_inputs;
            _ = n_neurons;
            return LayerError.InvalidLayerType;
        }

        /// Initialize the convolutional layer with random weights and biases
        pub fn convInit(ctx: *anyopaque, input_channels: usize, output_channels: usize, kernel_size: [2]usize) !void {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));
            //std.debug.print("\nInit ConvolutionalLayer: input_channels = {}, output_channels = {}, kernel_size = {:?}, Type = {}\n", .{ input_channels, output_channels, kernel_size, @TypeOf(T) });

            // Check parameters
            if (input_channels <= 0 or output_channels <= 0) return LayerError.InvalidParameters;
            if (kernel_size[0] <= 0 or kernel_size[1] <= 0) return LayerError.InvalidParameters;

            // Initialize layer configuration
            self.input_channels = input_channels;
            self.output_channels = output_channels;
            self.kernel_size = kernel_size;
            self.allocator = alloc;

            // Initialize weights and biases
            var weight_shape: [4]usize = [_]usize{ output_channels, input_channels, kernel_size[0], kernel_size[1] };
            var bias_shape: [2]usize = [_]usize{ output_channels, 1 };

            const weight_array = try Layer.randn(T, 1, output_channels * input_channels * kernel_size[0] * kernel_size[1]);
            const bias_array = try Layer.randn(T, 1, output_channels);

            self.weights = try Tensor.Tensor(T).fromArray(alloc, weight_array, &weight_shape);
            self.bias = try Tensor.Tensor(T).fromArray(alloc, bias_array, &bias_shape);

            // Initialize gradients
            self.w_gradients = try Tensor.Tensor(T).fromShape(self.allocator, &weight_shape);
            self.b_gradients = try Tensor.Tensor(T).fromShape(self.allocator, &bias_shape);
        }

        /// Deallocate the convolutional layer resources
        pub fn deinit(ctx: *anyopaque) void {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            // Deallocate tensors if allocated
            if (self.weights.data.len > 0) {
                self.weights.deinit();
            }

            if (self.output.data.len > 0) {
                self.output.deinit();
            }

            if (self.bias.data.len > 0) {
                self.bias.deinit();
            }

            if (self.w_gradients.data.len > 0) {
                self.w_gradients.deinit();
            }

            if (self.b_gradients.data.len > 0) {
                self.b_gradients.deinit();
            }

            if (self.input.data.len > 0) {
                self.input.deinit();
            }

            //std.debug.print("\nConvolutionalLayer resources deallocated.\n", .{});
        }

        /// Forward pass of the convolutional layer
        pub fn forward(ctx: *anyopaque, input: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            // Save input for backward pass
            if (self.input.data.len > 0) {
                self.input.deinit();
            }
            self.input = try input.copy();

            // Perform convolution operation
            self.output = try TensMath.CPU_convolve_tensors_with_bias(T, T, &self.input, &self.weights, &self.bias);

            return self.output;
        }

        /// Backward pass of the convolutional layer
        pub fn backward(ctx: *anyopaque, dValues: *Tensor.Tensor(T)) !Tensor.Tensor(T) {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            // Initialize gradient tensors if not already initialized
            if (self.w_gradients.data.len > 0) {
                self.w_gradients.deinit();
            }
            if (self.b_gradients.data.len > 0) {
                self.b_gradients.deinit();
            }

            // Compute gradients with respect to biases
            // Sum over the spatial dimensions
            self.b_gradients = try TensMath.convolution_backward_biases(T, dValues);

            // Compute gradients with respect to weights
            self.w_gradients = try TensMath.convolution_backward_weights(T, &self.input, dValues);

            // Compute gradients with respect to input
            var dInput = try TensMath.convolution_backward_input(T, dValues, &self.weights);
            _ = &dInput;
            return dInput;
        }

        /// Print the convolutional layer information (To be written)
        pub fn printLayer(ctx: *anyopaque, choice: u8) void {
            _ = ctx;
            _ = choice;
        }

        //---------------------------------------------------------------
        //---------------------------- Getters --------------------------
        //---------------------------------------------------------------
        pub fn get_n_inputs(ctx: *anyopaque) usize {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            // For convolutional layers, n_inputs can be considered as input_channels
            return self.input_channels;
        }

        pub fn get_n_neurons(ctx: *anyopaque) usize {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            // For convolutional layers, n_neurons can be considered as output_channels
            return self.output_channels;
        }

        pub fn get_weights(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            return &self.weights;
        }

        pub fn get_bias(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            return &self.bias;
        }

        pub fn get_input(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            return &self.input;
        }

        pub fn get_output(ctx: *anyopaque) *Tensor.Tensor(T) {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            return &self.output;
        }

        pub fn get_weightGradients(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            return &self.w_gradients;
        }

        pub fn get_biasGradients(ctx: *anyopaque) *const Tensor.Tensor(T) {
            const self: *ConvolutionalLayer(T, alloc) = @ptrCast(@alignCast(ctx));

            return &self.b_gradients;
        }
    };
}
