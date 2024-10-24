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

/// The `Model` struct represents a neural network model composed of multiple layers.
/// This model can be configured with a specific data type (`T`) and allocator. It supports
/// adding layers, running forward and backward passes, and manages the allocation and
/// deallocation of resources.
pub fn Model(comptime T: type, comptime allocator: *const std.mem.Allocator) type {
    return struct {
        layers: []layer.Layer(T, allocator) = undefined, // Array of layers in the model.
        allocator: *const std.mem.Allocator, // Allocator reference for dynamic memory allocation.
        input_tensor: tensor.Tensor(T), // Tensor that holds the model's input data.

        /// Initializes the model, setting up an empty list of layers and initializing
        /// the input tensor.
        ///
        /// # Errors
        /// Returns an error if memory allocation for the `layers` array or `input_tensor` fails.
        pub fn init(self: *@This()) !void {
            self.layers = try self.allocator.alloc(layer.Layer(T, allocator), 0);
            self.input_tensor = try tensor.Tensor(T).init(self.allocator);
        }

        /// Deinitializes the model, releasing memory for each layer and the input tensor.
        ///
        /// This method iterates through each layer, deinitializes it, and then frees
        /// the layer array and input tensor memory.
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

        /// Adds a new layer to the model.
        ///
        /// # Parameters
        /// - `new_layer`: A pointer to the new layer to add to the model.
        ///
        /// # Errors
        /// Returns an error if reallocating the `layers` array fails.
        pub fn addLayer(self: *@This(), new_layer: *layer.Layer(T, allocator)) !void {
            self.layers = try self.allocator.realloc(self.layers, self.layers.len + 1);
            self.layers[self.layers.len - 1] = new_layer.*;
        }

        /// Executes the forward pass through the model with the specified input tensor.
        ///
        /// # Parameters
        /// - `input`: A pointer to the input tensor.
        ///
        /// # Returns
        /// A tensor containing the model's output after the forward pass.
        ///
        /// # Errors
        /// Returns an error if any layer's forward pass or tensor copying fails.
        pub fn forward(self: *@This(), input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            self.input_tensor.deinit(); //Why this?! it is ok since in each epooch the input tensor must be initialized with the new incomming batch
            self.input_tensor = try input.copy();

            for (0..self.layers.len) |i| {
                try DataProc.normalize(T, try self.getPrevOut(i), NormalizType.UnityBasedNormalizartion);
                _ = try self.layers[i].forward(try self.getPrevOut(i));
            }
            return (try self.getPrevOut(self.layers.len)).*;
        }

        /// Executes the backward pass through the model with the specified gradient tensor.
        ///
        /// # Parameters
        /// - `gradient`: A pointer to the gradient tensor to backpropagate.
        ///
        /// # Returns
        /// A pointer to the final gradient tensor after the backward pass.
        ///
        /// # Errors
        /// Returns an error if any layer's backward pass or tensor copying fails.
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

        /// Retrieves the output of the specified layer or the input tensor for the first layer.
        ///
        /// # Parameters
        /// - `layer_numb`: The index of the layer whose output tensor is to be retrieved.
        ///
        /// # Returns
        /// A pointer to the output tensor of the specified layer, or the input tensor if
        /// `layer_numb` is zero.
        ///
        /// # Errors
        /// Returns an error if the index is out of bounds or other tensor-related errors occur.
        fn getPrevOut(self: *@This(), layer_numb: usize) !*tensor.Tensor(T) {
            if (layer_numb == 0) {
                return &self.input_tensor;
            } else {
                return &self.layers[layer_numb - 1].denseLayer.outputActivation;
            }
        }
    };
}
