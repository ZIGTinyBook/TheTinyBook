//! This file contains the definition of the layers that can be used in the neural network.
//! There are function to initiialize random weigths, initialization right now is completely random but in the future
//! it will possible to use proper initialization techniques.
//! Layer can be stacked in a model and they implement proper forward and backward methods.

const std = @import("std");
const tensor = @import("tensor");
const TensMath = @import("tensor_m");
const Architectures = @import("architectures").Architectures;
const TensorError = @import("tensor_m").TensorError;
const ArchitectureError = @import("tensor_m").ArchitectureError;
const ActivLib = @import("activation_function");
const ActivationType = @import("activation_function").ActivationType;
//import error libraries
const LayerError = @import("errorHandler").LayerError;

pub const LayerType = enum {
    DenseLayer,
    DefaultLayer,
    ConvolutionalLayer,
    ActivationLayer,
    null,
};

//------------------------------------------------------------------------------------------------------
/// UTILS
/// Initialize a matrix of random values with a normal distribution
pub fn randn(comptime T: type, n_inputs: usize, n_neurons: usize) ![][]T {
    var rng = std.Random.Xoshiro256.init(12345);

    const matrix = try std.heap.page_allocator.alloc([]T, n_inputs);
    for (matrix) |*row| {
        row.* = try std.heap.page_allocator.alloc(T, n_neurons);
        for (row.*) |*value| {
            value.* = rng.random().floatNorm(T) + 1; // fix me!! why +1 ??
        }
    }
    return matrix;
}
///Function used to initialize a matrix of zeros used for bias
pub fn zeros(comptime T: type, n_inputs: usize, n_neurons: usize) ![][]T {
    const matrix = try std.heap.page_allocator.alloc([]T, n_inputs);
    for (matrix) |*row| {
        row.* = try std.heap.page_allocator.alloc(T, n_neurons);
        for (row.*) |*value| {
            value.* = 0;
        }
    }
    return matrix;
}

//------------------------------------------------------------------------------------------------------
// INTERFACE LAYER

pub fn Layer(comptime T: type, allocator: *const std.mem.Allocator) type {
    return struct {
        layer_type: LayerType,
        layer_ptr: *anyopaque,
        layer_impl: *const Basic_Layer_Interface,

        pub const Basic_Layer_Interface = struct {
            init: *const fn (ctx: *anyopaque, n_inputs: usize, n_neurons: usize) anyerror!void,
            convInit: *const fn (ctx: *anyopaque, input_channels: usize, output_channels: usize, kernel_size: [2]usize) anyerror!void,
            deinit: *const fn (ctx: *anyopaque) void,
            forward: *const fn (ctx: *anyopaque, input: *tensor.Tensor(T)) anyerror!tensor.Tensor(T),
            backward: *const fn (ctx: *anyopaque, dValues: *tensor.Tensor(T)) anyerror!tensor.Tensor(T),
            printLayer: *const fn (ctx: *anyopaque, choice: u8) void,
            get_n_inputs: *const fn (ctx: *anyopaque) usize,
            get_n_neurons: *const fn (ctx: *anyopaque) usize,
            get_input: *const fn (ctx: *anyopaque) *const tensor.Tensor(T),
            get_output: *const fn (ctx: *anyopaque) *tensor.Tensor(T),
        };

        pub fn init(self: Layer(T, allocator), n_inputs: usize, n_neurons: usize) anyerror!void {
            return self.layer_impl.init(self.layer_ptr, n_inputs, n_neurons);
        }

        pub fn convInit(self: Layer(T, allocator), input_channels: usize, output_channels: usize, kernel_size: [2]usize) anyerror!void {
            return self.layer_impl.convInit(self.layer_ptr, input_channels, output_channels, kernel_size);
        }

        /// When deinit() pay attention to:
        /// - Double-freeing memory.
        /// - Using uninitialized or already-deallocated pointers.
        /// - Incorrect allocation or deallocation logic.
        ///
        pub fn deinit(self: Layer(T, allocator)) void {
            return self.layer_impl.deinit(self.layer_ptr);
        }
        pub fn forward(self: Layer(T, allocator), input: *tensor.Tensor(T)) !tensor.Tensor(T) {
            return self.layer_impl.forward(self.layer_ptr, input);
        }
        pub fn backward(self: Layer(T, allocator), dValues: *tensor.Tensor(T)) !tensor.Tensor(T) {
            return self.layer_impl.backward(self.layer_ptr, dValues);
        }
        pub fn printLayer(self: Layer(T, allocator), choice: u8) void {
            return self.layer_impl.printLayer(self.layer_ptr, choice);
        }
        pub fn get_n_inputs(self: Layer(T, allocator)) usize {
            return self.layer_impl.get_n_inputs(self.layer_ptr);
        }
        pub fn get_n_neurons(self: Layer(T, allocator)) usize {
            return self.layer_impl.get_n_neurons(self.layer_ptr);
        }
        pub fn get_input(self: Layer(T, allocator)) *const tensor.Tensor(T) {
            return self.layer_impl.get_input(self.layer_ptr);
        }
        pub fn get_output(self: Layer(T, allocator)) *tensor.Tensor(T) {
            return self.layer_impl.get_output(self.layer_ptr);
        }
    };
}
