//! Tensor math contains all the functions to perform operations on tensors,
//! when we will pass to
//! GPU or STM32 we will have to implement the same functions for
//! those architectures usally these are called kernels
//!
const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const Architectures = @import("architectures").Architectures; //Import Architectures type
const Converter = @import("typeC");
//import error libraries
const TensorMathError = @import("errorHandler").TensorMathError;
const ArchitectureError = @import("errorHandler").ArchitectureError;
const TensorError = @import("errorHandler").TensorError;

/// Function that add the bias for all the features in the tensor
pub fn add_bias(comptime T: anytype, tensor: *Tensor(T), bias: *Tensor(T)) !void {
    // Checks:
    if (tensor.size == 0) {
        return TensorError.EmptyTensor;
    }
    if (bias.size == 0) {
        return TensorError.EmptyTensor;
    }
    if (bias.shape.len != 1) {
        return TensorMathError.InputTensorsWrongShape;
    }
    const len = bias.shape[0];
    if (len != tensor.shape[tensor.shape.len - 1]) {
        return TensorMathError.InputTensorDimensionMismatch;
    }

    // Allocate an array for threads, one for each row of the tensor
    const allocator = std.heap.page_allocator;
    const num_threads = tensor.size / bias.size;

    var threads = try allocator.alloc(std.Thread, num_threads); //Array to save thread handles

    var index: usize = 0;
    var i: usize = 0;

    // Start a thread for each row of the tensor
    while (index < tensor.size) : (i += 1) {
        threads[i] = try std.Thread.spawn(.{}, add_bias_thread, .{ T, tensor.data, index, len, bias });
        index += len;
    }

    // Merges all threads
    for (threads) |*thread| {
        thread.join(); // Use try to catch any errors
    }

    // Free the thread array
    allocator.free(threads);
}

fn add_bias_thread(comptime T: anytype, array: []T, start: usize, len: usize, bias: *Tensor(T)) void {
    for (0..len) |i| {
        array[start + i] += bias.data[i];
    }
}
/// Performs the mean of a given tensor. It is a reduction operation, collapsing the whole tenosr into a single value.
pub fn mean(comptime T: anytype, tensor: *Tensor(T)) f32 {
    var res: f32 = 0;

    for (tensor.data) |*d| {
        res += Converter.convert(T, f32, d.*);
    }
    res = res / Converter.convert(usize, f32, tensor.size);
    return res;
}

///Returns a Tensor with the same shape pf t1 and t2, where each element --> out[location] = t1[location] + t2[location]
pub fn sum_tensors(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, t1: *Tensor(Tin), t2: *Tensor(Tin)) !Tensor(Tout) {

    //selecting between all possible architectures
    return switch (arch) {
        Architectures.CPU => return CPU_sum_tensors(Tin, Tout, t1, t2),

        Architectures.GPU => {
            std.debug.print("{} is under developement \n", .{arch});
            return ArchitectureError.UnderDevelopementArchitecture;
        },
        Architectures.SP32 => {
            std.debug.print("{} is under developement \n", .{arch});
            return ArchitectureError.UnderDevelopementArchitecture;
        },
        else => return ArchitectureError.UnknownArchitecture,
    };
}

//Return the sum of the tensors inside another Tensor (t3)
fn CPU_sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {
    // CHECKS:
    if (t1.size != t2.size) return TensorMathError.InputTensorDifferentSize;

    if (@bitSizeOf(outputType) <= 16) { // quantized
        if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
    } else { // non-quant
        if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
    }

    // Allocating the array for the sum
    var out_sum = try t1.allocator.alloc(outputType, t1.size);
    defer t1.allocator.free(out_sum); // Ensure out_sum gets freed in case of error

    var i: usize = 0;
    const unroll_factor: usize = 4;

    // Loop unrolling
    while (i + unroll_factor <= t1.size) : (i += 4) {
        out_sum[i] = t1.data[i] + t2.data[i];
        out_sum[i + 1] = t1.data[i + 1] + t2.data[i + 1];
        out_sum[i + 2] = t1.data[i + 2] + t2.data[i + 2];
        out_sum[i + 3] = t1.data[i + 3] + t2.data[i + 3];
    }

    // Handle any remaining elements
    while (i < t1.size) : (i += 1) {
        out_sum[i] = t1.data[i] + t2.data[i];
    }

    // Create output tensor
    const out_tensor = try Tensor(outputType).fromArray(t1.allocator, out_sum, t1.shape);

    // Remove the defer since the tensor will manage its own memory after creation
    return out_tensor;
}

/// Returns the dot product of two tensors. The dot product is the sum of the products of the corresponding entries of the two sequences of numbers.
/// Deprecated: use dot_product_tensor instead
pub fn compute_dot_product(comptime T: type, input: *Tensor(T), weights: *Tensor(T)) !Tensor(T) {
    return try CPU_dot_product_tensors(T, T, input, weights);
}

/// Returns the dot product of two tensors. The dot product is the sum of the products of the corresponding entries of the two sequences of numbers.
pub fn dot_product_tensor(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, t1: *Tensor(Tin), t2: *Tensor(Tin)) !Tensor(Tout) {
    return switch (arch) {
        Architectures.CPU => return CPU_dot_product_tensors(Tin, Tout, t1, t2),
        Architectures.GPU => {
            std.debug.print("{} is under development\n", .{arch});
            return ArchitectureError.UnderDevelopmentArchitecture;
        },
        Architectures.SP32 => {
            std.debug.print("{} is under development\n", .{arch});
            return ArchitectureError.UnderDevelopmentArchitecture;
        },
        else => return ArchitectureError.UnknownArchitecture,
    };
}
/// Implementation of dot product for CPU architecture still not parallelized
pub fn CPU_dot_product_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !Tensor(outputType) {

    //CHECKS :
    const nDimT1 = t1.shape.len; //number of dimesion of tensor 1
    const nDimT2 = t2.shape.len; //number of dimesion of tensor 2
    // -imput shape:
    if (nDimT1 != nDimT2) return TensorMathError.InputTensorDifferentShape;

    //-dimensional compatibility:
    // If you have two matrices A and B, to compute the product A×B, the number of columns in A must be equal to the number of rows in B.
    // If A is a matrix of dimensions m×n and B is a matrix of dimensions n×p, then the product A×B is defined, and it results in a matrix of dimensions m×p.
    if (t1.shape[nDimT1 - 1] != t2.shape[nDimT1 - 2]) return TensorMathError.InputTensorsWrongShape;

    // -this check is necassary to avoid loss of information/ overflow when working with quantized tensors
    // usually quantization reduce to a maximum of 16bit, to the next check is divided between quant and non-quant data
    //bool (1 bit)
    // u1 (1 bit)
    // i8 (8 bits)
    // u8 (8 bits)
    // i16 (16 bits)
    // u16 (16 bits)
    // f16 (16 bits)
    // i32 (32 bits)
    // u32 (32 bits)
    // f32 (32 bits)
    // i64 (64 bits)
    // u64 (64 bits)
    // f64 (64 bits)
    // i128 (128 bits)
    // u128 (128 bits)
    // f128 (128 bits)
    if (@TypeOf(outputType) == @TypeOf(inputType)) {
        // Se input e output sono dello stesso tipo, non eseguire il controllo
        // Evitiamo l'errore in questo caso
    } else {
        if (@bitSizeOf(outputType) <= 16) { //quantized
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
        } else { //non-quant
            if (@bitSizeOf(outputType) <= @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
        }
    }

    //CREATING output_tensor :

    const allocator = std.heap.page_allocator;
    var out_shape = try allocator.alloc(usize, nDimT1); //I had to use alloc() bacause nDimT1 is not known at comptime
    //defining the resulting shape
    for (0..(nDimT1 - 2)) |i| {
        out_shape[i] = t1.shape[i];
    }
    out_shape[nDimT1 - 2] = t1.shape[nDimT1 - 2];
    out_shape[nDimT1 - 1] = t2.shape[nDimT1 - 1];

    var out_tensor = try Tensor(outputType).fromShape(&allocator, out_shape);
    try out_tensor.set(0, 0);
    //initialize the current location to all 0
    const location = try allocator.alloc(usize, nDimT1);
    for (location) |*loc| {
        loc.* = 0;
    }

    //call mutidim_mat_mul to handle multidimensionality
    try multidim_multiplication(
        inputType,
        outputType,
        t1,
        t2,
        &out_tensor,
        0,
        location,
    );
    //print output tensor shape
    //std.debug.print("\n output tensor shape: {}", .{out_tensor.shape[0]});

    return out_tensor;
}
/// Function that performs the multiplication of two tensors used in a recursive way to handle multidimensional tensors
fn multidim_multiplication(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType), t3: *Tensor(outputType), current_depth: usize, location: []usize) !void {
    if (current_depth == (t1.shape.len - 2)) {

        //declaring sum
        var sum: outputType = 0;

        //with the first two for loop I iterate over t3
        for (0..t1.shape[current_depth]) |row| { //for each row of t1

            for (0..t2.shape[current_depth + 1]) |col| { //for each col of t2

                sum = 0;

                for (0..t1.shape[current_depth + 1]) |i| {

                    //compose the location on t1
                    location[t1.shape.len - 1] = i; //location
                    location[t1.shape.len - 2] = row; //location

                    //getting the correct numbers in t1
                    const a = try t1.get_at(location);

                    //compose the location on t2
                    location[t1.shape.len - 1] = col; //location
                    location[t1.shape.len - 2] = i; //location

                    //getting the correct numbers in t2
                    const b = try t2.get_at(location);

                    sum += a * b;
                }

                //compose the location on t3
                location[t1.shape.len - 1] = col; //col on the out tensor matrix
                location[t1.shape.len - 2] = row; //row on the out tensor matrix

                // std.debug.print("\n set at location: [", .{});
                // for (location) |l| {
                //     std.debug.print(" {}", .{l});
                // }
                //std.debug.print("] val: {} ", .{sum});
                try t3.set_at(location, sum);
            }
        }
    } else {
        for (0..t1.shape[current_depth]) |element_at_current_depth| {
            //print location:
            //std.debug.print("\n depth: {} element_at_current_depth: {}", .{ current_depth, element_at_current_depth });
            location[current_depth] = element_at_current_depth;
            //otherwise I have to go deeper
            try multidim_multiplication(
                inputType,
                outputType,
                t1,
                t2,
                t3,
                current_depth + 1,
                location,
            );
        }
    }
}
pub fn convolve_tensor(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, input: *Tensor(Tin), kernel: *Tensor(Tin)) !Tensor(Tout) {
    return switch (arch) {
        Architectures.CPU => return CPU_convolve_tensors(Tin, Tout, input, kernel),
        Architectures.GPU => {
            std.debug.print("{} è in fase di sviluppo\n", .{arch});
            return ArchitectureError.UnderDevelopmentArchitecture;
        },
        Architectures.SP32 => {
            std.debug.print("{} è in fase di sviluppo\n", .{arch});
            return ArchitectureError.UnderDevelopmentArchitecture;
        },
        else => return ArchitectureError.UnknownArchitecture,
    };
}

/// Convolution fro CPU Architecture
pub fn CPU_convolve_tensors(comptime inputType: anytype, comptime outputType: anytype, input: *Tensor(inputType), kernel: *Tensor(inputType)) !Tensor(outputType) {
    // CHECKS:
    const nDimInput = input.shape.len; // Dimension of the input tensor
    const nDimKernel = kernel.shape.len; // Dimension of the kernel tensor

    // Verify that the input and kernel tensors have the same number of dimensions
    if (nDimInput != nDimKernel) return TensorMathError.InputTensorDifferentShape;

    // Verify that the kernel is smaller than the input tensor in all dimensions
    for (0..nDimInput) |i| {
        if (kernel.shape[i] > input.shape[i]) return TensorMathError.InputTensorsWrongShape;
    }

    // Check that the output tensor is large enough to hold the result
    if (@TypeOf(outputType) == @TypeOf(inputType)) {} else {
        if (@bitSizeOf(outputType) <= 16) {
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
        } else { // Non quantizzato
            if (@bitSizeOf(outputType) <= @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
        }
    }

    // Creation of the output tensor
    const allocator = std.heap.page_allocator;
    var out_shape = try allocator.alloc(usize, nDimInput);
    for (0..nDimInput) |i| {
        out_shape[i] = input.shape[i] - kernel.shape[i] + 1;
    }

    var out_tensor = try Tensor(outputType).fromShape(&allocator, out_shape);
    try out_tensor.set(0, 0);

    const location = try allocator.alloc(usize, nDimInput);
    for (location) |*loc| {
        loc.* = 0;
    }

    // Multi-dimensional convolution
    try multidim_convolution(
        inputType,
        outputType,
        input,
        kernel,
        &out_tensor,
        0,
        location,
    );

    return out_tensor;
}

/// Function that performs the convolution of two tensors used in a recursive way to handle multidimensional tensors
fn multidim_convolution(comptime inputType: anytype, comptime outputType: anytype, input: *Tensor(inputType), kernel: *Tensor(inputType), output: *Tensor(outputType), current_dim: usize, location: []usize) !void {
    if (current_dim == input.shape.len) {
        // Base Case: calculate in this location

        var sum: outputType = 0;
        const nDims = input.shape.len;

        const kernel_indices = try std.heap.page_allocator.alloc(usize, nDims);
        const input_indices = try std.heap.page_allocator.alloc(usize, nDims);

        // SUm over the kernel
        try sum_over_kernel(
            inputType,
            outputType,
            input,
            kernel,
            &sum,
            location,
            kernel_indices,
            input_indices,
            0,
        );

        try output.set_at(location, sum);

        std.heap.page_allocator.free(kernel_indices);
        std.heap.page_allocator.free(input_indices);
    } else {
        for (0..output.shape[current_dim]) |i| {
            location[current_dim] = i;
            try multidim_convolution(
                inputType,
                outputType,
                input,
                kernel,
                output,
                current_dim + 1,
                location,
            );
        }
    }
}

/// Recursive function to sum over the kernel
fn sum_over_kernel(comptime inputType: anytype, comptime outputType: anytype, input: *Tensor(inputType), kernel: *Tensor(inputType), sum: *outputType, input_location: []usize, kernel_indices: []usize, input_indices: []usize, current_dim: usize) !void {
    if (current_dim == input.shape.len) {
        const input_value = try input.get_at(input_indices);
        const kernel_value = try kernel.get_at(kernel_indices);
        sum.* += input_value * kernel_value;
    } else {
        for (0..kernel.shape[current_dim]) |i| {
            kernel_indices[current_dim] = i;
            input_indices[current_dim] = input_location[current_dim] + i;

            try sum_over_kernel(
                inputType,
                outputType,
                input,
                kernel,
                sum,
                input_location,
                kernel_indices,
                input_indices,
                current_dim + 1,
            );
        }
    }
}

pub fn CPU_convolve_tensors_with_bias(
    comptime inputType: anytype,
    comptime outputType: anytype,
    input: *Tensor(inputType),
    kernel: *Tensor(inputType),
    bias: outputType,
) !Tensor(outputType) {
    // CHECKS:
    const nDimInput = input.shape.len; // Dimension of the input tensor
    const nDimKernel = kernel.shape.len; // Dimension of the kernel tensor

    // Verify that the input and kernel tensors have the same number of dimensions
    if (nDimInput != nDimKernel) return TensorMathError.InputTensorDifferentShape;

    // Verify that the kernel is smaller than the input tensor in all dimensions
    for (0..nDimInput) |i| {
        if (kernel.shape[i] > input.shape[i]) return TensorMathError.InputTensorsWrongShape;
    }

    // Check that the output tensor is large enough to hold the result
    if (@TypeOf(outputType) == @TypeOf(inputType)) {} else {
        if (@bitSizeOf(outputType) <= 16) {
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
        } else { // Non-quantized
            if (@bitSizeOf(outputType) <= @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
        }
    }

    // Creation of the output tensor
    const allocator = std.heap.page_allocator;
    var out_shape = try allocator.alloc(usize, nDimInput);
    defer allocator.free(out_shape);

    for (0..nDimInput) |i| {
        out_shape[i] = input.shape[i] - kernel.shape[i] + 1;
    }

    var out_tensor = try Tensor(outputType).fromShape(&allocator, out_shape);
    // Do not defer out_tensor.deinit() here; the caller will deinit it

    try out_tensor.set(0, 0);

    const location = try allocator.alloc(usize, nDimInput);
    defer allocator.free(location);

    for (location) |*loc| {
        loc.* = 0;
    }

    // Multi-dimensional convolution with bias
    try multidim_convolution_with_bias(
        inputType,
        outputType,
        input,
        kernel,
        &out_tensor,
        bias,
        0,
        location,
    );

    // Return the output tensor without deinitializing it
    return out_tensor;
}

/// Function that performs the convolution of two tensors with bias, used recursively to handle multidimensional tensors
/// Function that performs the convolution of two tensors with bias, used recursively to handle multidimensional tensors
fn multidim_convolution_with_bias(
    comptime inputType: anytype,
    comptime outputType: anytype,
    input: *Tensor(inputType),
    kernel: *Tensor(inputType),
    output: *Tensor(outputType),
    bias: outputType,
    current_dim: usize,
    location: []usize,
) !void {
    if (current_dim == input.shape.len) {
        // Base Case: calculate in this location

        var sum: outputType = 0;
        const nDims = input.shape.len;

        const kernel_indices = try std.heap.page_allocator.alloc(usize, nDims);
        defer std.heap.page_allocator.free(kernel_indices);
        const input_indices = try std.heap.page_allocator.alloc(usize, nDims);
        defer std.heap.page_allocator.free(input_indices);

        // Sum over the kernel
        try sum_over_kernel(
            inputType,
            outputType,
            input,
            kernel,
            &sum,
            location,
            kernel_indices,
            input_indices,
            0,
        );

        sum += bias; // Add the bias after summing over the kernel

        try output.set_at(location, sum);

        // Memory is freed automatically by defer statements

    } else {
        for (0..output.shape[current_dim]) |i| {
            location[current_dim] = i;
            try multidim_convolution_with_bias(
                inputType,
                outputType,
                input,
                kernel,
                output,
                bias,
                current_dim + 1,
                location,
            );
        }
    }
}
