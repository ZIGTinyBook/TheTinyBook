const std = @import("std");
const Tensor = @import("tensor.zig").Tensor; // Import Tensor type
const Architectures = @import("./architectures.zig").Architectures; //Import Architectures type

const ArchitectureError = error{
    UnknownArchitecture,
    UnderDevelopementArchitecture,
};

const TensorError = error{
    InputTensorDifferentSize,
    InputTensorDifferentShape,
    InputTensorsWrongShape, //launched in dot_product
    OutputTensorDifferentSize,
    TooSmallOutputType, //the type dimension of the output Tensor could coause a loss of information
};

pub fn sum_tensors(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, t1: *Tensor(Tin), t2: *Tensor(Tin), t3: *Tensor(Tout)) !void {

    //selecting between all possible architectures
    return switch (arch) {
        Architectures.CPU => CPU_sum_tensors(Tin, Tout, t1, t2, t3),

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

//return the sum of the tensors inside another Tensor and put into t3
fn CPU_sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType), t3: *Tensor(outputType)) !void {

    //CHECKS :
    // -input size
    if (t1.size != t2.size) return TensorError.InputTensorDifferentSize;

    // -output size
    if (t1.size != t3.size) return TensorError.OutputTensorDifferentSize;

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
    if (@sizeOf(outputType) <= 16) { //quantized
        if (@sizeOf(outputType) <= (@sizeOf(inputType) * 2)) return TensorError.TooSmallOutputType;
    } else { //non-quant
        if (@sizeOf(outputType) <= @sizeOf(inputType)) return TensorError.TooSmallOutputType;
    }

    var i: usize = 0;
    const unroll_factor: usize = 4;

    // loop unrolling
    while (i + unroll_factor <= t1.size) : (i += 4) {
        //since the Type of t3 is higher in number of bits the cast shoudl happen autonomously
        t3.data[i] = t1.data[i] + t2.data[i];
        t3.data[i + 1] = t1.data[i + 1] + t2.data[i + 1];
        t3.data[i + 2] = t1.data[i + 2] + t2.data[i + 2];
        t3.data[i + 3] = t1.data[i + 3] + t2.data[i + 3];
    }

    // Handle any remaining elements
    while (i < t1.size) : (i += 1) {
        t3.data[i] = t1.data[i] + t2.data[i];
    }
}

pub fn dot_product_tensor(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, t1: *Tensor(Tin), t2: *Tensor(Tin)) !*Tensor(Tout) {

    //We can see tensors as an array of arrays ... of matrixes
    //ex 3D:
    //      3D_Tensor = {{matr1},{matr2},{matr3}}
    //      4D_Tensor = { { {matr1},{matr2} } , { {matr3}{matr4} } }
    //this is important to udersand the code that follows

    //selecting between all possible architectures
    return switch (arch) {
        Architectures.CPU => return CPU_dot_product_tensors(Tin, Tout, t1, t2),

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

pub fn CPU_dot_product_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !*Tensor(outputType) {

    //CHECKS :
    // -input size
    if (t1.size != t2.size) return TensorError.InputTensorDifferentSize;

    const nDimT1 = t1.shape.len; //number of dimesion of tensor 1
    const nDimT2 = t2.shape.len; //number of dimesion of tensor 2
    // -imput shape
    if (nDimT1 != nDimT2) return TensorError.InputTensorDifferentShape;
    if (t1.shape[nDimT1 - 1] != t1.shape[nDimT1 - 2] or t1.shape[nDimT1 - 2] != t1.shape[nDimT1 - 1]) return TensorError.InputTensorsWrongShape;

    //CREATING output_tensor :

    //get the size of the output
    var size: usize = 1;
    for (0..(nDimT1 - 2)) |i| {
        size = size * t1.shape[i];
    }
    size = size * t1.shape[nDimT1 - 1] * t1.shape[nDimT1 - 1];

    const allocator = std.heap.page_allocator;
    var out_shape = try allocator.alloc(outputType, nDimT1); //I had to use alloc() bacause nDimT1 is not known at comptime
    //defining the resulting shape
    for (0..(nDimT1 - 2)) |i| {
        out_shape[i] = t1.shape[i];
    }
    out_shape[nDimT1 - 2] = t1.shape[nDimT1 - 2];
    out_shape[nDimT1 - 1] = t2.shape[nDimT1 - 1];

    var out_tensor = try Tensor(u8).init(&allocator, &out_shape);

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
    if (@sizeOf(outputType) <= 16) { //quantized
        if (@sizeOf(outputType) <= (@sizeOf(inputType) * 2)) return TensorError.TooSmallOutputType;
    } else { //non-quant
        if (@sizeOf(outputType) <= @sizeOf(inputType)) return TensorError.TooSmallOutputType;
    }

    //initialize the current location to all 0
    const location: [nDimT1]usize = [_]usize{0} ** nDimT1;

    //call mutidim_mat_mul to handle multidimensionality
    multidim_multiplication(
        inputType,
        outputType,
        &t1,
        &t2,
        &out_tensor,
        0,
        location,
    );

    return *out_tensor;
}

pub fn multidim_multiplication(
    comptime inputType: anytype,
    comptime outputType: anytype,
    t1: *Tensor(inputType),
    t2: *Tensor(inputType),
    t3: *Tensor(outputType),
    current_depth: usize,
    location: []const usize,
) void {
    for (0..t1.shape[current_depth]) |element_at_current_depth| {
        //print location:
        std.debug.print("\n depth: {} location: [", .{element_at_current_depth});
        for (location) |l| {
            std.debug.print(" {}", .{l});
        }
        std.debug.print("]", .{});

        if (current_depth == (t1.shape.len - 2)) {
            //here I can do a classic matrix multiplication in 2D
            std.debug.print("\n out_tensor : ", .{});
            t3.set_at(location, 1);
        } else {
            //otherwise I have to go deeper
            multidim_multiplication(
                inputType,
                outputType,
                t1,
                t2,
                t3,
                current_depth + 1,
            );
        }
    }
}
