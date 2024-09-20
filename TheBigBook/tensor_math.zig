const std = @import("std");
const Tensor = @import("tensor.zig").Tensor; // Import Tensor type
const Architectures = @import("./architectures.zig").Architectures; //Import Architectures type

pub const ArchitectureError = error{
    UnknownArchitecture,
    UnderDevelopementArchitecture,
};

pub const TensorError = error{
    MemError,
    InputTensorDifferentSize,
    InputTensorDifferentShape,
    InputTensorsWrongShape, //launched in dot_product
    OutputTensorDifferentSize,
    TooSmallOutputType, //the type dimension of the output Tensor could coause a loss of information
    InputTensorDimensionMismatch,
};

pub fn sum_tensors(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, t1: *Tensor(Tin), t2: *Tensor(Tin), t3: *Tensor(Tout), allocator: *const std.mem.Allocator) !void {

    //selecting between all possible architectures
    return switch (arch) {
        Architectures.CPU => CPU_sum_tensors(Tin, Tout, t1, t2, t3, allocator),

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
fn CPU_sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType), t3: *Tensor(outputType), allocator: *const std.mem.Allocator) !void {
    if (t2.shape.len != 2 or t1.shape.len < 2) {
        return TensorError.InputTensorDimensionMismatch;
    }

    t3.deinit();

    const n_rows = t1.shape[0]; // Numero di righe di t1
    const n_neurons = t1.shape[t1.shape.len - 1]; // Numero di colonne (neuroni)
    const output_shape = [_]usize{ n_rows, n_neurons };

    // Inizializza i campi di `t3` senza riassegnare il puntatore
    t3.shape = try allocator.alloc(usize, output_shape.len);
    @memcpy(t3.shape, output_shape[0..]);
    t3.size = n_rows * n_neurons;
    t3.data = try allocator.alloc(outputType, t3.size);

    // Controlla che t2 sia un vettore riga (1 x numero_colonne_di_t1)
    if (t2.shape[0] == 1 and t2.shape[1] == t1.shape[t1.shape.len - 1]) {
        var row: usize = 0;
        while (row < n_rows) {
            var col: usize = 0;
            while (col < n_neurons) {
                const idx = row * n_neurons + col;
                if (idx < t1.size and idx < t3.size) {
                    t3.data[idx] = t1.data[idx] + t2.data[col];
                }
                col += 1;
            }
            row += 1;
        }
    } else if (t1.size == t2.size) {
        // Somma classica tra tensori con dimensioni uguali
        var i: usize = 0;
        while (i < t1.size) {
            t3.data[i] = t1.data[i] + t2.data[i]; // Somma elemento per elemento
            i += 1;
        }
    } else {
        return TensorError.InputTensorDifferentSize;
    }
}

pub fn compute_dot_product(comptime T: type, input: *Tensor(T), weights: *Tensor(T)) !*Tensor(T) {
    return try CPU_dot_product_tensors(T, T, input, weights);
}

pub fn dot_product_tensor(comptime arch: Architectures, comptime Tin: anytype, comptime Tout: anytype, t1: *Tensor(Tin), t2: *Tensor(Tin)) !*Tensor(Tout) {
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

pub fn CPU_dot_product_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *Tensor(inputType), t2: *Tensor(inputType)) !*Tensor(outputType) {
    //CHECKS :
    // -input size
    if (t1.size != t2.size) return TensorError.InputTensorDifferentSize;

    const nDimT1 = t1.shape.len; //number of dimesion of tensor 1
    const nDimT2 = t2.shape.len; //number of dimesion of tensor 2
    // -imput shape
    if (nDimT1 != nDimT2) return TensorError.InputTensorDifferentShape;
    if (t1.shape[nDimT1 - 1] != t2.shape[nDimT1 - 2] or t1.shape[nDimT1 - 2] != t2.shape[nDimT1 - 1]) return TensorError.InputTensorsWrongShape;

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
            if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorError.TooSmallOutputType;
        } else { //non-quant
            if (@bitSizeOf(outputType) <= @bitSizeOf(inputType)) return TensorError.TooSmallOutputType;
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

    const out_tensor = try Tensor(outputType).init(&allocator, out_shape);

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
        out_tensor,
        0,
        location,
    );
    //print output tensor shape
    std.debug.print("\n output tensor shape: {}", .{out_tensor.shape[0]});

    return out_tensor;
}

pub fn multidim_multiplication(
    comptime inputType: anytype,
    comptime outputType: anytype,
    t1: *Tensor(inputType),
    t2: *Tensor(inputType),
    t3: *Tensor(outputType),
    current_depth: usize,
    location: []usize,
) !void {
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

                std.debug.print("\n set at location: [", .{});
                for (location) |l| {
                    std.debug.print(" {}", .{l});
                }
                std.debug.print("] val: {} ", .{sum});
                try t3.set_at(location, sum);
            }
        }
    } else {
        for (0..t1.shape[current_depth]) |element_at_current_depth| {
            //print location:
            std.debug.print("\n depth: {} element_at_current_depth: {}", .{ current_depth, element_at_current_depth });
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
