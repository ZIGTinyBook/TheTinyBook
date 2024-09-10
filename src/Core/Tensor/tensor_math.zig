const std = @import("std");
const Tensor = @import("tensor.zig").Tensor; // Import Tensor type
const Architectures = @import("./architectures.zig").Architectures; //Import Architectures type

const Error = error{
    UnknownArchitecture,
    UnderDevelopementArchitecture,
};

pub fn sum_tensors(comptime arch: Architectures, comptime T: anytype, t1: *Tensor(T), t2: *Tensor(T)) !void {

    //selecting between all possible architectures
    switch (arch) {
        Architectures.CPU => CPU_sum_tensors(T, t1, t2),

        Architectures.GPU => {
            std.debug.print("{} is under developement \n", .{arch});
            return Error.UnderDevelopementArchitecture;
        },
        Architectures.SP32 => {
            std.debug.print("{} is under developement \n", .{arch});
            return Error.UnderDevelopementArchitecture;
        },
        else => return Error.UnknownArchitecture,
    }
}

fn CPU_sum_tensors(comptime T: anytype, t1: *Tensor(T), t2: *Tensor(T)) void {
    ////WP by Mirko
    std.debug.print("\n sum of two tensors on CPU on {}\n", .{T});
    t1.info();
    t2.info();
}
