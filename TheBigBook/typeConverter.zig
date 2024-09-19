const std = @import("std");

pub fn convert(comptime T_in: type, comptime T_out: type, value: T_in) T_out {
    return switch (@typeInfo(T_in)) {
        .Int => switch (@typeInfo(T_out)) {
            .Int => @intCast(value), // Integer to integer
            .Float => @floatFromInt(value), // Integer to float
            else => @compileError("Unsupported conversion from integer to this type"),
        },
        .Float => switch (@typeInfo(T_out)) {
            .Int => @intFromFloat(value), // Float to integer
            .Float => @floatCast(value), // Float to float
            else => @compileError("Unsupported conversion from float to this type"),
        },
        else => @compileError("Unsupported input type"),
    };
}
