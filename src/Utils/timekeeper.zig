const std = @import("std");

//returns the duration of inputFunction's execution
pub fn timekeeper(comptime T: type, inputFunction: T) i64 {
    const start: i64 = std.time.timestamp();
    std.debug.print("\n     start: {}", .{start});

    // Call the function
    inputFunction();

    const end: i64 = std.time.timestamp();
    std.debug.print("\n     end: {}", .{end});
    const duration_ns = end - start;

    return duration_ns;
}

// Example function to be timed
fn exampleFunction() void {
    var sum: i64 = 0;
    var u: usize = 1;
    for (0..100000000) |j| {
        sum = 13;
        u = j;
        for (0..10) |i| {
            sum = sum * 2;
            u = i + 1 - 1;
        }
    }
    std.debug.print("\n sum: {}", .{sum});
}

pub fn main() void {
    const duration = timekeeper(@TypeOf(exampleFunction), exampleFunction);
    std.debug.print("\n duration: {}\n", .{duration});
}
