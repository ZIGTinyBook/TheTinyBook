const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;

//returns the duration of inputFunction's execution
pub fn timekeeper(comptime T: type, inputFunction: T) i128 {
    const start: i128 = std.time.nanoTimestamp();
    std.debug.print("\n     start: {}", .{start});

    // Call the function
    inputFunction();

    const end: i128 = std.time.nanoTimestamp();
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
    const allocator = std.heap.page_allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1.0, 2.0, 3.0 },
        [_]u8{ 4.0, 5.0, 6.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();

    const duration = timekeeper(@TypeOf(tensor.print()), exampleFunction);
    std.debug.print("\n duration: {} nanoseconds\n", .{duration});
}
