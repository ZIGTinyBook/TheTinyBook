const std = @import("std");
const tMath = @import("./tensor_math.zig");
const Architectures = @import("./architectures.zig").Architectures; //Import Architectures type
const tk = @import("timekeep");

pub fn Tensor(comptime T: type) type {
    return struct {
        data: []T,
        size: usize,
        shape: []usize,
        allocator: *const std.mem.Allocator,

        pub fn fromArray(allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !@This() {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }

            const flatArray = try allocator.alloc(T, total_size);

            _ = flattenArray(T, inputArray, flatArray, 0);

            return @This(){
                .data = flatArray,
                .size = total_size,
                .shape = shape,
                .allocator = allocator,
            };
        }

        pub fn init(allocator: *const std.mem.Allocator, shape: []usize) !@This() {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }

            const data = try allocator.alloc(T, total_size);

            return @This(){
                .data = data,
                .size = total_size,
                .shape = shape,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.data);
        }

        pub fn getSize(self: *@This()) usize {
            var total_size: usize = 1;
            for (self.shape) |dim| {
                total_size *= dim;
            }
            return total_size;
        }

        pub fn get(self: *const @This(), idx: usize) !T {
            if (idx >= self.data.len) {
                return error.IndexOutOfBounds;
            }
            return self.data[idx];
        }

        pub fn set(self: *@This(), idx: usize, value: T) !void {
            if (idx >= self.data.len) {
                return error.IndexOutOfBounds;
            }
            self.data[idx] = value;
        }

        pub fn flatten_index(self: *const @This(), indices: []const usize) !usize {
            var idx: usize = 0;
            var stride: usize = 1;
            for (self.shape, 0..) |dim, i| {
                idx += indices[i] * stride;
                stride *= dim;
            }
            return idx;
        }

        pub fn get_at(self: *const @This(), indices: []const usize) !T {
            const idx = try self.flatten_index(indices);
            return self.get(idx);
        }

        pub fn set_at(self: *@This(), indices: []const usize, value: T) !void {
            const idx = try self.flatten_index(indices);
            return self.set(idx, value);
        }

        pub fn info(self: *@This()) void {
            std.debug.print("\ntensor infos: ", .{});
            std.debug.print("\n  data type:{}", .{@TypeOf(self.data[0])});
            std.debug.print("\n  size:{}", .{self.size});
            std.debug.print("\n  shape: [ ", .{});
            for (self.shape) |val| {
                std.debug.print("{} ", .{val});
            }
            std.debug.print("] ", .{});
        }

        pub fn print(self: *@This()) void {
            std.debug.print("\ntensor data: ", .{});
            for (self.data) |val| {
                std.debug.print("{} ", .{val});
            }
            std.debug.print("\n", .{});
        }
    };
}

// Funzione ricorsiva per appiattire un array multidimensionale
fn flattenArray(T: type, arr: anytype, flatArr: []T, startIndex: usize) usize {
    var idx = startIndex;

    if (@TypeOf(arr[0]) == T) {
        for (arr) |val| {
            flatArr[idx] = val;
            idx += 1;
        }
    } else {
        for (arr) |subArray| {
            idx = flattenArray(T, subArray, flatArr, idx);
        }
    }
    return idx;
}
