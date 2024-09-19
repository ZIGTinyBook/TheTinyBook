const std = @import("std");
const tMath = @import("./tensor_math.zig");
const Architectures = @import("./architectures.zig").Architectures; //Import Architectures type

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
            for (0..shape.len) |i| {
                total_size *= shape[i];
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
            for (0..self.shape.len) |i| {
                std.debug.print("{} ", .{self.shape[i]});
            }
            std.debug.print("] ", .{});
        }

        pub fn print(self: *@This()) void {
            std.debug.print("\ntensor data: ", .{});
            for (0..self.data.len) |i| {
                std.debug.print("{} ", .{self.data[i]});
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

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1.0, 2.0, 3.0 },
        [_]u8{ 4.0, 5.0, 6.0 },
    };
    var inputArray2: [2][3]u8 = [_][3]u8{
        [_]u8{ 6.0, 5.0, 4.0 },
        [_]u8{ 3.0, 2.0, 1.0 },
    };
    var inputArray3: [2][3]i32 = [_][3]i32{
        [_]i32{ 6.0, 5.0, 4.0 },
        [_]i32{ 3.0, 2.0, 1.0 },
    };
    var inputArray4: [3][2]u8 = [_][2]u8{
        [_]u8{ 6.0, 5.0 },
        [_]u8{ 3.0, 2.0 },
        [_]u8{ 1.0, 2.0 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };
    var shape4: [2]usize = [_]usize{ 3, 2 };

    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();
    tensor.print();

    var tensor2 = try Tensor(u8).fromArray(&allocator, &inputArray2, &shape);
    defer tensor2.deinit();
    //tensor2.info();

    var tensor3 = try Tensor(i32).fromArray(&allocator, &inputArray3, &shape);
    defer tensor3.deinit();
    //tensor3.info();

    //Just a bunch of trials
    try tMath.sum_tensors(Architectures.CPU, u8, i32, &tensor, &tensor2, &tensor3);

    var tensor4 = try Tensor(u8).fromArray(&allocator, &inputArray4, &shape4);
    defer tensor4.deinit();
    tensor4.print();

    const tensor5 = try tMath.dot_product_tensor(Architectures.CPU, u8, i32, &tensor, &tensor4);
    defer tensor5.deinit();

    tensor5.info();
    tensor5.print();
}
