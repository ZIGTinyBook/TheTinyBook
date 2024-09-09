const std = @import("std");

pub fn Tensor(comptime T: type) type {
    return struct {
        data: []T, // Array of elements of type T
        size: usize, // Size of the tensor
        shape: []usize,
        allocator: *const std.mem.Allocator,

        pub fn init(allocator: *const std.mem.Allocator, shape: []usize) !@This() {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }

            const data = try allocator.alloc(T, total_size);

            std.debug.print("\nInit tensor", .{});

            return @This(){
                .data = data,
                .size = total_size,
                .shape = shape,
                .allocator = allocator,
            };
        }

        /// Frees the memory allocated for the Tensor
        pub fn deinit(self: *@This()) void {
            std.debug.print("\nDeinit tensor", .{});

            self.allocator.free(self.data);
        }

        /// Returns the number of elements in the Tensor
        pub fn getSize(self: *@This()) usize {
            var total_size: usize = 1;
            for (self.shape) |dim| {
                total_size *= dim;
            }
            return total_size;
        }

        /// Gets the element at the given flattened index
        pub fn get(self: *const @This(), idx: usize) !f64 {
            if (idx >= self.data.len) {
                return error.IndexOutOfBounds;
            }
            return self.data[idx];
        }

        /// Sets the element at the given flattened index
        pub fn set(self: *@This(), idx: usize, value: f64) !void {
            if (idx >= self.data.len) {
                return error.IndexOutOfBounds;
            }
            self.data[idx] = value;
        }

        /// Flattens multi-dimensional indices into a single index
        pub fn flatten_index(self: *const @This(), indices: []const usize) !usize {
            var idx: usize = 0;
            var stride: usize = 1;
            for (self.shape, 0..) |dim, i| {
                idx += indices[i] * stride;
                stride *= dim;
            }
            return idx;
        }

        /// Get element using multi-dimensional indices
        pub fn get_at(self: *const @This(), indices: []const usize) !f64 {
            const idx = try self.flatten_index(indices);
            return self.get(idx);
        }

        /// Set element using multi-dimensional indices
        pub fn set_at(self: *@This(), indices: []const usize, value: f64) !void {
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
    };
}

pub fn main() !void {
    std.debug.print("\nmain start", .{});
    const allocator = std.heap.page_allocator;

    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor(f64).init(&allocator, &shape);
    defer tensor.deinit();

    tensor.info();

    std.debug.print("\nmain end", .{});
}
