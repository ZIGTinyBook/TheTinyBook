const std = @import("std");

const Tensor = struct {
    data: []f64, // Array of elements of type T
    shape: []usize, // Shape of the tensor (e.g., [2, 3] for a 2x3 matrix)
    allocator: *const std.mem.Allocator,

    /// Creates a new Tensor with the given shape and allocates memory for it
    pub fn init(comptime T: type, allocator: *const std.mem.Allocator, shape: []usize) !Tensor {
        var total_size: usize = 1;
        for (shape) |dim| {
            total_size *= dim;
        }

        const data = try allocator.alloc(T, total_size);

        return Tensor{
            .data = data,
            .shape = shape,
            .allocator = allocator,
        };
    }

    /// Frees the memory allocated for the Tensor
    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
    }

    /// Returns the number of elements in the Tensor
    pub fn size(self: *Tensor) usize {
        var total_size: usize = 1;
        for (self.shape) |dim| {
            total_size *= dim;
        }
        return total_size;
    }

    /// Gets the element at the given flattened index
    pub fn get(self: *const Tensor, idx: usize) !f64 {
        if (idx >= self.data.len) {
            return error.IndexOutOfBounds;
        }
        return self.data[idx];
    }

    /// Sets the element at the given flattened index
    pub fn set(self: *Tensor, idx: usize, value: f64) !f64 {
        if (idx >= self.data.len) {
            return error.IndexOutOfBounds;
        }
        self.data[idx] = value;
    }

    /// Flattens multi-dimensional indices into a single index
    pub fn flatten_index(self: *const Tensor, indices: []usize) !usize {
        if (indices.len != self.shape.len) {
            return error.InvalidShape;
        }

        var flat_index: usize = 0;
        var multiplier: usize = 1;

        for (self.shape.len - 1) |i| {
            flat_index += indices[i] * multiplier;
            multiplier *= self.shape[i];
        }

        return flat_index;
    }

    /// Get element using multi-dimensional indices
    pub fn get_at(self: *const Tensor, indices: []usize) !f64 {
        const idx = try self.flatten_index(indices);
        return self.get(idx);
    }

    /// Set element using multi-dimensional indices
    pub fn set_at(self: *Tensor, indices: []usize, value: f64) !void {
        const idx = try self.flatten_index(indices);
        return self.set(idx, value);
    }
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Create a 2x3 Tensor of i32 values
    var shape: [2]usize = [_]usize{ 2, 3 };
    var tensor = try Tensor.init(f64, &allocator, &shape);

    defer tensor.deinit();

    // Set some values
    try tensor.set_at([_]usize{ 1, 2 }, 42);
    const val = try tensor.get_at([_]usize{ 1, 2 });

    std.debug.print("Tensor element at [1, 2] is {}\n", .{val});
}
