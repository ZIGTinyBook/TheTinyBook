//! Tensor has a crucial role in all the project. Is the foundamental class around witch everything
//! is constructed. A tensor is a multi-dimensional array or a mathematical object that generalizes
//! the concept of scalars, vectors, and matrices to higher dimensions. A scalar is a 0-dimensional
//! tensor, a vector is a 1-dimensional tensor, and a matrix is a 2-dimensional tensor. Tensors can extend
//! to even higher dimensions (3D, 4D, etc.).

const std = @import("std");
const tMath = @import("tensor_m");
const Architectures = @import("architectures").Architectures;

pub const TensorError = error{
    TensorNotInitialized,
    InputArrayWrongType,
    InputArrayWrongSize,
    EmptyTensor,
    ZeroSizeTensor,
    NotOneHotEncoded,
    NanValue,
    NotFiniteValue,
    NegativeInfValue,
    PositiveInfValue,
};

///Class Tensor.
///Return a generic type structure
pub fn Tensor(comptime T: type) type {
    return struct {
        data: []T, //contains all the data of the tensor in a monodimensional array
        size: usize, //dimension of the tensor, equal to data.len
        shape: []usize, //defines the multidimensional structure of the tensor
        allocator: *const std.mem.Allocator, //allocator used in the memory initialization of the tensor

        ///Method used to initialize an undefined Tensor. It just set the allocator.
        /// More usefull methods are:
        ///  - fromArray()
        ///  - copy()
        ///  - fromShape()
        pub fn init(allocator: *const std.mem.Allocator) !@This() {
            return @This(){
                .data = &[_]T{},
                .size = 0,
                .shape = &[_]usize{},
                .allocator = allocator,
            };
        }

        ///Free all the possible allocation, use it every time you create a new Tensor ( defer yourTensor.deinit() )
        pub fn deinit(self: *@This()) void {
            if (self.size > 0) {
                if (self.data.len > 0) {
                    self.allocator.free(self.data);
                    self.data = &[_]T{};
                }
                if (self.shape.len > 0) {
                    self.allocator.free(self.shape);
                    self.shape = &[_]usize{};
                }
            }
        }

        ///Given a multidimensional array with its shape, returns the equivalent Tensor.
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn fromArray(allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !@This() {

            // Calculate total size based on shape
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }

            // Allocate memory for tensor shape
            const tensorShape = try allocator.alloc(usize, shape.len);
            @memcpy(tensorShape, shape);

            // Allocate memory for tensor data
            const tensorData = try allocator.alloc(T, total_size);

            // Flatten the input array into tensor data
            _ = flattenArray(T, inputArray, tensorData, 0);

            // Return the new tensor
            return @This(){
                .data = tensorData,
                .size = total_size,
                .shape = tensorShape,
                .allocator = allocator,
            };
        }

        /// Given the Tensor (self) returns the equivalent multidimensional array.
        /// See constructMultidimensionalArray() in this file.
        /// IMPORTANT: Remember to cal yourAllocator.free(yourMultidimArray) otherwise it generates a memory leak!
        pub fn toArray(self: @This(), comptime dimension: usize) !MagicalReturnType(T, dimension) {
            if (dimension == 1) {
                return self.data;
            }
            return constructMultidimensionalArray(self.allocator, T, self.data, self.shape, 0, dimension);
        }

        /// Returns a Tensor witch is the copy of this Tensor (self).
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn copy(self: *@This()) !Tensor(T) {
            return try Tensor(T).fromArray(self.allocator, self.data, self.shape);
        }

        /// Return a all-zero tensor starting from the given shape
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn fromShape(allocator: *const std.mem.Allocator, shape: []usize) !@This() {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }

            const tensorShape = try allocator.alloc(usize, shape.len);
            @memcpy(tensorShape, shape);

            const tensorData = try allocator.alloc(T, total_size);
            for (tensorData) |*data| {
                data.* = 0;
            }

            return @This(){
                .data = tensorData,
                .size = total_size,
                .shape = tensorShape,
                .allocator = allocator,
            };
        }

        /// Given any array and its shape it reshape the tensor and update .data
        pub fn fill(self: *@This(), inputArray: anytype, shape: []usize) !void {

            //deinitialize data e shape
            self.deinit(); //if the Tensor has been just init() this function does nothing

            //than, filling with the new values
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }
            const tensorShape = try self.allocator.alloc(usize, shape.len);
            @memcpy(tensorShape, shape);

            const tensorData = try self.allocator.alloc(T, total_size);
            _ = flattenArray(T, inputArray, tensorData, 0);

            self.data = tensorData;
            self.size = total_size;
            self.shape = tensorShape;
        }

        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///--------------------------------------------------------------------------getters and setters---------------------------------------------------------------
        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///Set the shape of a Tensor.
        pub fn setShape(self: *@This(), shape: []usize) !void {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }
            self.shape = shape;
            self.size = total_size;
        }

        ///Returns the size of the Tensor.
        pub fn getSize(self: *@This()) usize {
            return self.size;
        }

        ///Given an index, return the value at self.data[index].
        /// Errors:
        ///     - error.IndexOutOfBounds;
        pub fn get(self: *const @This(), idx: usize) !T {
            if (idx >= self.data.len) {
                return error.IndexOutOfBounds;
            }
            return self.data[idx];
        }

        ///Set to value the data at self.data[idx].
        /// Errors:
        ///     - error.IndexOutOfBounds;
        pub fn set(self: *@This(), idx: usize, value: T) !void {
            if (idx >= self.data.len) {
                return error.IndexOutOfBounds;
            }
            self.data[idx] = value;
        }

        /// Given the coordinates (indices) it returns the correspondant value in the
        /// multidimensional array.
        /// See flatten_index().
        pub fn get_at(self: *const @This(), indices: []const usize) !T {
            const idx = try self.flatten_index(indices);
            return self.get(idx);
        }

        /// Given the the value and the coordinates (indices), it sets the value in
        /// the multidimensional array at the specified coordinates.
        /// See flatten_index().
        pub fn set_at(self: *@This(), indices: []const usize, value: T) !void {
            const idx = try self.flatten_index(indices);
            return self.set(idx, value);
        }

        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///-------------------------------------------------------------------------------------utils------------------------------------------------------------------
        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///Starting from the monodimensional array self.data and the shape self.shape, it returns the equivalent multidimensional array
        fn constructMultidimensionalArray(
            allocator: *const std.mem.Allocator,
            comptime ElementType: type,
            data: []ElementType,
            shape: []usize,
            comptime depth: usize,
            comptime dimension: usize,
        ) !MagicalReturnType(ElementType, dimension - depth) {
            if (depth == dimension - 1) {
                return data;
            }

            const current_dim = shape[depth];
            var result = try allocator.alloc(
                MagicalReturnType(ElementType, dimension - depth - 1),
                current_dim,
            );

            // defer allocator.free(result); ??????????? MARCO : era già commentata, ci va o meno la .free()? non credo vada liberato perchè è lui stesso l'array multidim.
            // non andrebbe però creato un metodo freeMultidimensionalArray() che fa la stessa cosa ma librando spazio?
            // AGGIORANEMENTO: nei tests_tensor mi è bastato fare: line 197 -> defer allocator.free(array_from_tensor);

            var offset: usize = 0;
            const sub_array_size = calculateProduct(shape[(depth + 1)..]);

            for (0..current_dim) |i| {
                result[i] = try constructMultidimensionalArray(
                    allocator,
                    ElementType,
                    data[offset .. offset + sub_array_size],
                    shape,
                    depth + 1,
                    dimension,
                );
                offset += sub_array_size;
            }

            return result;
        }

        fn MagicalReturnType(comptime DataType: type, comptime dim_count: usize) type {
            return if (dim_count == 1) []DataType else []MagicalReturnType(DataType, dim_count - 1);
        }

        fn calculateProduct(slice: []usize) usize {
            var product: usize = 1;
            for (slice) |elem| {
                product *= elem;
            }
            return product;
        }

        /// Modify, if possible, the shape of a tensor, use it wisely.
        /// Errors:
        ///     - TensorError.InputArrayWrongSize
        pub fn reshape(self: *@This(), shape: []usize) !void {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }
            if (total_size != self.size) {
                return TensorError.InputArrayWrongSize;
            }

            self.allocator.free(self.shape);
            const tensorShape = try self.allocator.alloc(usize, shape.len);
            // copy elements of shape
            @memcpy(tensorShape, shape);

            self.shape = tensorShape;
        }

        /// Given the coordinates (indices) of a multidimensional Tensor returns the correspondant potition in the monodimensional space of self.data
        pub fn flatten_index(self: *const @This(), indices: []const usize) !usize {
            var idx: usize = 0;
            var stride: usize = 1;
            for (0..self.shape.len) |i| {
                idx += indices[self.shape.len - 1 - i] * stride;
                stride *= self.shape[self.shape.len - 1 - i];
            }
            return idx;
        }

        /// Prints all the possible details of a tensor.
        /// Very usefull in debugging.
        pub fn info(self: *@This()) void {
            std.debug.print("\ntensor infos: ", .{});
            std.debug.print("\n  data type:{}", .{@TypeOf(self.data[0])});
            std.debug.print("\n  size:{}", .{self.size});
            std.debug.print("\n shape.len:{} shape: [ ", .{self.shape.len});
            for (0..self.shape.len) |i| {
                std.debug.print("{} ", .{self.shape[i]});
            }
            std.debug.print("] ", .{});
            self.print();
        }

        /// Prints all the array self.data in an array.
        pub fn print(self: *@This()) void {
            std.debug.print("\n  tensor data: ", .{});
            for (0..self.size) |i| {
                std.debug.print("{} ", .{self.data[i]});
            }
            std.debug.print("\n", .{});
        }

        /// Print the Tensor() in the shape of a matrix
        pub fn printMultidim(self: *@This()) void {
            const dim = self.shape.len;
            for (0..self.shape[dim - 2]) |i| {
                std.debug.print("\n[ ", .{});
                for (0..self.shape[dim - 1]) |j| {
                    std.debug.print("{} ", .{self.data[i * self.shape[dim - 1] + j]});
                }
                std.debug.print("]", .{});
            }
        }

        /// Returns a Tensor self transposed. Does not modify self.
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn transpose2D(self: *@This()) !Tensor(T) {
            if (self.shape.len != 2) {
                return error.InvalidDimension; // For simplicity, let's focus on 2D for now
            }

            const allocator = self.allocator;

            // Shape of the transposed tensor
            const transposed_shape: [2]usize = [_]usize{ self.shape[1], self.shape[0] };
            const tensorShape = try allocator.alloc(usize, self.shape.len);
            @memcpy(tensorShape, &transposed_shape);

            // Allocate space for transposed data
            const transposed_data = try allocator.alloc(T, self.size);

            // Perform the transposition
            for (0..self.shape[0]) |i| {
                for (0..self.shape[1]) |j| {
                    // For 2D tensor, flatten the index and swap row/column positions
                    const old_idx = i * self.shape[1] + j;
                    const new_idx = j * self.shape[0] + i;
                    transposed_data[new_idx] = self.data[old_idx];
                }
            }

            return Tensor(T){
                .data = transposed_data,
                .size = self.size,
                .shape = tensorShape,
                .allocator = allocator,
            };
        }

        /// Returns true if the Tensor is one-hot encoded
        fn isOneHot(self: *@This()) !bool {
            const elems_row = self.shape[self.shape.len - 1];
            if (elems_row == 0) {
                return TensorError.EmptyTensor;
            }
            const numb_rows = self.size / elems_row;
            if (numb_rows == 0) {
                return TensorError.ZeroSizeTensor;
            }

            for (0..numb_rows) |row| {
                var oneHotFound = false;
                for (0..self.shape[self.shape.len - 1]) |i| {
                    if (self.data[row * elems_row + i] == 1 and !oneHotFound) {
                        if (!oneHotFound) oneHotFound = true else return TensorError.NotOneHotEncoded;
                    }
                }
            }

            return true;
        }

        /// Returns true only if all the values of shape and data are valid numbers
        pub fn isSafe(self: *@This()) !void {
            switch (@typeInfo(T)) {
                .Float => {
                    // Loop over tensor data
                    for (self.data) |*value| {
                        if (std.math.isNan(value.*)) return TensorError.NanValue;
                        if (!std.math.isFinite(value.*)) return TensorError.NotFiniteValue;
                    }

                    // Loop over tensor shape
                    for (self.shape) |*value| {
                        if (std.math.isNan(value.*)) return TensorError.NanValue;
                    }
                },
                else => {
                    // If T is not Float, skip isSafe checks
                    return;
                },
            }
        }
    };
}

/// Recursive function to flatten a multidimensional array
fn flattenArray(comptime T: type, arr: anytype, flatArr: []T, startIndex: usize) usize {
    var idx = startIndex;

    const arrTypeInfo = @typeInfo(@TypeOf(arr));

    // Check if arr is an Array or a Slice
    if (arrTypeInfo == .Array or arrTypeInfo == .Pointer) {
        if (@TypeOf(arr[0]) == T) {
            // If arr is a 1D array or slice
            for (arr) |val| {
                flatArr[idx] = val;
                idx += 1;
            }
        } else {
            // If arr is multidimensional, recursively flatten
            for (arr) |subArray| {
                idx = flattenArray(T, subArray, flatArr, idx);
            }
        }
    } else {
        std.debug.print("The type of `arr` is not compatible with the required type. Type found: {}\n", .{@TypeOf(arr)});
        @panic("The type of `arr` is not compatible with the required type.");
    }

    return idx;
}
