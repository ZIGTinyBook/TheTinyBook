const std = @import("std");
const tMath = @import("./tensor_math.zig");
const Architectures = @import("./architectures.zig").Architectures; //Import Architectures type

pub const TensorError = error{
    TensorNotInitialized,
    InputArrayWrongType,
    InputArrayWrongSize,
    EmptyTensor,
    ZeroSizeTensor,
};

pub fn Tensor(comptime T: type) type {
    return struct {
        data: []T,
        size: usize,
        shape: []usize,
        allocator: *const std.mem.Allocator,

        pub fn init(allocator: *const std.mem.Allocator) !@This() {
            return @This(){
                .data = &[_]T{},
                .size = 0,
                .shape = &[_]usize{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            //std.debug.print("\n deinit tensor:\n", .{});
            // Verifica se `data` è valido e non vuoto prima di liberarlo
            if (self.data.len > 0) {
                //std.debug.print("Liberazione di data con lunghezza: {}\n", .{self.data.len});
                self.allocator.free(self.data);
                self.data = &[_]T{}; // Resetta lo slice a vuoto
            }
            // Verifica se `shape` è valido e non vuoto prima di liberarlo
            if (self.shape.len > 0) {
                //std.debug.print("Liberazione di shape con lunghezza: {}\n", .{self.shape.len});
                self.allocator.free(self.shape);
                self.shape = &[_]usize{}; // Resetta lo slice a vuoto
            }
        }

        pub fn fromArray(allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !@This() {
            //std.debug.print("\n fromArray initialization...", .{});
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }
            const tensorShape = try allocator.alloc(usize, shape.len);
            @memcpy(tensorShape, shape);

            const tensorData = try allocator.alloc(T, total_size);
            _ = flattenArray(T, inputArray, tensorData, 0);

            return @This(){
                .data = tensorData,
                .size = total_size,
                .shape = tensorShape,
                .allocator = allocator,
            };
        }

        fn MagicalReturnType(comptime DataType: type, comptime dim_count: usize) type {
            return if (dim_count == 1) []DataType else []MagicalReturnType(DataType, dim_count - 1);
        }

        pub fn toArray(self: @This(), comptime dimension: usize) !MagicalReturnType(T, dimension) {
            if (dimension == 1) {
                return self.data;
            }
            return constructMultidimensionalArray(self.allocator, T, self.data, self.shape, 0, dimension);
        }

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

            //defer allocator.free(result);

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

        fn calculateProduct(slice: []usize) usize {
            var product: usize = 1;
            for (slice) |elem| {
                product *= elem;
            }
            return product;
        }

        //copy self and return it in another Tensor
        pub fn copy(self: *@This()) !Tensor(T) {
            return try Tensor(T).fromArray(self.allocator, self.data, self.shape);
        }

        //inizialize and return a all-zero tensor starting from the shape
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

        pub fn reshape(self: *@This(), shape: []usize) !void {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }
            if (total_size != self.size) {
                return TensorError.InputArrayWrongSize;
            }
            //use cycle to copy elements of shape
            for (shape, 0..) |dim, i| {
                self.shape[i] = dim;
            }
        }

        //pay attention, the fill() can also perform a reshape
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

        pub fn setShape(self: *@This(), shape: []usize) !void {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }
            self.shape = shape;
            self.size = total_size;
        }

        pub fn getSize(self: *@This()) usize {
            return self.size;
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
            for (0..self.shape.len) |i| {
                idx += indices[self.shape.len - 1 - i] * stride;
                stride *= self.shape[self.shape.len - 1 - i];
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
            std.debug.print("\n shape.len:{} shape: [ ", .{self.shape.len});
            for (0..self.shape.len) |i| {
                std.debug.print("{} ", .{self.shape[i]});
            }
            std.debug.print("] ", .{});
            self.print();
        }

        pub fn print(self: *@This()) void {
            std.debug.print("\n  tensor data: ", .{});
            for (0..self.size) |i| {
                std.debug.print("{} ", .{self.data[i]});
            }
            std.debug.print("\n", .{});
        }

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

    // Inizializzazione degli array di input
    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{ 1, 2, 3 },
        [_]u8{ 4, 5, 6 },
    };

    var inputArray2: [2][3]u8 = [_][3]u8{
        [_]u8{ 6, 5, 4 },
        [_]u8{ 3, 2, 1 },
    };

    var inputArray3: [2][3]i32 = [_][3]i32{
        [_]i32{ 6, 5, 4 },
        [_]i32{ 3, 2, 1 },
    };

    var shape: [2]usize = [_]usize{ 2, 3 };
    var shape2: [2]usize = [_]usize{ 3, 2 };

    // Creazione e gestione del primo tensore (tensor)
    var tensor = try Tensor(u8).fromArray(&allocator, &inputArray, &shape);
    defer tensor.deinit();
    std.debug.print("Tensor 1 Info:\n", .{});
    tensor.info();

    // Creazione e gestione del secondo tensore (tensor2)
    var tensor2 = try Tensor(u8).fromArray(&allocator, &inputArray2, &shape);
    defer tensor2.deinit();
    std.debug.print("Tensor 2 Info:\n", .{});
    tensor2.info();

    // Creazione e gestione del terzo tensore (tensor3)
    var tensor3 = try Tensor(i32).fromArray(&allocator, &inputArray3, &shape);
    defer tensor3.deinit();
    std.debug.print("Tensor 3 Info:\n", .{});
    tensor3.info();

    // Inizializzazione di un tensore vuoto (tensor4)
    var tensor4 = try Tensor(u8).init(&allocator);
    defer tensor4.deinit();
    std.debug.print("Empty Tensor 4 Info:\n", .{});
    tensor4.info();

    // Test della funzione toArray
    std.debug.print("\nTesting toArray on Tensor 1:\n", .{});
    const array_from_tensor = tensor.toArray(2);
    std.debug.print("Array extracted from tensor: {any}\n", .{array_from_tensor});

    std.debug.print("\nTesting reshape 2:\n", .{});
    try tensor.reshape(&shape2);
    const array_from_tensor_R = tensor.toArray(2);
    std.debug.print("Array extracted from tensor after reshape: {any}\n", .{array_from_tensor_R});

    std.debug.print("\nTesting toArray on Tensor 3:\n", .{});
    const array_from_tensor3 = tensor3.toArray(2);
    std.debug.print("Array extracted from tensor3: {any}\n", .{array_from_tensor3});

    // È possibile inserire ulteriori operazioni sui tensori come somma o prodotto scalare
    // Prova con una somma di tensori (commentata per ora):
    // try tMath.sum_tensors(Architectures.CPU, u8, i32, &tensor, &tensor2, &tensor3);
    // tensor3.info();

    // Prova con un prodotto scalare (commentato per ora):
    // const tensor5 = try tMath.dot_product_tensor(Architectures.CPU, u8, i32, &tensor, &tensor4);
    // defer tensor5.deinit();
    // tensor5.info();
    // tensor5.print();
}
