const std = @import("std");
const cwd = std.fs.cwd();
const Model = @import("model").Model;
const Layer = @import("layers");
const LayerType = @import("layers").LayerType;
const Tensor = @import("tensor").Tensor;
const ActivationType = @import("activation_function").ActivationType;

pub fn exportModel(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    model: Model(T, allocator),
    file_path: []const u8,
) !void {
    var file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();

    const writer = file.writer();
}

pub fn exportLayer(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    layer: Layer.Layer(T, allocator),
    file_path: []const u8,
) !void {
    var file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();

    const writer = file.writer();

    //TODO: handle Default layer and null layer
    if (layer.layer_type == LayerType.DenseLayer) {
        try writer.write("Dense.....");
        try exportLayerDense(T, layer.layer_ptr.*);
    } else if (layer.layer_type == LayerType.ActivationLayer) {
        try writer.write("Activation");
        //try exportLayerActivation();
    }
}

pub fn exportLayerDense(comptime T: type, allocator: *const std.mem.Allocator, layer: Layer.DenseLayer(T, allocator), writer: std.fs.File.Writer) !void {
    try exportTensor(T, layer.weights, allocator);
    try exportTensor(T, layer.bias, allocator);
    try exportTensor(T, layer.input, allocator);
    try exportTensor(T, layer.output, allocator);
    try writer.writeInt(usize, layer.n_inputs, std.builtin.Endian.big);
    try writer.writeInt(usize, layer.n_neurons, std.builtin.Endian.big);
    try exportTensor(T, layer.w_gradients, allocator);
    try exportTensor(T, layer.b_gradients, allocator);
}

pub fn exportLayerActivation(comptime T: type, allocator: *const std.mem.Allocator, layer: Layer.ActivationLayer(T, allocator), writer: std.fs.File.Writer) !void {
    try writer.writeInt(usize, layer.n_inputs, std.builtin.Endian.big);
    try writer.writeInt(usize, layer.n_neurons, std.builtin.Endian.big);
    try exportTensor(T, layer.input, allocator);
    try exportTensor(T, layer.output, allocator);

    if (layer.activationFunction == ActivationType.ReLU) {
        try writer.write("ReLU......");
    } else if (layer.activationFunction == ActivationType.Sigmoid) {
        try writer.write("Sigmoid...");
    } else if (layer.activationFunction == ActivationType.Softmax) {
        try writer.write("Softmax...");
    } else if (layer.activationFunction == ActivationType.None) {
        try writer.write("None......");
    }
}

pub fn exportTensor(comptime T: type, tensor: Tensor(T), writer: std.fs.File.Writer) !void {

    // Write tensor size and shape
    try writer.writeInt(usize, tensor.size, std.builtin.Endian.big);
    try writer.writeInt(usize, tensor.shape.len, std.builtin.Endian.big);
    for (tensor.shape) |dim| {
        try writer.writeInt(usize, dim, std.builtin.Endian.big);
    }

    // Write tensor data
    for (tensor.data) |value| {
        try writer.writeInt(T, value, std.builtin.Endian.big);
    }
}

pub fn importTensor(allocator: *const std.mem.Allocator, comptime T: type, file_path: []const u8) !Tensor(T) {
    var file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const reader = file.reader();

    // Read tensor size
    const tensor_size: usize = try reader.readInt(usize, std.builtin.Endian.big);

    // Read tensor shape lenght
    const tensor_shapeLen: usize = try reader.readInt(usize, std.builtin.Endian.big);

    // Read tensor shape
    const tensor_shape = try allocator.alloc(usize, tensor_shapeLen);
    for (0..tensor_shapeLen) |i| {
        tensor_shape[i] = try reader.readInt(usize, std.builtin.Endian.big);
    }

    const tensor_data = try allocator.alloc(T, tensor_size);
    // Read tensor data
    for (0..tensor_size) |i| {
        tensor_data[i] = try reader.readInt(T, std.builtin.Endian.big);
    }

    return Tensor(T){
        .data = tensor_data,
        .size = tensor_size,
        .shape = tensor_shape,
        .allocator = allocator,
    };
}
