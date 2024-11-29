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

    try writer.writeInt(usize, model.layers.items.len, std.builtin.Endian.big);
    for (model.layers.items) |*l| {
        try exportLayer(T, allocator, l.*, writer);
    }
    return;
}

pub fn exportLayer(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    layer: Layer.Layer(T, allocator),
    writer: std.fs.File.Writer,
) !void {

    //TODO: handle Default layer and null layer
    if (layer.layer_type == LayerType.DenseLayer) {
        try writer.write("Dense.....");
        try exportLayerDense(T, allocator, layer.layer_ptr.*, writer);
    } else if (layer.layer_type == LayerType.ActivationLayer) {
        try writer.write("Activation");
        try exportLayerActivation(T, allocator, layer.layer_ptr.*, writer);
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

pub fn importModel(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    file_path: []const u8,
) Model(T, allocator) {
    var file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();
    const reader = file.reader();

    var model: Model(T, allocator) = Model(T, allocator){
        .layers = undefined,
        .allocator = &allocator,
        .input_tensor = undefined,
    };
    try model.init();

    const n_layers = try reader.readInt(usize, std.builtin.Endian.big);
    for (0..n_layers) |i| {
        const newLayer: Layer.Layer(T, allocator) = try importLayer(T, allocator, l.*, writer);
        model.addLayer(newLayer);
    }
    return;
}

pub fn importLayer(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !Layer.Layer(T, allocator) {
    const layer_type_string: [10]u8 = undefined;
    try reader.read(layer_type_string);

    //TODO: handle Default layer and null layer
    if (std.mem.eql(u8, layer_type_string, "Dense.....")) {
        const denseLayer: Layer.DenseLayer(T, allocator) = try importLayerDense(T, allocator, reader);
        return Layer.DenseLayer(f64, &allocator).create(&denseLayer);
    } else if (std.mem.eql(u8, layer_type_string, "Activation")) {
        //const denseLayer: Layer.ActivationLayer(T, allocator) = try importLayerActivation(T, allocator, reader);
    }
}

pub fn importLayerDense(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !Layer.DenseLayer(T, allocator) {
    const weights_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const bias_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const input_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const output_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const n_inputs = try reader.readInt(usize, std.builtin.Endian.big);
    const n_neurons = try reader.readInt(usize, std.builtin.Endian.big);
    const w_grad_tens = try importTensor(T, allocator, reader);
    const b_grad_tens = try importTensor(T, allocator, reader);

    return Layer.DenseLayer(f64, allocator){
        .weights = weights_tens,
        .bias = bias_tens,
        .input = input_tens,
        .output = output_tens,
        .n_inputs = n_inputs,
        .n_neurons = n_neurons,
        .w_gradients = w_grad_tens,
        .b_gradients = b_grad_tens,
        .allocator = allocator,
    };
}

pub fn importLayerActivation(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !Layer.ActivationLayer(T, allocator) {
    const n_inputs = try reader.readInt(usize, std.builtin.Endian.big);
    const n_neurons = try reader.readInt(usize, std.builtin.Endian.big);

    const input_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const output_tens: Tensor(T) = try importTensor(T, allocator, reader);

    const activation_type_string: [10]u8 = undefined;
    try reader.read(activation_type_string);

    const layerActiv = Layer.ActivationLayer(T, allocator){
        .input = input_tens,
        .output = output_tens,
        .n_inputs = n_inputs,
        .n_neurons = n_neurons,
        .activationFunction = undefined,
        .allocator = allocator,
    };

    if (std.mem.eql(u8, activation_type_string, "ReLU......")) {
        layerActiv.activationFunction = ActivationType.ReLU;
    } else if (std.mem.eql(u8, activation_type_string, "Sigmoid...")) {
        layerActiv.activationFunction = ActivationType.Sigmoid;
    } else if (std.mem.eql(u8, activation_type_string, "Softmax...")) {
        layerActiv.activationFunction = ActivationType.Softmax;
    } else if (std.mem.eql(u8, activation_type_string, "None......")) {
        layerActiv.activationFunction = ActivationType.None;
    }

    return layerActiv;
}

pub fn importTensor(comptime T: type, allocator: *const std.mem.Allocator, reader: std.fs.File.Reader) !Tensor(T) {

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
