const std = @import("std");
const cwd = std.fs.cwd();
const Model = @import("model").Model;
const Layer = @import("layers");
const LayerType = @import("layers").LayerType;
const Tensor = @import("tensor").Tensor;
const ActivationType = @import("activation_function").ActivationType;

// ----------------- Export -----------------
pub fn exportModel(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    model: Model(T, allocator),
    file_path: []const u8,
) !void {
    std.debug.print("\n ..... EXPORTING THE MODEL ......", .{});
    var file = try std.fs.cwd().createFile(file_path, .{});
    std.debug.print("\n ..... file created ......", .{});
    defer file.close();

    const writer = file.writer();
    std.debug.print("\n ..... writer created ......", .{});

    try writer.writeInt(usize, model.layers.items.len, std.builtin.Endian.big);
    for (model.layers.items) |*l| {
        try exportLayer(T, allocator, l.*, writer);
    }
    return;
}

pub fn exportLayer(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    layer: Layer.Layer(T, allocator),
    writer: std.fs.File.Writer,
) !void {
    std.debug.print("\n ... export layer... ", .{});

    //TODO: handle Default layer and null layer
    if (layer.layer_type == LayerType.DenseLayer) {
        _ = try writer.write("Dense.....");
        // const denseLayer: *Layer.DenseLayer(T, allocator) = @ptrCast(layer.layer_ptr); //cast to : *Layer.DenseLayer(T, allocator)
        // try exportLayerDense(T, allocator, denseLayer.*, writer);
        const denseLayer: *Layer.DenseLayer(T, allocator) = @alignCast(@ptrCast(layer.layer_ptr));
        try exportLayerDense(T, allocator, denseLayer.*, writer);
    } else if (layer.layer_type == LayerType.ActivationLayer) {
        _ = try writer.write("Activation");
        const activationLayer: *Layer.ActivationLayer(T, allocator) = @alignCast(@ptrCast(layer.layer_ptr));
        try exportLayerActivation(T, allocator, activationLayer.*, writer);
    }
}

pub fn exportLayerDense(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    layer: Layer.DenseLayer(T, allocator),
    writer: std.fs.File.Writer,
) !void {
    std.debug.print(" dense ", .{});

    try exportTensor(T, layer.weights, writer);
    try exportTensor(T, layer.bias, writer);
    try exportTensor(T, layer.input, writer);
    try exportTensor(T, layer.output, writer);
    try writer.writeInt(usize, layer.n_inputs, std.builtin.Endian.big);
    try writer.writeInt(usize, layer.n_neurons, std.builtin.Endian.big);
    try exportTensor(T, layer.w_gradients, writer);
    try exportTensor(T, layer.b_gradients, writer);
}

pub fn exportLayerActivation(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    layer: Layer.ActivationLayer(T, allocator),
    writer: std.fs.File.Writer,
) !void {
    std.debug.print(" activation ", .{});

    try writer.writeInt(usize, layer.n_inputs, std.builtin.Endian.big);
    try writer.writeInt(usize, layer.n_neurons, std.builtin.Endian.big);
    try exportTensor(T, layer.input, writer);
    try exportTensor(T, layer.output, writer);

    if (layer.activationFunction == ActivationType.ReLU) {
        _ = try writer.write("ReLU......");
    } else if (layer.activationFunction == ActivationType.Sigmoid) {
        _ = try writer.write("Sigmoid...");
    } else if (layer.activationFunction == ActivationType.Softmax) {
        _ = try writer.write("Softmax...");
    } else if (layer.activationFunction == ActivationType.None) {
        _ = try writer.write("None......");
    }
}

pub fn exportTensor(
    comptime T: type,
    tensor: Tensor(T),
    writer: std.fs.File.Writer,
) !void {
    std.debug.print("\n ... export tensor... ", .{});

    // Write tensor size and shape
    try writer.writeInt(usize, tensor.size, std.builtin.Endian.big);
    try writer.writeInt(usize, tensor.shape.len, std.builtin.Endian.big);
    for (tensor.shape) |dim| {
        try writer.writeInt(usize, dim, std.builtin.Endian.big);
    }

    // Write tensor data
    for (tensor.data) |value| {
        try writeNumber(T, value, writer);
    }
}

pub fn writeNumber(
    comptime T: type,
    number: T,
    writer: std.fs.File.Writer,
) !void {
    const size = @sizeOf(T);
    var buffer: [size]u8 = @bitCast(number);
    try writer.writeAll(&buffer);
}

// ----------------- Import -----------------
pub fn importModel(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    file_path: []const u8,
) !Model(T, allocator) {
    std.debug.print("\n ..... IMPORTING THE MODEL ......", .{});

    var file = try std.fs.cwd().openFile(file_path, .{});
    std.debug.print("\n ..... file created ......", .{});

    defer file.close();
    const reader = file.reader();
    std.debug.print("\n ..... reader created ......", .{});

    var model: Model(T, allocator) = Model(T, allocator){
        .layers = undefined,
        .allocator = allocator,
        .input_tensor = undefined,
    };
    try model.init();

    const n_layers = try reader.readInt(usize, std.builtin.Endian.big);
    for (0..n_layers) |_| {
        const newLayer: Layer.Layer(T, allocator) = try importLayer(T, allocator, reader);
        // std.debug.print("\n ..... {any}......", .{newLayer.get_n_inputs()});
        // std.debug.print("\n ..... {any}......", .{newLayer.layer_ptr});

        newLayer.printLayer(1);
        try model.addLayer(newLayer);
    }
    return model;
}

pub fn importLayer(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !Layer.Layer(T, allocator) {
    std.debug.print("\n ... import layer... ", .{});

    var layer_type_string: [10]u8 = undefined;
    _ = try reader.read(&layer_type_string);

    //TODO: handle Default layer and null layer
    if (std.mem.eql(u8, &layer_type_string, "Dense.....")) {

        // Dynamically allocate memory for the DenseLayer
        const denseLayerPtr = try allocator.create(Layer.DenseLayer(T, allocator));
        denseLayerPtr.* = try importLayerDense(T, allocator, reader);

        const newLayer: Layer.Layer(T, allocator) = Layer.DenseLayer(T, allocator).create(denseLayerPtr);
        return newLayer;
    } else if (std.mem.eql(u8, &layer_type_string, "Activation")) {
        var activationLayer: Layer.ActivationLayer(T, allocator) = try importLayerActivation(T, allocator, reader);
        return Layer.ActivationLayer(T, allocator).create(&activationLayer);
    } else {
        return error.impossibleLayer;
    }
}

pub fn importLayerDense(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !Layer.DenseLayer(T, allocator) {
    std.debug.print(" dense ", .{});

    // const weights_tens: Tensor(T) = try importTensor(T, allocator, reader);
    // const bias_tens: Tensor(T) = try importTensor(T, allocator, reader);
    // const input_tens: Tensor(T) = try importTensor(T, allocator, reader);
    // const output_tens: Tensor(T) = try importTensor(T, allocator, reader);
    // const n_inputs = try reader.readInt(usize, std.builtin.Endian.big);
    // const n_neurons = try reader.readInt(usize, std.builtin.Endian.big);
    // const w_grad_tens = try importTensor(T, allocator, reader);
    // const b_grad_tens = try importTensor(T, allocator, reader);

    // return Layer.DenseLayer(f64, allocator){
    //     .weights = weights_tens,
    //     .bias = bias_tens,
    //     .input = input_tens,
    //     .output = output_tens,
    //     .n_inputs = n_inputs,
    //     .n_neurons = n_neurons,
    //     .w_gradients = w_grad_tens,
    //     .b_gradients = b_grad_tens,
    //     .allocator = allocator,
    // };

    return Layer.DenseLayer(f64, allocator){
        .weights = try importTensor(T, allocator, reader),
        .bias = try importTensor(T, allocator, reader),
        .input = try importTensor(T, allocator, reader),
        .output = try importTensor(T, allocator, reader),
        .n_inputs = try reader.readInt(usize, std.builtin.Endian.big),
        .n_neurons = try reader.readInt(usize, std.builtin.Endian.big),
        .w_gradients = try importTensor(T, allocator, reader),
        .b_gradients = try importTensor(T, allocator, reader),
        .allocator = allocator,
    };
}

pub fn importLayerActivation(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !Layer.ActivationLayer(T, allocator) {
    std.debug.print(" activation ", .{});

    const n_inputs = try reader.readInt(usize, std.builtin.Endian.big);
    const n_neurons = try reader.readInt(usize, std.builtin.Endian.big);

    const input_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const output_tens: Tensor(T) = try importTensor(T, allocator, reader);

    var activation_type_string: [10]u8 = undefined;
    _ = try reader.read(&activation_type_string);

    var layerActiv = Layer.ActivationLayer(T, allocator){
        .input = input_tens,
        .output = output_tens,
        .n_inputs = n_inputs,
        .n_neurons = n_neurons,
        .activationFunction = undefined,
        .allocator = allocator,
    };

    if (std.mem.eql(u8, &activation_type_string, "ReLU......")) {
        layerActiv.activationFunction = ActivationType.ReLU;
    } else if (std.mem.eql(u8, &activation_type_string, "Sigmoid...")) {
        layerActiv.activationFunction = ActivationType.Sigmoid;
    } else if (std.mem.eql(u8, &activation_type_string, "Softmax...")) {
        layerActiv.activationFunction = ActivationType.Softmax;
    } else if (std.mem.eql(u8, &activation_type_string, "None......")) {
        layerActiv.activationFunction = ActivationType.None;
    }

    return layerActiv;
}

pub fn importTensor(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !Tensor(T) {
    std.debug.print("\n ... import tensor... ", .{});

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
        tensor_data[i] = try readNumber(T, reader);
    }

    return Tensor(T){
        .data = tensor_data,
        .size = tensor_size,
        .shape = tensor_shape,
        .allocator = allocator,
    };
}

inline fn readNumber(
    comptime T: type,
    reader: std.fs.File.Reader,
) !T {
    const size = @sizeOf(T);
    var buffer: [size]u8 = undefined;
    _ = try reader.readAll(&buffer);
    return @bitCast(buffer);
}
