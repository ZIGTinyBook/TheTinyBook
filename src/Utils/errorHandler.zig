const std = @import("std");

/// Los function errors
pub const LossError = error{
    SizeMismatch,
    ShapeMismatch,
    InvalidPrediction,
};

/// Layer errors
pub const LayerError = error{
    NullLayer,
    InvalidParameters,
    InvalidLayerType,
};

/// Type errors
pub const TypeError = error{
    UnsupportedType,
};

/// Architecture errors
pub const ArchitectureError = error{
    UnknownArchitecture,
    UnderDevelopementArchitecture,
};

/// Tensor Math errors
pub const TensorMathError = error{
    MemError,
    InputTensorDifferentSize,
    InputTensorDifferentShape,
    InputTensorsWrongShape, //launched in dot_product
    OutputTensorDifferentSize,
    TooSmallOutputType, //the type dimension of the output Tensor could coause a loss of information
    InputTensorDimensionMismatch,
};

/// Tensor errors
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
    InvalidSliceIndices,
    InvalidSliceShape,
    SliceOutOfBounds,
    InvalidIndices,
};

/// A union type to represent any of the errors
pub const ErrorUnion = union(enum) {
    Loss: LossError,
    Layer: LayerError,
    Type: TypeError,
    Architecture: ArchitectureError,
    TensorMath: TensorMathError,
    Tensor: TensorError,
};

/// Function that returns the description of each error
/// #parameters:
///    myError: any error in this class, not ErrorUnion
/// #example of usage:
///    t1 and t2 Tensors,
///    _ = TensMath.dot_product_tensor(Architectures.CPU, f32, f64, &t1, &t2) catch |err| {
///        std.debug.print("\n _______ {s} ______", .{ErrorHandler.errorDetails(err)});
///    };
pub fn errorDetails(myError: anyerror) []const u8 {
    return switch (myError) {
        //LOSSS
        LossError.SizeMismatch => "Loss: size mismatch between expected and actual",
        LossError.ShapeMismatch => "Loss: shape mismatch between tensors",
        LossError.InvalidPrediction => "Loss: invalid prediction value",

        //LAYER
        LayerError.NullLayer => "Layer: null layer encountered",
        LayerError.InvalidParameters => "Layer: invalid parameters specified",

        //TYPE
        TypeError.UnsupportedType => "the Type you choose is not supported by this method/class",

        //ARCHITECTURE
        ArchitectureError.UnknownArchitecture => "Architecture: unknown architecture specified",
        ArchitectureError.UnderDevelopementArchitecture => "Architecture: architecture under development",

        //TENSORMATH
        TensorMathError.MemError => "TensorMath: memory error encountered",
        TensorMathError.InputTensorDifferentSize => "TensorMath: input tensor size mismatch",
        TensorMathError.InputTensorDifferentShape => "TensorMath: input tensor shape mismatch",
        TensorMathError.InputTensorsWrongShape => "TensorMath: input tensors have incompatible shapes",
        TensorMathError.OutputTensorDifferentSize => "TensorMath: output tensor size mismatch",
        TensorMathError.TooSmallOutputType => "TensorMath: output tensor type may lose information",
        TensorMathError.InputTensorDimensionMismatch => "TensorMath: input tensor dimension mismatch",

        //TENSOR
        TensorError.TensorNotInitialized => "Tensor: tensor not initialized",
        TensorError.InputArrayWrongType => "Tensor: input array has wrong type",
        TensorError.InputArrayWrongSize => "Tensor: input array size mismatch",
        TensorError.EmptyTensor => "Tensor: empty tensor",
        TensorError.ZeroSizeTensor => "Tensor: tensor has zero size",
        TensorError.NotOneHotEncoded => "Tensor: tensor not one-hot encoded",
        TensorError.NanValue => "Tensor: NaN value in tensor",
        TensorError.NotFiniteValue => "Tensor: tensor has non-finite value",
        TensorError.NegativeInfValue => "Tensor: tensor has negative infinity value",
        TensorError.PositiveInfValue => "Tensor: tensor has positive infinity value",

        else => "Unknown error type",
    };
}
