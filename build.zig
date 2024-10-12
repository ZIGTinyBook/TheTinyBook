const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the main modules
    const tensor_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/tensor.zig") });
    const tensor_math_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/tensor_math.zig") });
    const architectures_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/architectures.zig") });

    const model_mod = b.createModule(.{ .root_source_file = b.path("src/Model/model.zig") });
    const layers_mod = b.createModule(.{ .root_source_file = b.path("src/Model/layers.zig") });
    const optim_mod = b.createModule(.{ .root_source_file = b.path("src/Model/optim.zig") });
    const dataloader_mod = b.createModule(.{ .root_source_file = b.path("src/DataLoader/dataLoader.zig") });
    const dataProcessor_mod = b.createModule(.{ .root_source_file = b.path("src/DataLoader/dataProcessor.zig") });
    const loss_mod = b.createModule(.{ .root_source_file = b.path("src/Model/lossFunction.zig") });
    const activation_mod = b.createModule(.{ .root_source_file = b.path("src/Model/activation_function.zig") });
    const typeC_mod = b.createModule(.{ .root_source_file = b.path("src/Utils/typeConverter.zig") });

    // Add only the necessary dependencies without duplications
    //
    //************************************************MODEL DEPENDENCIES************************************************

    model_mod.addImport("tensor", tensor_mod);
    model_mod.addImport("layers", layers_mod);
    model_mod.addImport("optim", optim_mod); // Do not remove duplicate
    model_mod.addImport("loss", loss_mod);
    model_mod.addImport("typeC", typeC_mod);
    model_mod.addImport("dataloader", dataloader_mod);
    model_mod.addImport("tensor_m", tensor_math_mod);
    model_mod.addImport("dataprocessor", dataProcessor_mod);
    model_mod.addImport("activation_function", activation_mod);

    //************************************************LAYER DEPENDENCIES************************************************

    layers_mod.addImport("tensor", tensor_mod);
    layers_mod.addImport("activation_function", activation_mod);
    layers_mod.addImport("tensor_m", tensor_math_mod);
    layers_mod.addImport("architectures", architectures_mod);

    //************************************************DATA LOADER DEPENDENCIES************************************************

    dataloader_mod.addImport("tensor", tensor_mod);

    //************************************************DATA PROCESSOR DEPENDENCIES************************************************

    dataProcessor_mod.addImport("tensor", tensor_mod);

    //************************************************TENSOR DEPENDENCIES************************************************

    tensor_mod.addImport("tensor_m", tensor_math_mod);
    tensor_mod.addImport("architectures", architectures_mod);

    //************************************************TENSOR MATH DEPENDENCIES************************************************

    tensor_math_mod.addImport("tensor", tensor_mod);
    tensor_math_mod.addImport("typeC", typeC_mod);
    tensor_math_mod.addImport("architectures", architectures_mod);

    //************************************************ACTIVATION DEPENDENCIES************************************************

    activation_mod.addImport("tensor", tensor_mod);

    //************************************************LOSS DEPENDENCIES************************************************

    loss_mod.addImport("tensor", tensor_mod);
    loss_mod.addImport("tensor_m", tensor_math_mod);
    loss_mod.addImport("typeC", typeC_mod);

    //************************************************OPTIMIZER DEPENDENCIES************************************************

    optim_mod.addImport("tensor", tensor_mod);
    optim_mod.addImport("model", model_mod);
    optim_mod.addImport("layers", layers_mod);

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>MAIN EXE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // Definition of the main executable
    const exe = b.addExecutable(.{
        .name = "Main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    //************************************************EXE DEPENDENCIES************************************************

    exe.root_module.addImport("tensor", tensor_mod);
    exe.root_module.addImport("model", model_mod);
    exe.root_module.addImport("layers", layers_mod);
    exe.root_module.addImport("dataloader", dataloader_mod);
    exe.root_module.addImport("dataprocessor", dataProcessor_mod);
    exe.root_module.addImport("activation_function", activation_mod);

    // Installation of the executable
    b.installArtifact(exe);

    // Creation of the execution command
    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Esegui l'applicazione");
    run_step.dependOn(&run_cmd.step);

    // Definition of unified tests
    const unit_tests = b.addTest(.{
        .name = "lib_test",
        .root_source_file = b.path("src/tests/lib_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    //************************************************UNIT_TEST DEPENDENCIES************************************************

    unit_tests.root_module.addImport("tensor", tensor_mod);
    unit_tests.root_module.addImport("model", model_mod);
    unit_tests.root_module.addImport("layers", layers_mod);
    unit_tests.root_module.addImport("optim", optim_mod); // Add here
    unit_tests.root_module.addImport("loss", loss_mod);
    unit_tests.root_module.addImport("tensor", tensor_mod);
    unit_tests.root_module.addImport("tensor_m", tensor_math_mod);
    unit_tests.root_module.addImport("activation_function", activation_mod);
    unit_tests.root_module.addImport("dataloader", dataloader_mod);
    unit_tests.root_module.addImport("dataprocessor", dataProcessor_mod);
    unit_tests.root_module.addImport("architectures", architectures_mod);

    // Run tests for module `optim`s
    const optim_tests = b.addTest(.{
        .name = "optim_test",
        .root_source_file = b.path("src/tests/tests_optim.zig"),
        .target = target,
        .optimize = optimize,
    });
    optim_tests.root_module.addImport("optim", optim_mod); // Import `optim` in tests
    const run_optim_tests = b.addRunArtifact(optim_tests);
    const test_optim_step = b.step("test_optim", "Test for Optim");
    test_optim_step.dependOn(&run_optim_tests.step);

    // Add step to run all unit tests
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test_all", "Unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
