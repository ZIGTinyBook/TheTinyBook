const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Creiamo i moduli principali
    const tensor_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/tensor.zig") });
    const tensor_math_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/tensor_math.zig") });
    const architectures_mod = b.createModule(.{ .root_source_file = b.path("src/Core/Tensor/architectures.zig") });

    const model_mod = b.createModule(.{ .root_source_file = b.path("src/Model/model.zig") });
    const layers_mod = b.createModule(.{ .root_source_file = b.path("src/Model/layers.zig") });
    const optim_mod = b.createModule(.{ .root_source_file = b.path("src/Model/optim.zig") });
    const dataloader_mod = b.createModule(.{ .root_source_file = b.path("src/DataLoader/dataLoader.zig") });
    const loss_mod = b.createModule(.{ .root_source_file = b.path("src/Model/lossFunction.zig") });
    const activation_mod = b.createModule(.{ .root_source_file = b.path("src/Model/activation_function.zig") });
    const typeC_mod = b.createModule(.{ .root_source_file = b.path("src/Utils/typeConverter.zig") });

    // Aggiungi solo le dipendenze necessarie senza duplicazioni
    model_mod.addImport("tensor", tensor_mod);
    model_mod.addImport("layers", layers_mod);
    model_mod.addImport("optim", optim_mod); // Non rimuovere duplicato
    model_mod.addImport("loss", loss_mod);
    model_mod.addImport("typeC", typeC_mod);
    model_mod.addImport("dataloader", dataloader_mod);
    model_mod.addImport("tensor_m", tensor_math_mod);

    layers_mod.addImport("tensor", tensor_mod);
    layers_mod.addImport("activation_function", activation_mod);
    layers_mod.addImport("tensor_m", tensor_math_mod);
    layers_mod.addImport("architectures", architectures_mod);

    dataloader_mod.addImport("tensor", tensor_mod);

    tensor_math_mod.addImport("tensor", tensor_mod);
    tensor_math_mod.addImport("typeC", typeC_mod);
    tensor_math_mod.addImport("architectures", architectures_mod);

    activation_mod.addImport("tensor", tensor_mod);

    loss_mod.addImport("tensor", tensor_mod);
    loss_mod.addImport("tensor_m", tensor_math_mod);
    loss_mod.addImport("typeC", typeC_mod);

    optim_mod.addImport("model", model_mod);

    // Definizione dell'eseguibile principale
    const exe = b.addExecutable(.{
        .name = "Main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("tensor", tensor_mod);
    exe.root_module.addImport("model", model_mod);
    exe.root_module.addImport("layers", layers_mod);
    exe.root_module.addImport("dataloader", dataloader_mod);

    // Installazione dell'eseguibile
    b.installArtifact(exe);

    // Creazione del comando di esecuzione
    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Esegui l'applicazione");
    run_step.dependOn(&run_cmd.step);

    // Definizione dei test unificati
    const unit_tests = b.addTest(.{
        .name = "lib_test",
        .root_source_file = b.path("src/tests/lib_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Aggiunta delle dipendenze ai test
    unit_tests.root_module.addImport("tensor", tensor_mod);
    unit_tests.root_module.addImport("model", model_mod);
    unit_tests.root_module.addImport("layers", layers_mod);
    unit_tests.root_module.addImport("optim", optim_mod); // Aggiungi qui
    unit_tests.root_module.addImport("loss", loss_mod);

    // Esegui test per modulo `optim`
    const optim_tests = b.addTest(.{
        .name = "optim_test",
        .root_source_file = b.path("src/tests/tests_optim.zig"),
        .target = target,
        .optimize = optimize,
    });
    optim_tests.root_module.addImport("optim", optim_mod); // Importa `optim` nei test
    const run_optim_tests = b.addRunArtifact(optim_tests);
    const test_optim_step = b.step("test_optim", "Esegui i test per Optim");
    test_optim_step.dependOn(&run_optim_tests.step);

    // Aggiunta dello step per eseguire tutti i test
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test_all", "Esegui tutti i test unitari");
    test_step.dependOn(&run_unit_tests.step);
}
