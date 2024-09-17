const std = @import("std");
//const Builder = std.build.Builder;
const Pkg = std.build.Pkg;

pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addStaticLibrary(.{
        .name = "TheTinyBook",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "TheTinyBook",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // TESTS BUILD --------------------------------------------------------------------------------------------
    //
    // IMPORTANT! before writing an addTest read this: https://ziglang.org/learn/build-system/#testing
    //
    const test_step = b.step("test_all", "Run unit tests");

    //add here your tests
    const test_list: []const []const u8 = &[_][]const u8{
        "src/Core/Tensor/tests_tensor.zig",
        "src/Core/Tensor/tests_tensor_math.zig",
        "src/Utils/utils_tests.zig",
        "src/Utils/Dataprocessing/tests_dataLoader.zig",
        //"src/Model/LossFunction/tests_lossFunction.zig",
    };

    for (test_list) |path| {
        const unit_tests = b.addTest(.{
            .root_source_file = b.path(path),
            .target = target,
            .optimize = optimize,
        });

        const run_unit_tests = b.addRunArtifact(unit_tests);
        test_step.dependOn(&run_unit_tests.step);
    }
}
