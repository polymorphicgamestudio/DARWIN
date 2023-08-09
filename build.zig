const std = @import("std");

const llama_path = "./libs/llama.cpp";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "DARWIN",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const llama = b.addStaticLibrary(.{
        .name = "llama",
        .target = target,
        .optimize = optimize,
    });
    llama.addIncludePath(.{ .path = llama_path });
    // llama.addIncludePath(.{ .path = llama_path ++ "/examples" });
    llama.linkLibCpp();
    llama.linkLibC();
    llama.addCSourceFiles(&.{
        llama_path ++ "/ggml.c",
        llama_path ++ "/ggml-alloc.c",
    }, &.{"-std=c11"});
    llama.addCSourceFiles(&.{llama_path ++ "/llama.cpp"}, &.{"-std=c++11"});
    if (target.isWindows()) llama.want_lto = false;

    exe.linkLibC();
    exe.linkLibCpp();
    exe.addIncludePath(.{ .path = llama_path });
    exe.linkLibrary(llama);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
