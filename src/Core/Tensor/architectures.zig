const std = @import("std");

pub const Architectures = enum {
    CPU,
    GPU,
    SP32,
    WildTarzan,
};

pub const ArchitectureError = error{
    UnknownArchitecture,
    UnderDevelopementArchitecture,
};
