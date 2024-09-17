const std = @import("std");

pub fn DataLoader(comptime XType: type, comptime YType: type) type {
    return struct {
        X: [][]XType,
        y: []YType,
        allocator: *const std.mem.Allocator,

        pub fn fromCSV(allocator: *std.mem.Allocator, filePath: []const u8, featureCols: []const usize, labelCol: usize) !@This() {
            const file = try std.fs.cwd().openFile(filePath, .{});
            defer file.close();

            var reader = file.reader();
            const lineBuf = try allocator.alloc(u8, 1024);
            defer allocator.free(lineBuf);

            var xData = try allocator.alloc([][]XType, 0); // Array di slice di XType
            var yData = try allocator.alloc(YType, 0); // Array delle label
            var rows: usize = 0;

            while (try readCSVLine(&reader, lineBuf)) |line| {
                rows += 1;

                const columns = try splitCSVLine(line, allocator);
                defer freeCSVColumns(allocator, columns);

                // Aumenta la dimensione dell'array di righe
                if (!allocator.resize(&xData, rows)) {
                    return error.OutOfMemory;
                }
                if (!allocator.resize(&yData, rows)) {
                    return error.OutOfMemory;
                }

                // Alloca una nuova riga per X
                var features = try allocator.alloc(XType, featureCols.len);
                for (featureCols, 0..) |colIndex, i| {
                    features[i] = try parseXType(XType, columns[colIndex]);
                }

                // Alloca una nuova riga di slice di slice
                xData[rows - 1] = features; // Corretto per assegnare la slice direttamente

                // Estrai la label (y)
                yData[rows - 1] = try parseYType(YType, columns[labelCol]);
            }

            return @This(){
                .X = xData,
                .y = yData,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            for (self.X) |x_row| {
                self.allocator.free(x_row);
            }
            self.allocator.free(self.X);
            self.allocator.free(self.y);
        }

        pub fn print(self: *@This()) void {
            std.debug.print("Features (X):\n", .{});
            for (self.X) |row| {
                for (row) |val| {
                    std.debug.print("{} ", .{val});
                }
                std.debug.print("\n", .{});
            }

            std.debug.print("\nLabels (y):\n", .{});
            for (self.y) |label| {
                std.debug.print("{}\n", .{label});
            }
        }
    };
}

// Funzione per leggere una riga dal file CSV
fn readCSVLine(reader: *std.fs.File.Reader, lineBuf: []u8) !?[]u8 {
    const line = try reader.readUntilDelimiterOrEof(lineBuf, '\n');
    if (line) |l| {
        if (l.len == 0) return null;
        return l;
    }
    return null;
}

// Funzione per splittare una riga CSV nei singoli campi
fn splitCSVLine(line: []u8, allocator: *std.mem.Allocator) ![]const []u8 {
    var columns = std.ArrayList([]u8).init(allocator.*);
    defer columns.deinit();

    var start: usize = 0;
    for (line, 0..) |c, i| {
        if (c == ',' or c == '\n') {
            try columns.append(line[start..i]);
            start = i + 1;
        }
    }
    if (start < line.len) {
        try columns.append(line[start..line.len]);
    }

    return columns.toOwnedSlice();
}

// Funzione per deallocare i campi della riga CSV
fn freeCSVColumns(allocator: *std.mem.Allocator, columns: []const []u8) void {
    allocator.free(columns);
}

// Funzioni di parsing per i tipi generici XType e YType
fn parseXType(comptime XType: type, self: []const u8) !XType {
    return try std.fmt.parseInt(XType, self, 10);
}

fn parseYType(comptime YType: type, self: []const u8) !YType {
    return try std.fmt.parseInt(YType, self, 10);
}

pub fn main() !void {
    var allocator = std.heap.page_allocator;

    const featureCols: [2]usize = [_]usize{ 0, 1 }; // Esempio: prime due colonne come features
    const labelCol: usize = 2; // Esempio: terza colonna come label

    var loader = try DataLoader(i32, i32).fromCSV(&allocator, "data.csv", &featureCols, labelCol);
    defer loader.deinit();

    loader.print();
}
