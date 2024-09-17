const std = @import("std");

pub fn DataLoader(comptime Ftype: type, comptime LabelType: type) type {
    return struct {
        X: [][]Ftype,
        y: []LabelType,
        x_index: usize = 0,
        y_index: usize = 0,

        pub fn xNext(self: *@This()) ?[]Ftype {
            const index = self.x_index;
            for (self.X[index..]) |x_row| {
                self.x_index += 1;
                return x_row;
            }
            return null;
        }

        pub fn yNext(self: *@This()) ?LabelType {
            const index = self.y_index;
            for (self.y[index..]) |label| {
                self.y_index += 1;
                return label;
            }
            return null;
        }

        pub fn fromCSV(self: *@This(), allocator: *std.mem.Allocator, filePath: []const u8, featureCols: []const usize, labelCol: usize) !void {
            const file = try std.fs.cwd().openFile(filePath, .{});
            defer file.close();
            var reader = file.reader();
            const lineBuf = try allocator.alloc(u8, 1024);
            defer allocator.free(lineBuf);

            var numRows: usize = 0;
            while (true) {
                const maybeLine = try readCSVLine(&reader, lineBuf);
                if (maybeLine == null) break; // Fine del file
                numRows += 1;
            }

            try file.seekTo(0);

            self.X = try allocator.alloc([]Ftype, numRows);
            self.y = try allocator.alloc(LabelType, numRows);

            var rowIndex: usize = 0;
            while (true) {
                const maybeLine = try readCSVLine(&reader, lineBuf);
                if (maybeLine == null) break;

                const line = maybeLine.?;
                const columns = try splitCSVLine(line, allocator);
                defer freeCSVColumns(allocator, columns);

                self.X[rowIndex] = try allocator.alloc(Ftype, featureCols.len);

                for (featureCols, 0..) |colIndex, i| {
                    self.X[rowIndex][i] = try parseXType(Ftype, columns[colIndex]);
                }

                self.y[rowIndex] = try parseYType(LabelType, columns[labelCol]);

                rowIndex += 1;
            }
        }
        pub fn loadMNISTImages(self: *@This(), allocator: *std.mem.Allocator, filePath: []const u8) !void {
            const file = try std.fs.cwd().openFile(filePath, .{});
            defer file.close();
            var reader = file.reader();

            //  magic number (4 byte, big-endian)
            const magicNumber = try reader.readInt(u32, .big);
            if (magicNumber != 2051) {
                return error.InvalidFileFormat;
            }
            std.debug.print("Magic number: {d}\n", .{magicNumber});

            // num img (4 byte, big-endian)
            const numImages = try reader.readInt(u32, .big);

            // rows (4 byte, big-endian)
            const numRows = try reader.readInt(u32, .big);

            // columns (4 byte, big-endian)
            const numCols = try reader.readInt(u32, .big);

            // (28x28)
            if (numRows != 28 or numCols != 28) {
                return error.InvalidImageDimensions;
            }

            self.X = try allocator.alloc([]Ftype, numImages);

            const imageSize = numRows * numCols;
            var i: usize = 0;

            while (i < numImages) {
                self.X[i] = try allocator.alloc(Ftype, imageSize);

                const pixels = try allocator.alloc(u8, imageSize);
                defer allocator.free(pixels);

                try reader.readNoEof(pixels);

                var j: usize = 0;
                while (j < imageSize) {
                    self.X[i][j] = pixels[j];
                    j += 1;
                }

                i += 1;
            }
        }

        pub fn loadMNISTLabels(self: *@This(), allocator: *std.mem.Allocator, filePath: []const u8) !void {
            const file = try std.fs.cwd().openFile(filePath, .{});
            defer file.close();
            var reader = file.reader();

            // Magic number (4 byte, big-endian)
            const magicNumber = try reader.readInt(u32, .big);
            if (magicNumber != 2049) {
                return error.InvalidFileFormat;
            }
            std.debug.print("Magic number (labels): {d}\n", .{magicNumber});

            // Number of labels (4 byte, big-endian)
            const numLabels = try reader.readInt(u32, .big);

            self.y = try allocator.alloc(LabelType, numLabels);

            var i: usize = 0;
            while (i < numLabels) {
                const label = try reader.readByte();
                self.y[i] = label;
                i += 1;
            }
        }

        pub fn loadMNISTDataParallel(self: *@This(), allocator: *std.mem.Allocator, imageFilePath: []const u8, labelFilePath: []const u8) !void {
            const image_thread = try std.Thread.spawn(.{}, loadImages, .{ self, allocator, imageFilePath });
            defer image_thread.join();

            const label_thread = try std.Thread.spawn(.{}, loadLabels, .{ self, allocator, labelFilePath });
            defer label_thread.join();
        }

        fn loadImages(loader: *@This(), allocator: *std.mem.Allocator, imageFilePath: []const u8) !void {
            try loader.loadMNISTImages(allocator, imageFilePath);
        }

        fn loadLabels(loader: *@This(), allocator: *std.mem.Allocator, labelFilePath: []const u8) !void {
            try loader.loadMNISTLabels(allocator, labelFilePath);
        }

        pub fn readCSVLine(reader: *std.fs.File.Reader, lineBuf: []u8) !?[]u8 {
            const line = try reader.readUntilDelimiterOrEof(lineBuf, '\n');
            if (line) |l| {
                if (l.len == 0) return null;
                return l;
            }
            return null;
        }

        pub fn splitCSVLine(line: []u8, allocator: *const std.mem.Allocator) ![]const []u8 {
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

        fn freeCSVColumns(allocator: *std.mem.Allocator, columns: []const []u8) void {
            allocator.free(columns);
        }

        fn parseXType(comptime XType: type, self: []const u8) !XType {
            const type_info = @typeInfo(XType);
            if (type_info == .Float) {
                return try std.fmt.parseFloat(XType, self);
            } else {
                return try std.fmt.parseInt(XType, self, 10);
            }
        }

        fn parseYType(comptime YType: type, self: []const u8) !YType {
            const type_info = @typeInfo(YType);
            if (type_info == .Float) {
                return try std.fmt.parseFloat(YType, self);
            } else {
                return try std.fmt.parseInt(YType, self, 10);
            }
        }
        pub fn deinit(self: *@This(), allocator: *std.mem.Allocator) void {
            for (self.X) |features| {
                allocator.free(features);
            }

            allocator.free(self.X);

            allocator.free(self.y);
        }
    };
}
