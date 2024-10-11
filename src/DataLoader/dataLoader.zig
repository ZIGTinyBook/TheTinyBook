const std = @import("std");
const tensor = @import("tensor");

//Look at to array to have the x type of custom dimension not just 2 (batch x features)
pub fn DataLoader(comptime OutType: type, comptime Ftype: type, comptime LabelType: type, batchSize: i16) type {
    return struct {
        X: [][]OutType,
        y: []OutType,
        x_index: usize = 0,
        y_index: usize = 0,
        xTensor: tensor.Tensor(OutType),
        yTensor: tensor.Tensor(OutType),
        batchSize: usize = batchSize,
        XBatch: [][]OutType,
        yBatch: []OutType,

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

        pub fn toTensor(self: *@This(), allocator: *const std.mem.Allocator, shapeX: *[]usize, shapeY: *[]usize) !void {
            self.xTensor = try tensor.Tensor(OutType).fromArray(allocator, self.XBatch, shapeX.*);
            self.yTensor = try tensor.Tensor(OutType).fromArray(allocator, self.yBatch, shapeY.*);
        }

        pub fn reset(self: *@This()) void {
            self.x_index = 0;
            self.y_index = 0;
        }
        //Maybe do batch size as a "attribute of the struct"
        pub fn xNextBatch(self: *@This(), batch_size: usize) ?[][]OutType {
            const start = self.x_index;
            const end = @min(start + batch_size, self.X.len);

            if (start >= end) return null;

            const batch = self.X[start..end];
            self.x_index = end;
            self.XBatch = batch;
            return batch;
        }

        pub fn yNextBatch(self: *@This(), batch_size: usize) ?[]OutType {
            const start = self.y_index;
            const end = @min(start + batch_size, self.y.len);

            if (start >= end) return null;

            const batch = self.y[start..end];
            self.y_index = end;
            self.yBatch = batch;
            return batch;
        }

        //We are using Knuth shuffle algorithm with complexity O(n)

        pub fn shuffle(self: *@This(), rng: *std.Random.DefaultPrng) void {
            const len = self.X.len;

            if (len <= 1) return;

            var i: usize = len - 1;
            while (true) {
                const j = rng.random().uintLessThan(usize, i + 1);

                const temp_feature = self.X[i];
                self.X[i] = self.X[j];
                self.X[j] = temp_feature;

                const temp_label = self.y[i];
                self.y[i] = self.y[j];
                self.y[j] = temp_label;

                if (i == 0) break;
                i -= 1;
            }
        }

        pub fn fromCSV(self: *@This(), allocator: *const std.mem.Allocator, filePath: []const u8, featureCols: []const usize, labelCol: usize) !void {
            const file = try std.fs.cwd().openFile(filePath, .{});
            defer file.close();
            var reader = file.reader();
            const lineBuf = try allocator.alloc(u8, 1024);
            defer allocator.free(lineBuf);

            // Conta il numero di righe
            var numRows: usize = 0;
            while (true) {
                const maybeLine = try readCSVLine(&reader, lineBuf);
                if (maybeLine == null) break;
                numRows += 1;
            }

            // Ripristina il puntatore al file
            try file.seekTo(0);

            // Alloca lo spazio per i dati di X e y
            self.X = try allocator.alloc([]OutType, numRows);
            self.y = try allocator.alloc(OutType, numRows);

            var rowIndex: usize = 0;
            while (true) {
                const maybeLine = try readCSVLine(&reader, lineBuf);
                if (maybeLine == null) break;

                const line = maybeLine.?;
                const columns = try splitCSVLine(line, allocator);
                defer freeCSVColumns(allocator, columns);

                // Alloca la memoria per le feature di ogni riga
                self.X[rowIndex] = try allocator.alloc(Ftype, featureCols.len);

                // Itera attraverso le colonne delle feature
                for (featureCols, 0..) |colIndex, i| {
                    const valueStr = columns[colIndex];
                    const parsedIntValue = try parseXType(OutType, valueStr); // Parse the value as an integer first

                    if (@TypeOf(Ftype) == f32 or @TypeOf(Ftype) == f64) {
                        // Se Ftype è float, usa @floatFromInt per la conversione
                        self.X[rowIndex][i] = @as(OutType, (parsedIntValue));
                    } else {
                        // Altrimenti, effettua il cast al tipo di output specificato
                        self.X[rowIndex][i] = @as(OutType, (parsedIntValue));
                    }
                }

                // Gestione della colonna etichetta
                const labelValueStr = columns[labelCol];
                const parsedLabelIntValue = try parseYType(OutType, labelValueStr);

                if (@TypeOf(LabelType) == f32 or @TypeOf(LabelType) == f64) {
                    // Se LabelType è float, usa @floatFromInt per la conversione
                    self.y[rowIndex] = @as(OutType, @floatFromInt(parsedLabelIntValue));
                } else {
                    // Altrimenti, effettua il cast al tipo di output specificato
                    self.y[rowIndex] = @as(OutType, parsedLabelIntValue);
                }

                rowIndex += 1;
            }
        }
        pub fn loadMNISTImages(self: *@This(), allocator: *const std.mem.Allocator, filePath: []const u8) !void {
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

            self.X = try allocator.alloc([]OutType, numImages);

            const imageSize = numRows * numCols;
            var i: usize = 0;

            while (i < numImages) {
                self.X[i] = try allocator.alloc(OutType, imageSize);

                const pixels = try allocator.alloc(u8, imageSize);
                defer allocator.free(pixels);

                try reader.readNoEof(pixels);

                var j: usize = 0;
                while (j < imageSize) {
                    self.X[i][j] = @as(OutType, @floatFromInt(pixels[j]));
                    j += 1;
                }

                i += 1;
            }
        }

        pub fn loadMNISTLabels(self: *@This(), allocator: *const std.mem.Allocator, filePath: []const u8) !void {
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

            self.y = try allocator.alloc(OutType, numLabels);

            var i: usize = 0;
            while (i < numLabels) {
                const label = @as(OutType, @floatFromInt(try reader.readByte()));
                self.y[i] = label;
                i += 1;
            }
        }

        pub fn loadMNISTDataParallel(self: *@This(), allocator: *const std.mem.Allocator, imageFilePath: []const u8, labelFilePath: []const u8) !void {
            const image_thread = try std.Thread.spawn(.{}, loadImages, .{ self, allocator, imageFilePath });
            defer image_thread.join();

            const label_thread = try std.Thread.spawn(.{}, loadLabels, .{ self, allocator, labelFilePath });
            defer label_thread.join();
        }

        fn loadImages(loader: *@This(), allocator: *const std.mem.Allocator, imageFilePath: []const u8) !void {
            try loader.loadMNISTImages(allocator, imageFilePath);
        }

        fn loadLabels(loader: *@This(), allocator: *const std.mem.Allocator, labelFilePath: []const u8) !void {
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

        fn freeCSVColumns(allocator: *const std.mem.Allocator, columns: []const []u8) void {
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
