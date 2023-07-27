/// Crack at a word level prob model.
const std = @import("std");

const debug = std.debug;
const hash_map = std.hash_map;
const mem = std.mem;
const rand = std.rand;

const data_path = "data/tiny_shakes.txt";
const block_length = 3;
const feature_length = 4;
const hidden_layer_neuron_count = 10;

fn HeapVector(comptime T: type) type {
    return struct {
        data: []T,
        dim: usize,
        ally: mem.Allocator,

        pub fn initPreheated(ally: mem.Allocator, dim: usize) !HeapVector(T) {
            var result = HeapVector(T){
                .data = undefined,
                .dim = dim,
                .ally = ally,
            };
            result.data = try ally.alloc(T, dim);
            return result;
        }

        pub fn addHeapVector(a: HeapVector(T), b: HeapVector(T)) !HeapVector(T) {
            debug.assert(a.dim == b.dim);
            var result = try HeapVector(T).initPreheated(a.ally, a.dim);
            var row_index: usize = 0;
            while (row_index < a.dim) : (row_index += 1)
                result.data[row_index] = a.data[row_index] + b.data[row_index];
            return result;
        }

        pub fn deinit(self: *HeapVector(T)) void {
            self.ally.free(self.data);
        }
    };
}

fn HeapMatrix(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        m: usize,
        n: usize,
        ally: mem.Allocator,

        pub fn initPreheated(ally: mem.Allocator, m: usize, n: usize) !HeapMatrix(T) {
            var result = HeapMatrix(T){
                .data = undefined,
                .m = m,
                .n = n,
                .ally = ally,
            };
            result.data = try ally.alloc(T, m * n);
            return result;
        }

        pub fn deinit(self: *HeapMatrix(T)) void {
            self.ally.free(self.data);
        }

        pub fn multHeapVector(self: HeapMatrix(T), v: HeapVector(T)) !HeapVector(T) {
            var result = try HeapVector(T).initPreheated(self.ally, self.m);
            var row_index: usize = 0;
            while (row_index < self.m) : (row_index += 1) {
                var col_index: usize = 0;
                while (col_index < self.n) : (col_index += 1)
                    result.data[row_index] += self.data[row_index * self.n + col_index] * v.data[row_index];
            }
            return result;
        }
    };
}

fn buildDataset(
    data_file: std.fs.File,
    vocab: std.AutoHashMap(u64, usize),
    inputs: *std.ArrayList(@Vector(block_length, usize)),
    outputs: *std.ArrayList(usize),
) !void {
    var line_buffer: [1024]u8 = undefined; // 1Kb should be more than enough for a line...
    var data_fbs = std.io.fixedBufferStream(&line_buffer);
    var data_reader = data_file.reader();
    var data_writer = data_fbs.writer();

    var context = mem.zeroes(@Vector(block_length, usize));
    try data_file.seekTo(0);
    while (true) {
        data_fbs.reset();
        data_reader.streamUntilDelimiter(data_writer, '\n', null) catch break;
        if (data_fbs.getWritten().len == 0) continue;

        // No nasty CRs please.
        std.debug.assert(data_fbs.getWritten()[data_fbs.pos - 1] != '\r');

        var word_toks = mem.splitSequence(u8, data_fbs.getWritten(), " ");
        var tok: ?[]const u8 = undefined;
        tok = word_toks.first();
        while (tok != null) : (tok = word_toks.next()) {
            const k = hash_map.hashString(tok.?);
            const v = vocab.get(k) orelse unreachable;
            try inputs.append(context);
            try outputs.append(v);
            var context_index = @as(usize, 1);
            while (context_index < block_length) : (context_index += 1)
                context[context_index - 1] = context[context_index];
            context[block_length - 1] = v;
        }
    }
}

pub fn main() !void {
    var arena_instance = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    var xoshi = rand.Xoroshiro128.init(1);
    const rng = xoshi.random();

    var vocab = std.AutoHashMap(u64, usize).init(arena);
    var vocab_count = @as(usize, 1);
    const data_file = try std.fs.cwd().openFile(data_path, .{});
    defer data_file.close();

    // Build vocab from input file
    var line_buffer: [1024]u8 = undefined; // 1Kb should be more than enough for a line...
    var data_fbs = std.io.fixedBufferStream(&line_buffer);
    var data_reader = data_file.reader();
    var data_writer = data_fbs.writer();
    while (true) {
        data_fbs.reset();
        data_reader.streamUntilDelimiter(data_writer, '\n', null) catch break;
        if (data_fbs.getWritten().len == 0) continue;

        // No nasty CRs please.
        std.debug.assert(data_fbs.getWritten()[data_fbs.pos - 1] != '\r');

        var word_toks = mem.splitSequence(u8, data_fbs.getWritten(), " ");
        var tok: ?[]const u8 = undefined;
        tok = word_toks.first();
        while (tok != null) : (tok = word_toks.next()) {
            const k = hash_map.hashString(tok.?);
            if (vocab.contains(k)) continue;
            try vocab.putNoClobber(k, vocab_count);
            vocab_count += 1;
        }
    }

    var inputs = std.ArrayList(@Vector(block_length, usize)).init(arena);
    var outputs = std.ArrayList(usize).init(arena);
    try buildDataset(data_file, vocab, &inputs, &outputs);

    var context_matrix = try arena.alloc(@Vector(feature_length, f32), vocab_count);
    for (context_matrix) |*row| {
        var feature_index = @as(usize, 0);
        while (feature_index < feature_length) : (feature_index += 1)
            row[feature_index] = rng.float(f32);
    }
    std.debug.print("Contstructed context matrix: {d}kb\n", .{@sizeOf(@Vector(feature_length, f32)) * vocab_count / 1024});

    // TODO(caleb): Save arena state and restore to it later

    // var output_layer_matrix = try HeapMatrix(f32).initPreheated(arena, vocab_count, hidden_layer_neuron_count);
    // var output_layer_bias_vector = try HeapVector(f32).initPreheated(arena, vocab_count);

    var hidden_layer_matrix = try HeapMatrix(f32).initPreheated(arena, hidden_layer_neuron_count, feature_length * block_length);
    var hidden_layer_bias_vector = try HeapVector(f32).initPreheated(arena, hidden_layer_neuron_count);

    // FORWARD PHASE ---------------------------------------------------------------------------------

    var random_context_indicies = HeapVector(usize).initPreheated(arena, vocab_count);
    for (&random_context_indicies.data) |*elm| elm.* = rng.uintLessThan(usize, vocab_count);

    // 1) Forward computation for word features layer. NOTE(caleb): (activation layer)
    var catd_feature_vector = try HeapVector(f32).initPreheated(arena, feature_length * block_length);

    // Each peice of context

    // 2) Forward computation for the hidden layer
    const weights_multd_catd_feature_vector = try HeapMatrix(f32).multHeapVector(
        hidden_layer_matrix,
        catd_feature_vector,
    );
    var tan_hidden_layer = try HeapVector(f32).addHeapVector(
        weights_multd_catd_feature_vector,
        hidden_layer_bias_vector,
    );
    for (&tan_hidden_layer.data) |*elm| elm.* = @tan(elm.*);

    // Multiply cated_feature_vector with hidden layer matrix

    // for (outputs.items, 0..) |expected, output_index|
    //     std.debug.print("{any} => {d}\n", .{ inputs.items[output_index], expected });

    // var vocab_iter = vocab.valueIterator();
    // while (vocab_iter.next()) |v| {
    //     std.debug.print("{d}, ", .{v.*});
    // }

    // next step is to build C - context matrix
    std.process.cleanExit();
}
