/// Crack at a word level prob model.
const std = @import("std");

const hash_map = std.hash_map;
const mem = std.mem;
const rand = std.rand;

const data_path = "data/tiny_shakes.txt";
const block_length = 3;
const feature_length = 4;

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
    var word_id = @as(usize, 1);
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
            try vocab.putNoClobber(k, word_id);
            word_id += 1;
        }
    }

    // Dataset(s)
    var inputs = std.ArrayList(@Vector(block_length, usize)).init(arena);
    var outputs = std.ArrayList(usize).init(arena);
    try buildDataset(data_file, vocab, &inputs, &outputs);

    // Build context matrix
    const row_count = word_id;
    var context_matrix = try arena.alloc(@Vector(feature_length, f32), row_count);
    for (context_matrix) |*row| {
        var feature_index = @as(usize, 0);
        while (feature_index < feature_index)
            row[feature_index] = rng.floatNorm(f32);
    }
    std.debug.print("Contstructed context matrix: {d}kb\n", .{@sizeOf(@Vector(feature_length, f32)) * row_count / 1024});

    // Embeddings
    // var emb = try arena.alloc(@Vector(feature_length, f32), row_count);

    // for (outputs.items, 0..) |expected, output_index|
    //     std.debug.print("{any} => {d}\n", .{ inputs.items[output_index], expected });

    // var vocab_iter = vocab.valueIterator();
    // while (vocab_iter.next()) |v| {
    //     std.debug.print("{d}, ", .{v.*});
    // }

    // next step is to build C - context matrix
    std.process.cleanExit();
}
