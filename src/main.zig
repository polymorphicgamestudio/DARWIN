//
// ____    ______  ____    __      __  ______   __  __
///\  _`\ /\  _  \/\  _`\ /\ \  __/\ \/\__  _\ /\ \/\ \
//\ \ \/\ \ \ \L\ \ \ \L\ \ \ \/\ \ \ \/_/\ \/ \ \ `\\ \
// \ \ \ \ \ \  __ \ \ ,  /\ \ \ \ \ \ \ \ \ \  \ \ , ` \
//  \ \ \_\ \ \ \/\ \ \ \\ \\ \ \_/ \_\ \ \_\ \__\ \ \`\ \
//   \ \____/\ \_\ \_\ \_\ \_\ `\___x___/ /\_____\\ \_\ \_\
//    \/___/  \/_/\/_/\/_/\/ /'\/__//__/  \/_____/ \/_/\/_/
//       a llama communication layer written in zig.
//

const c = @cImport({
    @cInclude("llama.h");
});

const std = @import("std");
const mem = std.mem;
const heap = std.heap;
const process = std.process;

const GPTParams = struct {
    const Self = @This();
    ally: mem.Allocator,

    /// RNG seed
    seed: c_uint = 1,
    n_threads: c_int = 0,
    /// new tokens to predict
    n_predict: c_int = -1,
    /// context size
    n_ctx: c_int = 512,
    /// batch size for prompt processing (must be >=32 to use BLAS)
    n_batch: c_int = 512,
    /// grouped-query attention factor (TODO: move to hparams)
    n_gqa: c_int = 1,
    /// number of tokens to keep from initial prompt
    n_keep: c_int = 0,
    /// max number of chunks to process (-1 = unlimited)
    n_chunks: c_int = -1,
    /// number of layers to store in VRAM
    n_gpu_layers: c_int = 0,
    /// the GPU that is used for scratch and small tensors
    main_gpu: c_int = 0,
    /// how split tensors should be distributed across GPUs
    tensor_split: [c.LLAMA_MAX_DEVICES]f32 = [c.LLAMA_MAX_DEVICES]f32{0},
    /// if greater than 0, output the probabilities of top n_probs tokens.
    n_probs: c_int = 0,
    /// rms norm epsilon
    rms_norm_eps: f32 = c.LLAMA_DEFAULT_RMS_EPS,
    /// RoPE base frequency
    rope_freq_base: f32 = 10000.0,
    /// RoPE frequency scaling factor
    rope_freq_scale: f32 = 1.0,

    // Sampling parameters ------------------------------------------------------------------------

    /// logit bias for specific tokens
    logit_bias: std.AutoHashMap(c.llama_token, f32) = undefined,
    /// <= 0 to use vocab size
    top_k: c_int = 40,
    /// 1.0 = disabled
    top_p: f32 = 0.95,
    /// 1.0 = disabled
    tfs_z: f32 = 1.00,
    /// 1.0 = disabled
    typical_p: f32 = 1.00,
    /// 1.0 = disabled
    temp: f32 = 0.80,
    /// 1.0 = disabled
    repeat_penalty: f32 = 1.10,
    /// last n tokens to penalize (0 = disable penalty, -1 = context size)
    repeat_last_n: c_int = 64,
    /// 0.0 = disabled
    frequency_penalty: f32 = 0.00,
    /// 0.0 = disabled
    presence_penalty: f32 = 0.00,
    /// 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    mirostat: c_int = 0,
    /// target entropy
    mirostat_tau: f32 = 5.00,
    /// learning rate
    mirostat_eta: f32 = 0.10,
    /// Classifier-Free Guidance
    /// https:///arxiv.org/abs/2306.17806
    /// string to help guidance
    cfg_negative_prompt: []u8 = undefined,
    /// How strong is guidance
    cfg_scale: f32 = 1.0,

    /// model path
    model: [:0]const u8 = "models/7B/ggml-model.bin",
    /// model alias
    model_alias: [:0]const u8 = "unknown",
    prompt: [:0]const u8 = "",
    /// path to file for saving/loading prompt eval state
    path_prompt_cache: []const u8 = "",
    /// string to prefix user inputs with
    input_prefix: []const u8 = "",
    /// string to suffix user inputs with
    input_suffix: []const u8 = "",
    /// optional BNF-like grammar to constrain sampling
    grammar: []const u8 = "",
    /// string upon seeing which more user input is prompted
    antiprompt: std.ArrayList([]u8) = undefined,

    /// lora adapter path
    lora_adapter: [:0]const u8 = "",
    /// base model path for the lora adapter
    lora_base: [:0]const u8 = "",

    /// compute HellaSwag score over random tasks from datafile supplied in prompt
    hellaswag: bool = false,
    /// number of tasks to use when computing the HellaSwag score
    hellaswag_tasks: isize = 400,

    /// if true, reduce VRAM usage at the cost of performance
    low_vram: bool = false,
    /// if true, use experimental mul_mat_q kernels
    mul_mat_q: bool = false,
    /// use f16 instead of f32 for memory kv
    memory_f16: bool = true,
    /// do not randomize prompt if none provided
    random_prompt: bool = false,
    /// use color to distinguish generations and inputs
    use_color: bool = false,
    /// interactive mode
    interactive: bool = false,
    /// save user input and generations to prompt cache
    prompt_cache_all: bool = false,
    /// open the prompt cache read-only and do not update it
    prompt_cache_ro: bool = false,

    /// get only sentence embedding
    embedding: bool = false,
    /// wait for user input immediately
    interactive_first: bool = false,
    /// reverse the usage of `\`
    multiline_input: bool = false,

    /// prefix BOS to user inputs, preceding input_prefix
    input_prefix_bos: bool = false,
    /// instruction mode (used for Alpaca models)
    instruct: bool = false,
    /// consider newlines as a repeatable token
    penalize_nl: bool = true,
    /// compute perplexity over the prompt
    perplexity: bool = false,
    /// use mmap for faster loads
    use_mmap: bool = true,
    /// use mlock to keep model in memory
    use_mlock: bool = false,
    /// compute maximum memory usage
    mem_test: bool = false,
    /// attempt optimizations that help on some NUMA systems
    numa: bool = false,
    /// export the computation graph
    export_cgraph: bool = false,
    /// Print prompt tokens before generation
    verbose_prompt: bool = false,

    pub fn init(ally: mem.Allocator) Self {
        var result = Self{ .ally = ally };
        result.n_threads = @as(i32, @intCast(std.Thread.getCpuCount() catch unreachable));
        return result;
    }

    pub fn parse(self: *Self, arg_iter: *process.ArgIterator) !void {
        while (arg_iter.next()) |arg| {
            if (mem.eql(u8, arg, "-p") or mem.eql(u8, arg, "--prompt")) {
                self.prompt = try self.ally.dupeZ(u8, arg_iter.next() orelse unreachable);
            } else if (mem.eql(u8, arg, "-m") or mem.eql(u8, arg, "--model")) {
                self.model = try self.ally.dupeZ(u8, arg_iter.next() orelse unreachable);
            }
        }
    }
};

fn tokenize(ally: std.mem.Allocator, ctx: ?*c.llama_context, text: [:0]const u8, bos: bool) !std.ArrayList(c.llama_token) {
    var embd_inp = std.ArrayList(c.llama_token).init(ally);

    try embd_inp.resize(text.len + if (bos) @as(usize, 1) else @as(usize, 0)); // NOTE(caleb): +1 for BOS (beginning of sentance token)
    const n = c.llama_tokenize(ctx, text.ptr, embd_inp.items.ptr, @as(c_int, @intCast(embd_inp.items.len)), bos);
    std.debug.assert(n >= 0);
    try embd_inp.resize(@as(usize, @intCast(n)));

    return embd_inp;
}

var ctx_ptr: *?*c.llama_context = undefined;

pub fn main() !void {
    var arena_instance = heap.ArenaAllocator.init(heap.page_allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    var stderr_file = std.io.getStdErr();
    const stderr = stderr_file.writer();
    var stdout_file = std.io.getStdOut();
    const stdout = stdout_file.writer();

    var gpt_params = GPTParams.init(arena);
    var arg_iter = try process.argsWithAllocator(arena);
    try gpt_params.parse(&arg_iter);

    c.llama_backend_init(gpt_params.numa);

    var lparams = mem.zeroes(c.llama_context_params);
    lparams.n_ctx = gpt_params.n_ctx;
    lparams.n_batch = gpt_params.n_batch;
    lparams.n_gqa = gpt_params.n_gqa;
    lparams.rms_norm_eps = gpt_params.rms_norm_eps;
    lparams.n_gpu_layers = gpt_params.n_gpu_layers;
    lparams.main_gpu = gpt_params.main_gpu;
    lparams.tensor_split = &gpt_params.tensor_split;
    lparams.low_vram = gpt_params.low_vram;
    lparams.mul_mat_q = gpt_params.mul_mat_q;
    lparams.seed = gpt_params.seed;
    lparams.f16_kv = gpt_params.memory_f16;
    lparams.use_mmap = gpt_params.use_mmap;
    lparams.use_mlock = gpt_params.use_mlock;
    lparams.logits_all = gpt_params.perplexity;
    lparams.embedding = gpt_params.embedding;
    lparams.rope_freq_base = gpt_params.rope_freq_base;
    lparams.rope_freq_scale = gpt_params.rope_freq_scale;

    var model = c.llama_load_model_from_file(gpt_params.model, lparams);
    if (model == null) {
        try stderr.print("failed to load model '{s}'\n", .{gpt_params.model});
        process.exit(1);
    }

    var lctx = c.llama_new_context_with_model(model, lparams);
    if (lctx == null) {
        try stderr.print("failed to create context with model '{s}'\n", .{gpt_params.model});
        process.exit(1);
    }
    ctx_ptr = &lctx;

    if (gpt_params.lora_adapter.len != 0) {
        var err = c.llama_model_apply_lora_from_file(model, gpt_params.lora_adapter, if (gpt_params.lora_base.len == 0) null else gpt_params.lora_base, gpt_params.n_threads);
        if (err != 0) {
            try stderr.print("failed to apply lora adapter\n", .{});
            c.llama_free(lctx);
            c.llama_free_model(model);
            process.exit(1);
        }
    }

    var ctx_guidance: ?*c.llama_context = null;
    if (gpt_params.cfg_scale > 1.0) {
        ctx_guidance = c.llama_new_context_with_model(model, lparams);
    }

    if (model == null) {
        try stderr.print("unable to load model\n", .{});
        process.exit(1);
    }

    _ = c.llama_print_system_info();

    // TODO(caleb): see llama.cpp:145
    // determine the maximum memory usage needed to do inference for the given n_batch and n_ctx parameters
    // uncomment the "used_mem" line in llama.cpp to see the results

    var path_session = gpt_params.path_prompt_cache;
    var session_tokens = std.ArrayList(c.llama_token).init(arena);
    session_tokens.deinit();

    // TODO(caleb): load session tokens see llama.cpp's main.cpp:172

    // Add a space in front of the first character to match OG llama tokenizer behavior
    gpt_params.prompt = try std.fmt.allocPrintZ(arena, " {s}", .{gpt_params.prompt});

    // Tokenize the prompt
    var embd_inp = try tokenize(arena, lctx, gpt_params.prompt, true);
    defer embd_inp.deinit();

    var guidance_inp = std.ArrayList(c.llama_token).init(arena);
    _ = guidance_inp;
    var guidance_offset: c_int = 0;
    var original_prompt_len: c_int = 0;
    _ = original_prompt_len;
    // TODO(caleb): Tokenize negative prompt see main.cpp:207

    const n_ctx = c.llama_n_ctx(lctx);
    if (embd_inp.items.len > n_ctx - 4) {
        try stderr.print("prompt is too long ({d} tokens, max {d})\n", .{ embd_inp.items.len, n_ctx - 4 });
        process.exit(1);
    }

    // Debug message about similarity of saved session, if applicable
    var n_matching_session_tokens: usize = 0;

    // const inp_pfx = try tokenizeInput(arena, lctx, "\n\n### Instruction:\n\n", true);
    // _ = inp_pfx;
    // const inp_sfx = try tokenizeInput(arena, lctx, "\n\n### Response:\n\n", false);
    // _ = inp_sfx;

    const llama_token_newline = try tokenize(arena, lctx, "\n", false);
    _ = llama_token_newline;

    try stderr.print("sampling: repeat_last_n = {d}, repeat_penalty = {d:.3}, presence_penalty = {d:.3}, frequency_penalty = {d:.3}, top_k = {d}, tfs_z = {d:.3}, top_p = {d:.3}, typical_p = {d:.3}, temp = {d:.3}, mirostat = {d}, mirostat_lr = {d:.3}, mirostat_ent = {d:.3}\n", .{ gpt_params.repeat_last_n, gpt_params.repeat_penalty, gpt_params.presence_penalty, gpt_params.frequency_penalty, gpt_params.top_k, gpt_params.tfs_z, gpt_params.top_p, gpt_params.typical_p, gpt_params.temp, gpt_params.mirostat, gpt_params.mirostat_eta, gpt_params.mirostat_tau });
    try stderr.print("generate: n_ctx = {d}, n_batch = {d}, n_predict = {d}, n_keep = {d}\n", .{ n_ctx, gpt_params.n_batch, gpt_params.n_predict, gpt_params.n_keep });
    try stderr.writeAll("\n\n");

    // TODO(llama.cpp author): replace with ring-buffer
    var last_n_tokens = try std.ArrayList(c.llama_token).initCapacity(arena, @as(usize, @intCast(n_ctx)));
    defer last_n_tokens.deinit();
    for (last_n_tokens.items) |*tok| tok.* = mem.zeroes(c.llama_token);

    var is_antiprompt: bool = false;
    var input_echo: bool = true;
    _ = input_echo;
    var need_to_save_session: bool = path_session.len != 0 and n_matching_session_tokens < embd_inp.items.len;
    _ = need_to_save_session;

    var n_past: c_int = 0;
    var n_remain: c_int = gpt_params.n_predict;
    var n_consumed: c_int = 0;
    _ = n_consumed;
    var n_session_consumed: c_int = 0;
    var n_past_guidance: c_int = 0;

    var embd = std.ArrayList(c.llama_token).init(arena);
    defer embd.deinit();
    var embd_guidance = std.ArrayList(c.llama_token).init(arena);
    defer embd_guidance.deinit();

    // Do one empty run to warm up the model
    {
        const restore_state = arena_instance.state;
        defer arena_instance.state = restore_state;
        var tmp = try std.ArrayList(c.llama_token).initCapacity(arena, 1);
        tmp.insertAssumeCapacity(0, c.llama_token_bos());
        _ = c.llama_eval(lctx, tmp.items.ptr, @as(c_int, @intCast(tmp.items.len)), 0, gpt_params.n_threads);
        _ = c.llama_reset_timings(lctx);
    }

    while ((n_remain != 0 and !is_antiprompt) or gpt_params.interactive) {
        // predict
        if (embd.items.len > 0) {
            // NOTE(llama.cpp author): n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            const max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if (@as(c_int, @intCast(embd.items.len)) > max_embd_size) {
                const skipped_tokens = @as(c_int, @intCast(embd.items.len)) - max_embd_size;
                try stdout.print("<<input too long: skipped {d} token{s}>>", .{ skipped_tokens, if (skipped_tokens != 1) "s" else "" });
                try embd.resize(@as(usize, @intCast(max_embd_size)));
            }

            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + @as(c_int, @intCast(embd.items.len)) + @max(0, guidance_offset) > n_ctx) {
                const n_left = n_past - gpt_params.n_keep;

                // always keep the first token - BOS
                n_past = @max(1, gpt_params.n_keep);
                n_past_guidance = @max(1, gpt_params.n_keep + guidance_offset);

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                for (0..@as(usize, @intCast(@divTrunc(n_left, 2)))) |tok_index|
                    try embd.insert(tok_index, last_n_tokens.items[tok_index]);

                // stop saving session if we run out of context
                path_session = "";

                try stdout.writeAll("\n---\n");
                try stdout.writeAll("resetting: '");
                for (embd.items) |tok| try stdout.print("{s}", .{c.llama_token_to_str(lctx, tok)});
                try stdout.writeAll("'\n");
                try stdout.writeAll("\n---\n");
            }

            //TODO(caleb): reuse matching prefix from loaded session main.cpp:458

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always

            //TODO(caleb): Handle context_guidance main.cpp:483

            for (embd.items, 0..) |*tok, tok_index| {
                var n_eval = @as(c_int, @intCast(embd.items.len)) - @as(c_int, @intCast(tok_index));
                if (n_eval > gpt_params.n_batch) n_eval = gpt_params.n_batch;
                if (c.llama_eval(lctx, tok, n_eval, n_past, gpt_params.n_threads) == 0) {
                    try stderr.writeAll("failed to eval\n");
                    process.exit(1);
                }
                n_past += n_eval;
            }

            if (embd.items.len > 0 and path_session.len != 0) {
                for (embd.items, 0..) |tok, tok_index| try session_tokens.insert(tok_index, tok);
                n_session_consumed = @intCast(session_tokens.items.len);
            }
        } // predict

        embd.clearRetainingCapacity();
        embd_guidance.clearRetainingCapacity();
    }
}
