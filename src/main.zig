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

    seed: u32 = 1, // RNG seed
    n_threads: i32 = 0,
    n_predict: i32 = -1, // new tokens to predict
    n_ctx: i32 = 512, // context size
    n_batch: i32 = 512, // batch size for prompt processing (must be >=32 to use BLAS)
    n_gqa: i32 = 1, // grouped-query attention factor (TODO: move to hparams)
    n_keep: i32 = 0, // number of tokens to keep from initial prompt
    n_chunks: i32 = -1, // max number of chunks to process (-1 = unlimited)
    n_gpu_layers: i32 = 0, // number of layers to store in VRAM
    main_gpu: i32 = 0, // the GPU that is used for scratch and small tensors
    tensor_split: [c.LLAMA_MAX_DEVICES]f32 = [c.LLAMA_MAX_DEVICES]f32{0}, // how split tensors should be distributed across GPUs
    n_probs: i32 = 0, // if greater than 0, output the probabilities of top n_probs tokens.
    rms_norm_eps: f32 = c.LLAMA_DEFAULT_RMS_EPS, // rms norm epsilon
    rope_freq_base: f32 = 10000.0, // RoPE base frequency
    rope_freq_scale: f32 = 1.0, // RoPE frequency scaling factor

    // sampling parameters
    logit_bias: std.AutoHashMap(c.llama_token, f32) = undefined, // logit bias for specific tokens
    top_k: i32 = 40, // <= 0 to use vocab size
    top_p: f32 = 0.95, // 1.0 = disabled
    tfs_z: f32 = 1.00, // 1.0 = disabled
    typical_p: f32 = 1.00, // 1.0 = disabled
    temp: f32 = 0.80, // 1.0 = disabled
    repeat_penalty: f32 = 1.10, // 1.0 = disabled
    repeat_last_n: i32 = 64, // last n tokens to penalize (0 = disable penalty, -1 = context size)
    frequency_penalty: f32 = 0.00, // 0.0 = disabled
    presence_penalty: f32 = 0.00, // 0.0 = disabled
    mirostat: i32 = 0, // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    mirostat_tau: f32 = 5.00, // target entropy
    mirostat_eta: f32 = 0.10, // learning rate

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    cfg_negative_prompt: []u8 = undefined, // string to help guidance
    cfg_scale: f32 = 1.0, // How strong is guidance

    model: [:0]const u8 = "models/7B/ggml-model.bin", // model path
    model_alias: [:0]const u8 = "unknown", // model alias
    prompt: [:0]const u8 = "",
    path_prompt_cache: []const u8 = "", // path to file for saving/loading prompt eval state
    input_prefix: []const u8 = "", // string to prefix user inputs with
    input_suffix: []const u8 = "", // string to suffix user inputs with
    grammar: []const u8 = "", // optional BNF-like grammar to constrain sampling
    antiprompt: std.ArrayList([]u8) = undefined, // string upon seeing which more user input is prompted

    lora_adapter: [:0]const u8 = "", // lora adapter path
    lora_base: [:0]const u8 = "", // base model path for the lora adapter

    hellaswag: bool = false, // compute HellaSwag score over random tasks from datafile supplied in prompt
    hellaswag_tasks: isize = 400, // number of tasks to use when computing the HellaSwag score

    low_vram: bool = false, // if true, reduce VRAM usage at the cost of performance
    mul_mat_q: bool = false, // if true, use experimental mul_mat_q kernels
    memory_f16: bool = true, // use f16 instead of f32 for memory kv
    random_prompt: bool = false, // do not randomize prompt if none provided
    use_color: bool = false, // use color to distinguish generations and inputs
    interactive: bool = false, // interactive mode
    prompt_cache_all: bool = false, // save user input and generations to prompt cache
    prompt_cache_ro: bool = false, // open the prompt cache read-only and do not update it

    embedding: bool = false, // get only sentence embedding
    interactive_first: bool = false, // wait for user input immediately
    multiline_input: bool = false, // reverse the usage of `\`

    input_prefix_bos: bool = false, // prefix BOS to user inputs, preceding input_prefix
    instruct: bool = false, // instruction mode (used for Alpaca models)
    penalize_nl: bool = true, // consider newlines as a repeatable token
    perplexity: bool = false, // compute perplexity over the prompt
    use_mmap: bool = true, // use mmap for faster loads
    use_mlock: bool = false, // use mlock to keep model in memory
    mem_test: bool = false, // compute maximum memory usage
    numa: bool = false, // attempt optimizations that help on some NUMA systems
    export_cgraph: bool = false, // export the computation graph
    verbose_prompt: bool = false, // print prompt tokens before generation

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

fn tokenizeInput(ally: std.mem.Allocator, ctx: ?*c.llama_context, text: [:0]const u8) !std.ArrayList(c.llama_token) {
    var embd_inp = std.ArrayList(c.llama_token).init(ally);

    try embd_inp.resize(text.len + 1); // NOTE(caleb): +1 for BOS (beginning of sentance token)
    const n = c.llama_tokenize(ctx, text.ptr, embd_inp.items.ptr, @as(c_int, @intCast(embd_inp.items.len)), true);
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

    // Add a space in front of the first character to match OG llama tokenizer behavior
    gpt_params.prompt = try std.fmt.allocPrintZ(arena, " {s}", .{gpt_params.prompt});

    // Tokenize the prompt
    var embd_inp = try tokenizeInput(arena, lctx, gpt_params.prompt);
    defer embd_inp.deinit();
}
