require "candle"

device = Candle::Device.best
passed = []
failed = []

def test(name, passed, failed)
  print "  #{name}... "
  result = yield
  puts "✅"
  passed << "#{name}"
  result
rescue => e
  puts "❌ #{e.message[0..150]}"
  failed << { name: name, error: e.message }
  nil
end

# ============================================================
# Build Info
# ============================================================
puts "=" * 80
puts "Build Info"
puts "-" * 80
info = Candle::BuildInfo.summary
puts "  Device: #{info[:default_device]} | Metal: #{info[:metal_available]} | CUDA: #{info[:cuda_available]}"

# ============================================================
# LLM Models (from MODEL_SUPPORT.md - Known Working)
# ============================================================
llm_models = [
  { name: "Mistral 7B (GGUF)", model: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    options: { gguf_file: "mistral-7b-instruct-v0.2.Q4_K_M.gguf", tokenizer: "mistralai/Mistral-7B-Instruct-v0.2" } },
  { name: "TinyLlama 1.1B (GGUF)", model: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    options: { gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" } },
  { name: "Gemma 3 4B (GGUF)", model: "google/gemma-3-4b-it-qat-q4_0-gguf",
    options: { gguf_file: "gemma-3-4b-it-q4_0.gguf", tokenizer: "google/gemma-3-4b-it" } },
  { name: "Qwen 2.5 1.5B (GGUF)", model: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    options: { gguf_file: "qwen2.5-1.5b-instruct-q4_k_m.gguf" } },
  { name: "Qwen3 0.6B (safetensors)", model: "Qwen/Qwen3-0.6B" },
  { name: "Qwen3 0.6B (GGUF)", model: "MaziyarPanahi/Qwen3-0.6B-GGUF",
    options: { gguf_file: "Qwen3-0.6B.Q4_K_M.gguf", tokenizer: "Qwen/Qwen3-0.6B" } },
  { name: "SmolLM2 360M (safetensors)", model: "HuggingFaceTB/SmolLM2-360M-Instruct" },
  { name: "SmolLM2 360M (GGUF)", model: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
    options: { gguf_file: "smollm2-360m-instruct-q8_0.gguf" } },
  { name: "Phi-2 (safetensors)", model: "microsoft/phi-2" },
  { name: "Phi-2 (GGUF)", model: "TheBloke/phi-2-GGUF",
    options: { gguf_file: "phi-2.Q4_K_M.gguf" } },
  { name: "Phi-3 mini 4k (safetensors)", model: "microsoft/Phi-3-mini-4k-instruct" },
  { name: "Phi-3 mini 4k (GGUF)", model: "microsoft/Phi-3-mini-4k-instruct-gguf",
    options: { gguf_file: "Phi-3-mini-4k-instruct-q4.gguf" } },
  { name: "Phi-4 (GGUF)", model: "microsoft/phi-4-gguf",
    options: { gguf_file: "phi-4-Q4_K_S.gguf" } },
  { name: "Yi-1.5 6B Chat (GGUF)", model: "bartowski/Yi-1.5-6B-Chat-GGUF",
    options: { gguf_file: "Yi-1.5-6B-Chat-Q4_K_M.gguf", tokenizer: "01-ai/Yi-1.5-6B-Chat" } },
  { name: "Granite 7B (safetensors)", model: "ibm-granite/granite-7b-instruct" },
  { name: "Granite 4.0 Micro (safetensors)", model: "ibm-granite/granite-4.0-micro" },
  { name: "GLM-4 9B (GGUF)", model: "bartowski/THUDM_GLM-4-9B-0414-GGUF",
    options: { gguf_file: "THUDM_GLM-4-9B-0414-Q4_K_M.gguf", tokenizer: "THUDM/GLM-4-9B-0414" } },
]

messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is Ruby?" }
]

llm_models.each do |entry|
  puts "=" * 80
  puts "LLM: #{entry[:name]}"
  puts "-" * 80

  llm = test("load", passed, failed) do
    if entry[:options]
      Candle::LLM.from_pretrained(entry[:model], device: device, **entry[:options])
    else
      Candle::LLM.from_pretrained(entry[:model], device: device)
    end
  end
  next unless llm

  config = Candle::GenerationConfig.balanced(max_length: 30)

  test("generate", passed, failed) { llm.generate("What is Ruby?", config: config) }
  test("generate (2nd call, KV cache)", passed, failed) { llm.generate("The capital of France is", config: config) }
  test("chat", passed, failed) { llm.chat(messages, config: config) }
  test("generate_stream", passed, failed) do
    tokens = []
    llm.generate_stream("Hello", config: config) { |t| tokens << t }
    raise "no tokens streamed" if tokens.empty?
    tokens.join
  end
  test("chat_stream", passed, failed) do
    tokens = []
    llm.chat_stream(messages, config: config) { |t| tokens << t }
    raise "no tokens streamed" if tokens.empty?
    tokens.join
  end
  test("tokenizer access", passed, failed) { llm.tokenizer }
  test("eos_token", passed, failed) { llm.eos_token }
  test("model_id", passed, failed) { llm.model_id }
  test("options", passed, failed) { llm.options }
  test("inspect", passed, failed) { llm.inspect }

  # Free model memory before loading the next one
  llm = nil
  GC.start(full_mark: true, immediate_sweep: true)
  puts
end

# ============================================================
# Structured Generation (uses TinyLlama GGUF — small & fast)
# ============================================================
puts "=" * 80
puts "Structured Generation"
puts "-" * 80

struct_llm = test("load TinyLlama for structured gen", passed, failed) do
  Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", device: device)
end

if struct_llm
  schema = {
    type: "object",
    properties: {
      answer: { type: "string", enum: ["yes", "no"] },
      confidence: { type: "number" }
    },
    required: ["answer"]
  }

  test("constraint_from_schema", passed, failed) { struct_llm.constraint_from_schema(schema) }
  test("constraint_from_regex", passed, failed) { struct_llm.constraint_from_regex('\d+') }
  test("generate_structured", passed, failed) do
    result = struct_llm.generate_structured("Is Ruby a language?", schema: schema, max_length: 50)
    raise "not a Hash" unless result.is_a?(Hash)
    result
  end
  test("generate_regex", passed, failed) do
    struct_llm.generate_regex("The answer is", pattern: '\d+', max_length: 10)
  end

  # Free model memory
  struct_llm = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

# ============================================================
# Tool Calling (uses Qwen3 GGUF — supports tool calling)
# ============================================================
puts
puts "=" * 80
puts "Tool Calling"
puts "-" * 80

tool_llm = test("load Qwen3 for tool calling", passed, failed) do
  Candle::LLM.from_pretrained("MaziyarPanahi/Qwen3-0.6B-GGUF",
    gguf_file: "Qwen3-0.6B.Q4_K_M.gguf", tokenizer: "Qwen/Qwen3-0.6B", device: device)
end

if tool_llm
  calculator = Candle::Tool.new(
    name: "calculate",
    description: "Evaluate a math expression",
    parameters: { type: "object", properties: { expression: { type: "string" } }, required: ["expression"] }
  ) { |args| { result: eval(args["expression"]).to_f } rescue { error: "invalid expression" } }

  tool_config = Candle::GenerationConfig.deterministic(max_length: 500)

  test("chat_with_tools (no execute)", passed, failed) do
    messages = [{ role: "user", content: "Calculate 6 * 7" }]
    result = tool_llm.chat_with_tools(messages, tools: [calculator], config: tool_config)
    raise "expected ToolCallResult" unless result.is_a?(Candle::ToolCallResult)
    result
  end

  test("chat_with_tools (execute: true)", passed, failed) do
    messages = [{ role: "user", content: "Calculate 6 * 7" }]
    result = tool_llm.chat_with_tools(messages, tools: [calculator], execute: true, config: tool_config)
    raise "expected ToolCallResult" unless result.is_a?(Candle::ToolCallResult)
    result
  end

  test("Agent loop", passed, failed) do
    agent = Candle::Agent.new(tool_llm, tools: [calculator], max_iterations: 3)
    result = agent.run("What is 6 * 7?", config: tool_config)
    raise "expected AgentResult" unless result.is_a?(Candle::AgentResult)
    raise "no response" unless result.response
    result
  end

  test("ToolCallParser", passed, failed) do
    output = '<tool_call>{"name": "calculate", "arguments": {"expression": "6*7"}}</tool_call>'
    result = Candle::ToolCallParser.parse(output)
    raise "no tool calls" unless result.has_tool_calls?
    raise "wrong name" unless result.tool_calls.first.name == "calculate"
    result
  end

  # Free model memory
  tool_llm = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

# ============================================================
# GenerationConfig presets
# ============================================================
puts
puts "=" * 80
puts "GenerationConfig"
puts "-" * 80

test("deterministic preset", passed, failed) { Candle::GenerationConfig.deterministic(max_length: 10) }
test("creative preset", passed, failed) { Candle::GenerationConfig.creative(max_length: 10) }
test("balanced preset", passed, failed) { Candle::GenerationConfig.balanced(max_length: 10) }
test(".with override", passed, failed) do
  config = Candle::GenerationConfig.balanced(max_length: 10)
  config2 = config.with(temperature: 0.5)
  raise "temperature not updated" unless config2.temperature == 0.5
  raise "max_length not preserved" unless config2.max_length == 10
  config2
end
test("inspect", passed, failed) { Candle::GenerationConfig.balanced.inspect }

# ============================================================
# Embedding Model
# ============================================================
puts
puts "=" * 80
puts "Embedding Model"
puts "-" * 80

emb = test("load", passed, failed) do
  Candle::EmbeddingModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", device: device)
end

if emb
  test("embedding (pooled_normalized)", passed, failed) do
    e = emb.embedding("Hello, world!")
    raise "empty embedding" if e.values.empty?
    e
  end
  test("embedding (cls)", passed, failed) { emb.embedding("Hello", pooling_method: "cls") }
  test("embedding (pooled)", passed, failed) { emb.embedding("Hello", pooling_method: "pooled") }
  test("embeddings (batch)", passed, failed) do
    results = emb.embeddings("Hello, world!")
    raise "empty" if results.values.empty?
    results
  end
  test("inspect", passed, failed) { emb.inspect }
end

# ============================================================
# Reranker
# ============================================================
puts
puts "=" * 80
puts "Reranker"
puts "-" * 80

reranker = test("load (BERT)", passed, failed) do
  Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2", device: device)
end

if reranker
  docs = ["Ruby is a programming language", "Python is a snake", "Java is an island"]
  test("rerank (pooler)", passed, failed) { reranker.rerank("What is Ruby?", docs, pooling_method: "pooler") }
  test("rerank (cls)", passed, failed) { reranker.rerank("What is Ruby?", docs, pooling_method: "cls") }
  test("rerank (mean)", passed, failed) { reranker.rerank("What is Ruby?", docs, pooling_method: "mean") }
  test("inspect", passed, failed) { reranker.inspect }

  # Free model memory before loading the next one
  reranker = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

bge_reranker = test("load (XLM-RoBERTa / BGE)", passed, failed) do
  Candle::Reranker.from_pretrained("BAAI/bge-reranker-base", device: device)
end

if bge_reranker
  docs = ["Ruby is a programming language", "Python is a snake", "Java is an island"]
  test("rerank (bge)", passed, failed) do
    results = bge_reranker.rerank("What is Ruby?", docs)
    raise "wrong top result" unless results[0][:text].include?("Ruby")
    results
  end
  test("model_id", passed, failed) { bge_reranker.model_id }
  test("tokenizer access", passed, failed) { bge_reranker.tokenizer }
  test("inspect", passed, failed) { bge_reranker.inspect }

  # Free model memory
  bge_reranker = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

# ============================================================
# NER
# ============================================================
puts
puts "=" * 80
puts "NER"
puts "-" * 80

ner = test("load", passed, failed) do
  Candle::NER.from_pretrained("dslim/bert-base-NER", device: device, tokenizer: "bert-base-cased")
end

if ner
  text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
  test("extract_entities", passed, failed) { ner.extract_entities(text) }
  test("entity_types", passed, failed) { ner.entity_types }
  test("analyze", passed, failed) { ner.analyze(text) }
  test("format_entities", passed, failed) { ner.format_entities(text) }
  test("extract_entity_type", passed, failed) { ner.extract_entity_type(text, "PER") }
  test("supports_entity?", passed, failed) { ner.supports_entity?("PER") }
  test("inspect", passed, failed) { ner.inspect }
end

# Pattern & Gazetteer NER
puts
puts "=" * 80
puts "Pattern & Gazetteer NER"
puts "-" * 80

test("PatternEntityRecognizer", passed, failed) do
  rec = Candle::PatternEntityRecognizer.new("EMAIL", [/\b[\w.]+@[\w.]+\.\w+\b/])
  results = rec.recognize("Contact us at hello@example.com")
  raise "no matches" if results.empty?
  results
end

test("GazetteerEntityRecognizer", passed, failed) do
  rec = Candle::GazetteerEntityRecognizer.new("LANG", ["Ruby", "Python", "Rust"])
  results = rec.recognize("I love Ruby and Rust")
  raise "no matches" if results.empty?
  results
end

test("HybridNER", passed, failed) do
  hybrid = Candle::HybridNER.new
  hybrid.add_pattern_recognizer("EMAIL", [/\b[\w.]+@[\w.]+\.\w+\b/])
  hybrid.add_gazetteer_recognizer("LANG", ["Ruby", "Python"])
  results = hybrid.extract_entities("Email ruby@dev.com about Ruby")
  raise "no matches" if results.empty?
  results
end

# ============================================================
# Tokenizer
# ============================================================
puts
puts "=" * 80
puts "Tokenizer"
puts "-" * 80

tok = test("load", passed, failed) { Candle::Tokenizer.from_pretrained("bert-base-uncased") }

if tok
  test("encode", passed, failed) { tok.encode("Hello, world!") }
  test("encode (no special tokens)", passed, failed) { tok.encode("Hello", add_special_tokens: false) }
  test("encode_to_tokens", passed, failed) { tok.encode_to_tokens("Hello, world!") }
  test("encode_with_tokens", passed, failed) do
    result = tok.encode_with_tokens("Hello")
    raise "missing ids" unless result[:ids]
    raise "missing tokens" unless result[:tokens]
    result
  end
  test("encode_batch", passed, failed) { tok.encode_batch(["Hello", "World"]) }
  test("decode", passed, failed) { tok.decode([101, 7592, 102]) }
  test("get_vocab", passed, failed) { tok.get_vocab }
  test("vocab_size", passed, failed) do
    size = tok.vocab_size
    raise "unexpected size" unless size > 1000
    size
  end
  test("id_to_token", passed, failed) { tok.id_to_token(101) }
  test("with_padding", passed, failed) { tok.with_padding(length: 128) }
  test("with_truncation", passed, failed) { tok.with_truncation(512) }
end

# ============================================================
# Tensor
# ============================================================
puts
puts "=" * 80
puts "Tensor"
puts "-" * 80

test("Tensor.new", passed, failed) { Candle::Tensor.new([1.0, 2.0, 3.0]) }
test("Tensor.zeros", passed, failed) { Candle::Tensor.zeros([2, 3]) }
test("Tensor.ones", passed, failed) { Candle::Tensor.ones([2, 3]) }
test("Tensor.rand", passed, failed) { Candle::Tensor.rand([2, 3]) }
test("to_f / to_i", passed, failed) do
  t = Candle::Tensor.new([42.0]).squeeze(0)
  raise "to_f failed" unless t.to_f == 42.0
  raise "to_i failed" unless t.to_i == 42
  t
end
test("each / Enumerable", passed, failed) do
  t = Candle::Tensor.new([1.0, 2.0, 3.0])
  values = t.map { |x| x }
  raise "wrong count" unless values.length == 3
  values
end

# ============================================================
# Summary
# ============================================================
puts
puts "=" * 80
puts "SUMMARY"
puts "=" * 80
puts "  ✅ Passed: #{passed.length}"
if failed.any?
  puts "  ❌ Failed: #{failed.length}"
  failed.each { |f| puts "     - #{f[:name]}: #{f[:error][0..100]}" }
end
puts "  Total: #{passed.length + failed.length}"
puts "=" * 80
exit(failed.empty? ? 0 : 1)
