# frozen_string_literal: true

require_relative "smoke_test_helper"

section "Build Info"
info = Candle::BuildInfo.summary
puts "  Device: #{info[:default_device]} | Metal: #{info[:metal_available]} | CUDA: #{info[:cuda_available]}"

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
  section "LLM: #{entry[:name]}"

  llm = test("load") do
    if entry[:options]
      Candle::LLM.from_pretrained(entry[:model], device: $device, **entry[:options])
    else
      Candle::LLM.from_pretrained(entry[:model], device: $device)
    end
  end
  next unless llm

  config = Candle::GenerationConfig.balanced(max_length: 30)

  test("generate") { llm.generate("What is Ruby?", config: config) }
  test("generate (2nd call, KV cache)") { llm.generate("The capital of France is", config: config) }
  test("chat") { llm.chat(messages, config: config) }
  test("generate_stream") do
    tokens = []
    llm.generate_stream("Hello", config: config) { |t| tokens << t }
    raise "no tokens streamed" if tokens.empty?
    tokens.join
  end
  test("chat_stream") do
    tokens = []
    llm.chat_stream(messages, config: config) { |t| tokens << t }
    raise "no tokens streamed" if tokens.empty?
    tokens.join
  end
  test("tokenizer access") { llm.tokenizer }
  test("eos_token") { llm.eos_token }
  test("model_id") { llm.model_id }
  test("options") { llm.options }
  test("inspect") { llm.inspect }

  llm = nil
  GC.start(full_mark: true, immediate_sweep: true)
  puts
end

smoke_test_summary
