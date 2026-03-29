# frozen_string_literal: true

require_relative "smoke_test_helper"

section "Structured Generation"

struct_llm = test("load TinyLlama for structured gen") do
  Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", device: $device)
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

  test("constraint_from_schema") { struct_llm.constraint_from_schema(schema) }
  test("constraint_from_regex") { struct_llm.constraint_from_regex('\d+') }
  test("generate_structured") do
    result = struct_llm.generate_structured("Is Ruby a language?", schema: schema, max_length: 50)
    raise "not a Hash" unless result.is_a?(Hash)
    result
  end
  test("generate_regex") do
    struct_llm.generate_regex("The answer is", pattern: '\d+', max_length: 10)
  end

  struct_llm = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

smoke_test_summary
