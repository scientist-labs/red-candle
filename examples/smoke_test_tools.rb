# frozen_string_literal: true

require_relative "smoke_test_helper"

section "Tool Calling"

tool_llm = test("load Qwen3 for tool calling") do
  Candle::LLM.from_pretrained("MaziyarPanahi/Qwen3-0.6B-GGUF",
    gguf_file: "Qwen3-0.6B.Q4_K_M.gguf", tokenizer: "Qwen/Qwen3-0.6B", device: $device)
end

if tool_llm
  calculator = Candle::Tool.new(
    name: "calculate",
    description: "Evaluate a math expression",
    parameters: { type: "object", properties: { expression: { type: "string" } }, required: ["expression"] }
  ) { |args| { result: eval(args["expression"]).to_f } rescue { error: "invalid expression" } }

  tool_config = Candle::GenerationConfig.deterministic(max_length: 500)

  test("chat_with_tools (no execute)") do
    messages = [{ role: "user", content: "Calculate 6 * 7" }]
    result = tool_llm.chat_with_tools(messages, tools: [calculator], config: tool_config)
    raise "expected ToolCallResult" unless result.is_a?(Candle::ToolCallResult)
    result
  end

  test("chat_with_tools (execute: true)") do
    messages = [{ role: "user", content: "Calculate 6 * 7" }]
    result = tool_llm.chat_with_tools(messages, tools: [calculator], execute: true, config: tool_config)
    raise "expected ToolCallResult" unless result.is_a?(Candle::ToolCallResult)
    result
  end

  test("Agent loop") do
    agent = Candle::Agent.new(tool_llm, tools: [calculator], max_iterations: 3)
    result = agent.run("What is 6 * 7?", config: tool_config)
    raise "expected AgentResult" unless result.is_a?(Candle::AgentResult)
    raise "no response" unless result.response
    result
  end

  test("ToolCallParser") do
    output = '<tool_call>{"name": "calculate", "arguments": {"expression": "6*7"}}</tool_call>'
    result = Candle::ToolCallParser.parse(output)
    raise "no tool calls" unless result.has_tool_calls?
    raise "wrong name" unless result.tool_calls.first.name == "calculate"
    result
  end

  tool_llm = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

smoke_test_summary
