require "candle"

# Load a model that supports tool calling
llm = Candle::LLM.from_pretrained(
  "MaziyarPanahi/Qwen3-0.6B-GGUF",
  gguf_file: "Qwen3-0.6B.Q4_K_M.gguf",
  tokenizer: "Qwen/Qwen3-0.6B"
)

# Define tools
calculator = Candle::Tool.new(
  name: "calculate",
  description: "Evaluate a math expression and return the result",
  parameters: {
    type: "object",
    properties: { expression: { type: "string", description: "Math expression to evaluate" } },
    required: ["expression"]
  }
) { |args| { result: eval(args["expression"]).to_f } }

config = Candle::GenerationConfig.deterministic(max_length: 500)

# --- Default: parse tool calls without executing ---
puts "=== Tool Calling (parse only) ==="
messages = [{ role: "user", content: "Calculate 42 * 7" }]
result = llm.chat_with_tools(messages, tools: [calculator], config: config)

if result.has_tool_calls?
  result.tool_calls.each do |tc|
    puts "Model wants to call: #{tc.name}(#{tc.arguments})"

    # You execute the tool yourself
    output = calculator.call(tc.arguments)
    puts "Result: #{output}"
  end
else
  puts "No tool call, text response: #{result.text_response}"
end

# --- With execute: true, tools run automatically ---
puts
puts "=== Tool Calling (auto-execute) ==="
result = llm.chat_with_tools(messages, tools: [calculator], execute: true, config: config)

if result.has_tool_calls?
  result.tool_results.each do |tr|
    puts "#{tr[:tool_call].name}(#{tr[:tool_call].arguments}) => #{tr[:result]}"
  end
end
