require "candle"

# Load a model that supports tool calling
# Larger models (4B+) handle multi-step reasoning better
llm = Candle::LLM.from_pretrained(
  "MaziyarPanahi/Qwen3-4B-GGUF",
  gguf_file: "Qwen3-4B.Q4_K_M.gguf",
  tokenizer: "Qwen/Qwen3-4B"
)

# Define tools
calculator = Candle::Tool.new(
  name: "calculate",
  description: "Evaluate a math expression and return the result",
  parameters: {
    type: "object",
    properties: { expression: { type: "string" } },
    required: ["expression"]
  }
) { |args| eval(args["expression"]).to_f }

lookup = Candle::Tool.new(
  name: "lookup_price",
  description: "Look up the unit price of a product in dollars",
  parameters: {
    type: "object",
    properties: { product: { type: "string" } },
    required: ["product"]
  }
) { |args| { product: args["product"], unit_price: 9.99 } }

config = Candle::GenerationConfig.deterministic(max_length: 1000)

# Agent handles the multi-turn loop automatically:
# generate -> parse tool calls -> execute -> feed results back -> repeat
agent = Candle::Agent.new(llm, tools: [calculator, lookup], max_iterations: 5)
result = agent.run("What is the price of a widget, and how much would 3 cost?", config: config)

puts "Answer: #{result.response}"
puts "Iterations: #{result.iterations}"
puts "Tool calls made: #{result.tool_calls_made}"
puts
puts "--- Full conversation ---"
result.messages.each do |m|
  puts "#{m[:role]}: #{m[:content][0..200]}"
  puts "---"
end
