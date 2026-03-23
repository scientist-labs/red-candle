require "candle"

# Load a model that supports tool calling
# Larger models (4B+) handle multi-step reasoning better
llm = Candle::LLM.from_pretrained(
  "MaziyarPanahi/Qwen3-4B-GGUF",
  gguf_file: "Qwen3-4B.Q4_K_M.gguf",
  tokenizer: "Qwen/Qwen3-4B"
)

# Define tools the model can't answer without
get_weather = Candle::Tool.new(
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: {
    type: "object",
    properties: { city: { type: "string", description: "City name" } },
    required: ["city"]
  }
) { |args| { city: args["city"], temperature: 72, condition: "sunny", humidity: 45 } }

lookup_price = Candle::Tool.new(
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
agent = Candle::Agent.new(llm, tools: [get_weather, lookup_price], max_iterations: 5)
result = agent.run("What's the weather in Paris, and how much does a widget cost?", config: config)

puts "--- Conversation ---"
result.messages.each do |m|
  content = m[:content].to_s.gsub(/<think>.*?<\/think>/m, "").strip
  next if content.empty?
  puts "#{m[:role]}: #{content[0..200]}"
  puts
end
puts "--- Summary ---"
puts "Final answer: #{result.response.to_s.gsub(/<think>.*?<\/think>/m, "").strip}"
puts "Iterations: #{result.iterations}"
puts "Tool calls made: #{result.tool_calls_made}"
