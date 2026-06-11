require "candle"

# Load a model that supports tool calling
llm = Candle::LLM.from_pretrained(
  "MaziyarPanahi/Qwen3-0.6B-GGUF",
  gguf_file: "Qwen3-0.6B.Q4_K_M.gguf",
  tokenizer: "Qwen/Qwen3-0.6B"
)

# Define a tool the model can't answer without — it doesn't know current weather
get_weather = Candle::Tool.new(
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: {
    type: "object",
    properties: { city: { type: "string", description: "City name" } },
    required: ["city"]
  }
) { |args| { city: args["city"], temperature: 72, condition: "sunny", humidity: 45 } }

config = Candle::GenerationConfig.deterministic(max_length: 500)

# --- Default: parse tool calls without executing ---
puts "=== Tool Calling (parse only) ==="
messages = [{ role: "user", content: "What's the weather in San Francisco?" }]
response = llm.chat(messages, tools: [get_weather], config: config)

if response.tool_calls?
  response.tool_calls.each do |tc|
    puts "Model wants to call: #{tc.name}(#{tc.arguments})"

    # You execute the tool yourself
    output = get_weather.call(tc.arguments)
    puts "Result: #{output}"
  end
else
  puts "No tool call, text response: #{response.content}"
end

# --- With execute: true, tools run automatically ---
puts
puts "=== Tool Calling (auto-execute) ==="
messages = [{ role: "user", content: "What's the weather in Tokyo?" }]
response = llm.chat(messages, tools: [get_weather], execute: true, config: config)

if response.tool_calls?
  response.tool_calls.each do |tc|
    puts "#{tc.name}(#{tc.arguments}) => #{tc.error || tc.result}"
  end
else
  puts "Text response: #{response.content}"
end
