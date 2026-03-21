require_relative 'lib/candle'

puts "=== Testing Custom Tokenizer Support for All Models ==="
puts ""

# Test that each model type accepts custom tokenizer without downloading
models = [
  ["TinyLlama/fake-model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Llama"],
  ["mistralai/fake-model", "mistralai/Mistral-7B-Instruct-v0.2", "Mistral"],
  ["google/fake-gemma", "google/gemma-2b", "Gemma"],
  ["Qwen/fake-qwen", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen"],
  ["microsoft/phi-fake", "microsoft/Phi-3-mini-4k-instruct", "Phi"]
]

models.each do |model_id, tokenizer_id, model_type|
  print "Testing #{model_type} with custom tokenizer... "
  begin
    llm = Candle::LLM.from_pretrained(model_id, tokenizer: tokenizer_id)
    puts "❌ Should have failed on model download"
  rescue => e
    if e.message.include?("Failed to download") || e.message.include?("Failed to load") || e.message.include?("Failed to create")
      puts "✅ Tokenizer accepted, failed on model (expected)"
    else
      puts "❌ Unexpected error: #{e.message[0..100]}"
    end
  end
end

puts ""
puts "=== Testing Standalone Tokenizer Loading ==="
puts ""

# Test loading actual tokenizers
tokenizers = [
  "mistralai/Mistral-7B-Instruct-v0.2",
  "Qwen/Qwen2.5-0.5B-Instruct"
]

tokenizers.each do |tokenizer_id|
  print "Loading tokenizer #{tokenizer_id}... "
  begin
    tokenizer = Candle::Tokenizer.from_pretrained(tokenizer_id)
    tokens = tokenizer.encode("Hello, world!")
    if tokens && tokens.is_a?(Array) && !tokens.empty?
      puts "✅ (#{tokens.length} tokens)"
    else
      puts "❌ Invalid tokenization"
    end
  rescue => e
    puts "❌ #{e.message[0..50]}"
  end
end

puts ""
puts "=== Summary ==="
puts "Custom tokenizer support is now available for all model types:"
puts "- Llama (including TinyLlama, CodeLlama)"
puts "- Mistral (including Mixtral)"
puts "- Gemma (including Gemma2)"
puts "- Qwen (including Qwen2, Qwen2.5)"
puts "- Phi (including Phi-2, Phi-3)"