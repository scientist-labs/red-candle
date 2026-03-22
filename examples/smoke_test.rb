require "candle"

device = Candle::Device.metal
models = [
  {
    model: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    options: [
      { gguf_file: "mistral-7b-instruct-v0.2.Q4_K_M.gguf", tokenizer: "mistralai/Mistral-7B-Instruct-v0.2" }
    ]
  },
  {
    model: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    options: [
      { gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf" },
      { gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" },
      { gguf_file:"tinyllama-1.1b-chat-v1.0.Q5_0.gguf" }
    ]
  },
  {
    model: "Qwen/Qwen3-0.6B"
  },
  {
    model: "google/gemma-3-4b-it-qat-q4_0-gguf",
    options: [
      { gguf_file: "gemma-3-4b-it-q4_0.gguf", tokenizer: "google/gemma-3-4b-it" }
    ]
  }
]

messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is Ruby?" }
]

config = Candle::GenerationConfig.balanced(debug_tokens: false, max_length: 50)
models.each do |entry|
  model = entry[:model]
  puts "-" * 80
  puts "-" * 80
  options = entry[:options]
  puts "#{model} - #{options}"
  if options
    options.each do |option|
      llm = Candle::LLM.from_pretrained(model, device: device, **option)
      puts "-" * 80
      # llm.generate_stream("What is Ruby?", config: config) { |t| print t }
      # puts llm.generate("Question: What is Ruby?\nAnswer:", config: config)
      # puts llm.chat(messages, config: config)
      puts llm.chat_stream(messages, config: config) { |t| print t }
      puts "-" * 80
    end
  else
    llm = Candle::LLM.from_pretrained(model, device: device)
    puts "-" * 80
    # llm.generate_stream("What is Ruby?", config: config) { |t| print t }
    # puts llm.generate("Question: What is Ruby?\nAnswer:", config: config)
    # puts llm.chat(messages, config: config)
    puts llm.chat_stream(messages, config: config) { |t| print t }
    puts "-" * 80
  end
end


model = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", device: device, gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf")
puts model.generate("Question: What is Ruby?\nAnswer:", config: config)

model = Candle::EmbeddingModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", device: device)
embedding = model.embedding("Hi there!")


query = "What is the capital of England?"
doc1 = "London is the capital of England"
doc2 = "Paris is the capital of France"
doc3 = "Berlin is the capital of Germany"
reranker = Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2", device: device)
results = reranker.rerank(query, [doc1, doc2, doc3])



ner = Candle::NER.from_pretrained("dslim/bert-base-NER", device: device, tokenizer: "bert-base-cased")
text = "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
entities = ner.extract_entities(text)
entities.each do |entity|
  puts "#{entity[:text]} (#{entity[:label]}) - confidence: #{entity[:confidence].round(2)}"
end


tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
token_ids = tokenizer.encode("Hello, world!")

llm = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", gguf_file: "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf", device: Candle::Device.metal)

schema = {
  type: "object",
  properties: {
    answer: { type: "string", enum: ["yes", "no"] },
    confidence: { type: "number", minimum: 0, maximum: 1 }
  },
  required: ["answer", "confidence"]
}
constraint = llm.constraint_from_schema(schema)
config = Candle::GenerationConfig.balanced(
  constraint: constraint,
  max_length: 50
)
prompt = "Is Ruby a programming language?"
result = llm.generate(prompt, config: config)

result = llm.generate_structured(prompt, schema: schema)







schema = {
  type: "object",
  properties: {
    name: { type: "string" },
    type: {
      type: "string",
      enum: ["person", "organization", "location"]
    }
  },
  required: ["name", "type"]
}
constraint = llm.constraint_from_schema(schema)
config = Candle::GenerationConfig.balanced(
  constraint: constraint,
  max_length: 500
)
prompt = "Extract the person's name from: John Smith is a developer."
result = llm.generate(prompt, config: config)

result = llm.generate_structured(prompt, schema: schema)
