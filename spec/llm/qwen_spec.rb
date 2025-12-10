require "spec_helper"

RSpec.describe "Qwen LLM" do
  before(:all) do
    @llm = nil
    @model_loaded = false
    @device = Candle::Device.cpu
    
    if ENV["CI"]
      skip "Skipping model download test in CI"
    end
    
    begin
      @llm = Candle::LLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        device: @device,
        gguf_file: "qwen2.5-1.5b-instruct-q4_k_m.gguf"
      )
      @model_loaded = true
    rescue => e
      @model_loaded = :failed
      @load_error = e
    end
  end
  
  before(:each) do
    if @model_loaded == :failed
      skip "Model loading failed: #{@load_error.message}"
    end
  end
  
  describe "tokenizer registry" do
    it "matches Qwen2 patterns" do
      # Qwen2 models get mapped to Qwen2-1.5B tokenizer (base version)
      expect(Candle::LLM.guess_tokenizer("Qwen/Qwen2-7B-GGUF"))
        .to eq("Qwen/Qwen2-1.5B")
      expect(Candle::LLM.guess_tokenizer("Qwen/Qwen2-7B-Instruct-GGUF"))
        .to eq("Qwen/Qwen2-1.5B")
    end
    
    it "matches Qwen2.5 patterns" do
      expect(Candle::LLM.guess_tokenizer("Qwen/Qwen2.5-0.5B-GGUF"))
        .to eq("Qwen/Qwen2.5-0.5B")
      # All Qwen2.5 models map to the 0.5B tokenizer
      expect(Candle::LLM.guess_tokenizer("Qwen/Qwen2.5-1.5B-Instruct-GGUF"))
        .to eq("Qwen/Qwen2.5-0.5B")
      expect(Candle::LLM.guess_tokenizer("Qwen/Qwen2.5-7B-Instruct-GGUF"))
        .to eq("Qwen/Qwen2.5-0.5B")
    end
    
    it "matches third-party Qwen patterns" do
      # Third-party models map to available Qwen tokenizers
      expect(Candle::LLM.guess_tokenizer("bartowski/Qwen2.5-7B-Instruct-GGUF"))
        .to match(/Qwen2.5/)
      expect(Candle::LLM.guess_tokenizer("lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF"))
        .to match(/Qwen2.5.*Coder|Qwen2.5/)
    end
  end
  
  describe "generation" do
    it "generates text" do
      skip unless @model_loaded == true
      
      prompt = "Write a haiku about Ruby:"
      config = Candle::GenerationConfig.new(max_length: 50)
      result = @llm.generate(prompt, config: config)
      
      expect(result).to be_a(String)
      expect(result).not_to be_empty
    end
    
    it "supports streaming generation" do
      skip unless @model_loaded == true
      
      prompt = "Count to 3:"
      chunks = []
      
      config = Candle::GenerationConfig.new(max_length: 30)
      @llm.generate_stream(prompt, config: config) do |chunk|
        chunks << chunk
      end
      
      expect(chunks).not_to be_empty
      expect(chunks.join).to be_a(String)
    end
  end
  
  describe "chat interface" do
    it "handles chat messages" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "user", content: "Hello" }
      ]
      
      config = Candle::GenerationConfig.new(max_length: 50)
      response = @llm.chat(messages, config: config)
      expect(response).to be_a(String)
      expect(response).not_to be_empty
    end
    
    it "applies Qwen chat template" do
      skip unless @model_loaded == true
      
      messages = [
        { role: "system", content: "You are helpful" },
        { role: "user", content: "Test" },
        { role: "assistant", content: "Response" },
        { role: "user", content: "Follow-up" }
      ]
      
      formatted = @llm.apply_chat_template(messages)
      expect(formatted).to include("<|im_start|>")
      expect(formatted).to include("<|im_end|>")
      expect(formatted).to include("system")
      expect(formatted).to include("user")
      expect(formatted).to include("assistant")
    end
  end
  
  describe "metadata" do
    it "has expected model methods" do
      skip unless @model_loaded == true

      expect(@llm).to respond_to(:generate)
      expect(@llm).to respond_to(:chat)
      expect(@llm).to respond_to(:apply_chat_template)
    end
  end

  describe "structured generation" do
    # These tests verify the fix for GPT-2 byte encoding in Qwen models.
    # Qwen uses GPT-2 tokenization where special bytes are represented differently
    # than SentencePiece (e.g., Ġ for space byte vs ▁).

    it "generates valid JSON with string fields" do
      skip unless @model_loaded == true

      schema = {
        type: 'object',
        properties: {
          name: { type: 'string' },
          age: { type: 'integer' }
        },
        required: ['name', 'age']
      }

      result = @llm.generate_structured(
        "Generate a person object for Bob who is 35 years old.",
        schema: schema,
        max_length: 50
      )

      expect(result).to be_a(Hash), "Expected Hash but got #{result.class}: #{result.inspect}"
      expect(result).to have_key("name")
      expect(result).to have_key("age")
      expect(result["name"]).to be_a(String)
      expect(result["age"]).to be_a(Integer)
    end

    it "generates valid JSON with enum fields" do
      skip unless @model_loaded == true

      schema = {
        type: 'object',
        properties: {
          answer: { type: 'string', enum: ['yes', 'no'] },
          confidence: { type: 'number', minimum: 0, maximum: 1 }
        },
        required: ['answer', 'confidence']
      }

      result = @llm.generate_structured(
        "Is Ruby a programming language?",
        schema: schema,
        max_length: 50
      )

      expect(result).to be_a(Hash), "Expected Hash but got #{result.class}: #{result.inspect}"
      expect(['yes', 'no']).to include(result['answer'])
      expect(result['confidence']).to be_a(Numeric)
    end

    it "does not produce truncated JSON output" do
      skip unless @model_loaded == true

      # This test specifically checks the bug where Qwen would produce
      # truncated output like '{"name":",' due to early constraint termination
      schema = {
        type: 'object',
        properties: {
          name: { type: 'string' },
          city: { type: 'string' }
        },
        required: ['name', 'city']
      }

      constraint = @llm.constraint_from_schema(schema)
      config = Candle::GenerationConfig.balanced(
        constraint: constraint,
        max_length: 100
      )

      raw_result = @llm.generate("Generate a person in New York:", config: config)

      # The raw result should be valid JSON, not truncated
      parsed = JSON.parse(raw_result) rescue nil
      expect(parsed).to be_a(Hash), "Raw output should be valid JSON, got: #{raw_result.inspect}"
      expect(parsed).to have_key("name")
      expect(parsed).to have_key("city")
    end

    it "creates constraints using from_schema_with_model" do
      skip unless @model_loaded == true

      # Test that StructuredConstraint.from_schema_with_model works correctly
      # This uses Vocabulary::from_pretrained for proper byte encoding
      schema = JSON.generate({
        type: 'object',
        properties: {
          value: { type: 'integer' }
        },
        required: ['value']
      })

      constraint = Candle::StructuredConstraint.from_schema_with_model(
        schema,
        "Qwen/Qwen2.5-0.5B"
      )

      expect(constraint).to be_a(Candle::StructuredConstraint)

      # Use the constraint in generation
      config = Candle::GenerationConfig.balanced(
        constraint: constraint,
        max_length: 30
      )

      result = @llm.generate("Generate a number:", config: config)
      parsed = JSON.parse(result) rescue nil
      expect(parsed).to be_a(Hash), "Should produce valid JSON: #{result.inspect}"
    end
  end
end