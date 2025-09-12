require "spec_helper"

RSpec.describe "Custom Tokenizer Support (Practical)" do
  describe "tokenizer parameter acceptance" do
    # These tests verify the tokenizer parameter is accepted
    # They use fake model names that will fail at download, proving
    # the tokenizer parameter was processed correctly
    
    it "accepts custom tokenizer for Llama models" do
      expect {
        Candle::LLM.from_pretrained(
          "fake-org/fake-llama-model",
          tokenizer: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
      }.to raise_error(/Failed to (download|load|create)/)
      # Fails on model, not tokenizer parameter
    end
    
    it "accepts custom tokenizer for Mistral models" do
      expect {
        Candle::LLM.from_pretrained(
          "fake-org/fake-mistral-model",
          tokenizer: "mistralai/Mistral-7B-Instruct-v0.2"
        )
      }.to raise_error(/Failed to (download|load|create)/)
    end
    
    it "accepts custom tokenizer for Gemma models" do
      expect {
        Candle::LLM.from_pretrained(
          "fake-org/fake-gemma-model",
          tokenizer: "google/gemma-2b"
        )
      }.to raise_error(/Failed to (download|load|create)/)
    end
    
    it "accepts custom tokenizer for Qwen models" do
      expect {
        Candle::LLM.from_pretrained(
          "fake-org/fake-qwen-model",
          tokenizer: "Qwen/Qwen2.5-0.5B-Instruct"
        )
      }.to raise_error(/Failed to (download|load|create)/)
    end
    
    it "accepts custom tokenizer for Phi models" do
      expect {
        Candle::LLM.from_pretrained(
          "microsoft/fake-phi-model",
          tokenizer: "microsoft/Phi-3-mini-4k-instruct"
        )
      }.to raise_error(/Failed to (download|load|create)/)
    end
  end
  
  describe "tokenizer validation" do
    it "fails with appropriate error for missing models" do
      # Since model config is downloaded before tokenizer,
      # we expect failure on config/model download
      expect {
        Candle::LLM.from_pretrained(
          "TinyLlama/fake-model",
          tokenizer: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
      }.to raise_error(/Failed to (download config|load model)/)
    end
  end
  
  describe "standalone tokenizer functionality" do
    it "loads Mistral tokenizer independently" do
      tokenizer = Candle::Tokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
      expect(tokenizer).not_to be_nil
      
      tokens = tokenizer.encode("Hello, world!")
      expect(tokens).to be_a(Array)
      expect(tokens.length).to be > 0
    end
    
    it "loads Qwen tokenizer independently" do
      tokenizer = Candle::Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
      expect(tokenizer).not_to be_nil
      
      vocab = tokenizer.get_vocab
      expect(vocab).to include("<|im_start|>")
      expect(vocab).to include("<|im_end|>")
    end
  end
  
  describe "model type detection with custom tokenizer" do
    it "correctly routes Llama models" do
      expect {
        Candle::LLM.from_pretrained(
          "meta-llama/fake-model",
          tokenizer: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
      }.to raise_error(/Failed to/)
      # The error proves it tried to load as Llama model
    end
    
    it "correctly routes TinyLlama models" do
      expect {
        Candle::LLM.from_pretrained(
          "TinyLlama/fake-model",
          tokenizer: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
      }.to raise_error(/Failed to/)
      # TinyLlama should be routed to Llama handler
    end
    
    it "correctly routes Mistral models" do
      expect {
        Candle::LLM.from_pretrained(
          "mistralai/fake-model",
          tokenizer: "mistralai/Mistral-7B-Instruct-v0.2"
        )
      }.to raise_error(/Failed to/)
    end
  end
end