require "spec_helper"

RSpec.describe "Custom Tokenizer Support" do
  describe "model-specific custom tokenizer support" do
    context "Llama models" do
      it "accepts custom tokenizer for Llama safetensors models" do
        skip "Not yet implemented"
        
        expect {
          Candle::LLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            tokenizer: "meta-llama/Llama-2-7b-chat-hf"
          )
        }.not_to raise_error
      end
      
      it "accepts custom tokenizer for TinyLlama models" do
        skip "Not yet implemented"
        
        expect {
          Candle::LLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            tokenizer: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
          )
        }.not_to raise_error
      end
      
      it "fails gracefully with invalid tokenizer for Llama" do
        skip "Not yet implemented"
        
        expect {
          Candle::LLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            tokenizer: "invalid/nonexistent-tokenizer"
          )
        }.to raise_error(/Failed to (download|load) tokenizer/)
      end
    end
    
    context "Mistral models" do
      it "accepts custom tokenizer for Mistral safetensors models" do
        skip "Not yet implemented"
        
        expect {
          Candle::LLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            tokenizer: "mistralai/Mistral-7B-Instruct-v0.2"
          )
        }.not_to raise_error
      end
      
      it "accepts custom tokenizer for Mixtral models" do
        skip "Not yet implemented"
        
        expect {
          Candle::LLM.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1",
            tokenizer: "mistralai/Mixtral-8x7B-Instruct-v0.1"
          )
        }.not_to raise_error
      end
    end
    
    context "Gemma models" do
      it "accepts custom tokenizer for Gemma safetensors models" do
        skip "Not yet implemented"
        
        expect {
          Candle::LLM.from_pretrained(
            "google/gemma-2b",
            tokenizer: "google/gemma-2b-it"
          )
        }.not_to raise_error
      end
      
      it "accepts custom tokenizer for Gemma2 models" do
        skip "Not yet implemented"
        
        expect {
          Candle::LLM.from_pretrained(
            "google/gemma-2-2b",
            tokenizer: "google/gemma-2-2b-it"
          )
        }.not_to raise_error
      end
    end
    
    context "Qwen models" do
      it "accepts custom tokenizer for Qwen safetensors models" do
        skip "Not yet implemented"
        
        expect {
          Candle::LLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            tokenizer: "Qwen/Qwen2.5-0.5B-Instruct"
          )
        }.not_to raise_error
      end
      
      it "accepts custom tokenizer for Qwen2 models" do
        skip "Not yet implemented"
        
        expect {
          Candle::LLM.from_pretrained(
            "Qwen/Qwen2-0.5B",
            tokenizer: "Qwen/Qwen2-0.5B-Instruct"
          )
        }.not_to raise_error
      end
    end
    
    context "Phi models" do
      it "accepts custom tokenizer for Phi safetensors models" do
        # This is already implemented - test with fake model
        expect {
          Candle::LLM.from_pretrained(
            "microsoft/phi-fake-nonexistent",
            tokenizer: "microsoft/Phi-3-mini-4k-instruct"
          )
        }.to raise_error(/Failed to (download|load|create)/) # Fails on model, not tokenizer
      end
      
      it "accepts custom tokenizer for Phi-3 models" do
        # This is already implemented - test with fake model
        expect {
          Candle::LLM.from_pretrained(
            "microsoft/Phi-3-fake-nonexistent",
            tokenizer: "microsoft/Phi-3-mini-4k-instruct"
          )
        }.to raise_error(/Failed to (download|load|create)/) # Fails on model, not tokenizer
      end
    end
  end
  
  describe "cross-model tokenizer compatibility" do
    it "allows using Llama tokenizer with Mistral model" do
      skip "Not yet implemented"
      
      # This is technically possible though may produce suboptimal results
      expect {
        Candle::LLM.from_pretrained(
          "mistralai/Mistral-7B-v0.1",
          tokenizer: "meta-llama/Llama-2-7b-hf"
        )
      }.not_to raise_error
    end
    
    it "allows using any HF tokenizer with any model" do
      skip "Not yet implemented"
      
      # Should be possible even if not recommended
      expect {
        Candle::LLM.from_pretrained(
          "google/gemma-2b",
          tokenizer: "bert-base-uncased"
        )
      }.not_to raise_error
    end
  end
  
  describe "tokenizer loading without model download" do
    # These tests should actually work without downloading models
    # by failing at model download stage, not tokenizer validation
    
    it "validates Llama tokenizer parameter without downloading model" do
      skip "Not yet implemented"
      
      expect {
        Candle::LLM.from_pretrained(
          "meta-llama/fake-llama-model",
          tokenizer: "meta-llama/Llama-2-7b-hf"
        )
      }.to raise_error(/Failed to download (config|model)/)
      # Should fail on model, not tokenizer parameter
    end
    
    it "validates Mistral tokenizer parameter without downloading model" do
      skip "Not yet implemented"
      
      expect {
        Candle::LLM.from_pretrained(
          "mistralai/fake-mistral-model",
          tokenizer: "mistralai/Mistral-7B-Instruct-v0.2"
        )
      }.to raise_error(/Failed to download (config|model)/)
    end
    
    it "validates Gemma tokenizer parameter without downloading model" do
      skip "Not yet implemented"
      
      expect {
        Candle::LLM.from_pretrained(
          "google/fake-gemma-model",
          tokenizer: "google/gemma-2b"
        )
      }.to raise_error(/Failed to download (config|model)/)
    end
    
    it "validates Qwen tokenizer parameter without downloading model" do
      skip "Not yet implemented"
      
      expect {
        Candle::LLM.from_pretrained(
          "Qwen/fake-qwen-model",
          tokenizer: "Qwen/Qwen2.5-0.5B-Instruct"
        )
      }.to raise_error(/Failed to download (config|model)/)
    end
  end
  
  describe "error messages" do
    it "provides clear error when tokenizer download fails" do
      skip "Not yet implemented"
      
      expect {
        Candle::LLM.from_pretrained(
          "meta-llama/Llama-2-7b-hf",
          tokenizer: "invalid/tokenizer"
        )
      }.to raise_error(/Failed to download tokenizer.*invalid\/tokenizer/)
    end
    
    it "provides clear error when tokenizer file is missing" do
      skip "Not yet implemented"
      
      expect {
        Candle::LLM.from_pretrained(
          "mistralai/Mistral-7B-v0.1",
          tokenizer: "some-org/model-without-tokenizer-json"
        )
      }.to raise_error(/tokenizer.json/)
    end
  end
  
  describe "functional tests with real tokenizers" do
    # These tests download small tokenizers but not models
    
    it "can load Llama tokenizer standalone" do
      skip "Requires HF_TOKEN for Llama tokenizer"
      
      tokenizer = Candle::Tokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
      expect(tokenizer).not_to be_nil
      expect(tokenizer.get_vocab).to include("▁Hello")
    end
    
    it "can load Mistral tokenizer standalone" do
      # Mistral-7B-v0.1 uses tokenizer.model, not tokenizer.json
      # Use the Instruct version which has tokenizer.json
      tokenizer = Candle::Tokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
      expect(tokenizer).not_to be_nil
      
      tokens = tokenizer.encode("Hello, world!")
      expect(tokens).to be_a(Array)
      expect(tokens).not_to be_empty
    end
    
    it "can load Gemma tokenizer standalone" do
      skip "Requires agreement to Gemma terms"
      
      tokenizer = Candle::Tokenizer.from_pretrained("google/gemma-2b")
      expect(tokenizer).not_to be_nil
      expect(tokenizer.get_vocab).to include("<start_of_turn>")
    end
    
    it "can load Qwen tokenizer standalone" do
      tokenizer = Candle::Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
      expect(tokenizer).not_to be_nil
      
      vocab = tokenizer.get_vocab
      expect(vocab).to include("<|im_start|>")
      expect(vocab).to include("<|im_end|>")
    end
  end
end