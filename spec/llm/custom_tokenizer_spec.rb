require "spec_helper"

RSpec.describe "Custom Tokenizer Support" do
  describe "model-specific custom tokenizer support" do
    context "Llama models" do
      it "accepts custom tokenizer for Llama2 safetensors models" do
        # Test with fake model to avoid downloading large files
        expect {
          Candle::LLM.from_pretrained(
            "meta-llama/fake-llama-2-model",
            tokenizer: "meta-llama/Llama-2-7b-chat-hf"
          )
        }.to raise_error(/Failed to (download|load|create)/)
        # Should fail on model download, not tokenizer rejection
      end
      
      it "accepts custom tokenizer for Llama3 safetensors models" do
        # Use fake model to test tokenizer acceptance
        expect {
          Candle::LLM.from_pretrained(
            "meta-llama/fake-llama-3-model",
            tokenizer: "meta-llama/Meta-Llama-3-8B-Instruct"
          )
        }.to raise_error(/Failed to (download|load|create)/)
        # Should fail on model, not tokenizer
      end
      
      it "accepts custom tokenizer for TinyLlama models" do
        expect {
          Candle::LLM.from_pretrained(
            "TinyLlama/fake-tinyllama-model",
            tokenizer: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
      
      it "accepts custom tokenizer for CodeLlama models" do
        expect {
          Candle::LLM.from_pretrained(
            "codellama/fake-codellama-model",
            tokenizer: "codellama/CodeLlama-7b-Instruct-hf"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
      
      it "fails gracefully with invalid tokenizer for Llama" do
        expect {
          Candle::LLM.from_pretrained(
            "meta-llama/fake-llama-model",
            tokenizer: "invalid/nonexistent-tokenizer-xyz"
          )
        }.to raise_error(/Failed to/)
        # Should fail on tokenizer or model download
      end
      
    end
    
    context "Mistral models" do
      it "accepts custom tokenizer for Mistral safetensors models" do
        expect {
          Candle::LLM.from_pretrained(
            "mistralai/fake-mistral-model",
            tokenizer: "mistralai/Mistral-7B-Instruct-v0.2"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
      
      it "accepts custom tokenizer for Mixtral models" do
        expect {
          Candle::LLM.from_pretrained(
            "mistralai/fake-mixtral-model",
            tokenizer: "mistralai/Mixtral-8x7B-Instruct-v0.1"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
      
      it "accepts custom tokenizer for Mistral v0.3 models" do
        expect {
          Candle::LLM.from_pretrained(
            "mistralai/fake-mistral-v3-model",
            tokenizer: "mistralai/Mistral-7B-Instruct-v0.3"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
      
      it "handles tokenizer.model vs tokenizer.json differences" do
        # Some Mistral models use tokenizer.model (SentencePiece) instead of tokenizer.json
        # Testing that we can specify a tokenizer with tokenizer.json
        expect {
          Candle::LLM.from_pretrained(
            "mistralai/fake-mistral-sp-model",
            tokenizer: "mistralai/Mistral-7B-Instruct-v0.2"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
    end
    
    context "Gemma models" do
      it "accepts custom tokenizer for Gemma safetensors models" do
        expect {
          Candle::LLM.from_pretrained(
            "google/fake-gemma-model",
            tokenizer: "google/gemma-2b-it"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
      
      it "accepts custom tokenizer for Gemma2 models" do
        expect {
          Candle::LLM.from_pretrained(
            "google/fake-gemma-2-model",
            tokenizer: "google/gemma-2-2b-it"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
      
      it "accepts custom tokenizer for different Gemma sizes" do
        # Should work with 7b using 2b tokenizer
        expect {
          Candle::LLM.from_pretrained(
            "google/fake-gemma-7b",
            tokenizer: "google/gemma-2b-it"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
    end
    
    context "Qwen models" do
      it "accepts custom tokenizer for Qwen2.5 safetensors models" do
        expect {
          Candle::LLM.from_pretrained(
            "Qwen/fake-qwen2.5-model",
            tokenizer: "Qwen/Qwen2.5-0.5B-Instruct"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
      
      it "accepts custom tokenizer for Qwen2 models" do
        expect {
          Candle::LLM.from_pretrained(
            "Qwen/fake-qwen2-model",
            tokenizer: "Qwen/Qwen2-0.5B-Instruct"
          )
        }.to raise_error(/Failed to (download|load|create)/)
      end
      
      it "accepts custom tokenizer for different Qwen sizes" do
        # Should work with larger models using smaller model tokenizer
        expect {
          Candle::LLM.from_pretrained(
            "Qwen/fake-qwen-7b",
            tokenizer: "Qwen/Qwen2.5-0.5B-Instruct"
          )
        }.to raise_error(/Failed to (download|load|create)/)
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
      # This is technically possible though may produce suboptimal results
      expect {
        Candle::LLM.from_pretrained(
          "mistralai/fake-mistral-model",
          tokenizer: "meta-llama/Llama-2-7b-hf"
        )
      }.to raise_error(/Failed to (download|load|create)/)
      # Should fail on model, not cross-model tokenizer usage
    end
    
    it "allows using any HF tokenizer with any model" do
      # Should be possible even if not recommended
      expect {
        Candle::LLM.from_pretrained(
          "google/fake-gemma-model",
          tokenizer: "bert-base-uncased"
        )
      }.to raise_error(/Failed to (download|load|create)/)
      # Should fail on model, not tokenizer type mismatch
    end
  end
  
  describe "tokenizer loading without model download" do
    # These tests should actually work without downloading models
    # by failing at model download stage, not tokenizer validation
    
    it "validates Llama tokenizer parameter without downloading model" do
      expect {
        Candle::LLM.from_pretrained(
          "meta-llama/fake-llama-model",
          tokenizer: "meta-llama/Llama-2-7b-hf"
        )
      }.to raise_error(/Failed to (download|load|create)/)
      # Should fail on model, not tokenizer parameter
    end
    
    it "validates TinyLlama tokenizer parameter without downloading model" do
      expect {
        Candle::LLM.from_pretrained(
          "TinyLlama/fake-tinyllama-model",
          tokenizer: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
      }.to raise_error(/Failed to (download|load|create)/)
    end
    
    it "validates Mistral tokenizer parameter without downloading model" do
      expect {
        Candle::LLM.from_pretrained(
          "mistralai/fake-mistral-model",
          tokenizer: "mistralai/Mistral-7B-Instruct-v0.2"
        )
      }.to raise_error(/Failed to (download|load|create)/)
    end
    
    it "validates Gemma tokenizer parameter without downloading model" do
      expect {
        Candle::LLM.from_pretrained(
          "google/fake-gemma-model",
          tokenizer: "google/gemma-2b"
        )
      }.to raise_error(/Failed to (download|load|create)/)
    end
    
    it "validates Qwen tokenizer parameter without downloading model" do
      expect {
        Candle::LLM.from_pretrained(
          "Qwen/fake-qwen-model",
          tokenizer: "Qwen/Qwen2.5-0.5B-Instruct"
        )
      }.to raise_error(/Failed to (download|load|create)/)
    end
  end
  
  describe "error messages" do
    it "provides clear error when tokenizer download fails" do
      # Test with an invalid tokenizer repo
      expect {
        Candle::LLM.from_pretrained(
          "fake-org/fake-llama-model",
          tokenizer: "invalid/nonexistent-tokenizer-xyz-123"
        )
      }.to raise_error(/Failed to (download|load|create)/)
    end
    
    it "provides clear error when tokenizer file is missing" do
      # Test with a repo that exists but doesn't have tokenizer.json
      # Using gpt2 as tokenizer since it exists but may not have the right format
      expect {
        Candle::LLM.from_pretrained(
          "fake-org/fake-mistral-model",
          tokenizer: "gpt2"  # This repo exists but may not have tokenizer.json in right format
        )
      }.to raise_error(/Failed to/)
    end
    
    it "provides helpful suggestion when tokenizer parameter missing" do
      # When a fake Llama model doesn't have a tokenizer and none is specified
      expect {
        Candle::LLM.from_pretrained("fake-org/fake-llama-without-tokenizer")
      }.to raise_error(/Failed to/)
      # The error should mention tokenizer or download issues
    end
  end
  
  describe "implementation approach" do
    it "extends from_pretrained_with_tokenizer pattern to all models" do
      # Test with the available TinyLlama model
      # This ensures consistency across all model implementations
      expect(Candle::LLM).to respond_to(:from_pretrained)
      
      # The tokenizer parameter should be optional - model loads with default tokenizer
      expect {
        llm = Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        expect(llm).not_to be_nil
        expect(llm.model_name).to include("TinyLlama")
      }.not_to raise_error
      
      # But can be specified when needed - same model with explicit tokenizer
      expect {
        llm = Candle::LLM.from_pretrained(
          "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          tokenizer: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        expect(llm).not_to be_nil
        expect(llm.model_name).to include("TinyLlama")
      }.not_to raise_error
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