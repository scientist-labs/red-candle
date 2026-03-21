# frozen_string_literal: true

require "spec_helper"

RSpec.describe "Chat Template BOS Handling" do
  let(:messages_with_system) do
    [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hello" }
    ]
  end

  let(:messages_without_system) do
    [
      { role: "user", content: "Hello" }
    ]
  end

  let(:multi_turn_messages) do
    [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello!" },
      { role: "user", content: "How are you?" }
    ]
  end

  context "with TinyLlama (Llama 2 template)", :llm do
    before(:all) do
      @llm = begin
        Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
      rescue => e
        nil
      end
      @tokenizer = @llm&.tokenizer
    end

    after(:all) do
      @llm = nil
      @tokenizer = nil
      GC.start
    end

    it "does not hardcode <s> BOS in the template" do
      skip "Model not loaded" unless @llm
      formatted = @llm.apply_chat_template(messages_with_system)
      expect(formatted).not_to start_with("<s>")
    end

    it "produces no duplicate BOS tokens when encoded" do
      skip "Model not loaded" unless @llm
      formatted = @llm.apply_chat_template(messages_with_system)
      tokens = @tokenizer.encode(formatted)

      bos_id = @tokenizer.get_vocab["<s>"]
      bos_count = tokens.count { |id| id == bos_id }
      expect(bos_count).to be <= 1, "Expected at most 1 BOS token, got #{bos_count}. Template may hardcode <s>"
    end

    it "includes [INST] markers in Llama 2 format" do
      skip "Model not loaded" unless @llm
      formatted = @llm.apply_chat_template(messages_with_system)
      expect(formatted).to include("[INST]")
      expect(formatted).to include("[/INST]")
    end

    it "includes system message content" do
      skip "Model not loaded" unless @llm
      formatted = @llm.apply_chat_template(messages_with_system)
      expect(formatted).to include("You are helpful.")
      expect(formatted).to include("Hello")
    end

    it "handles messages without system prompt" do
      skip "Model not loaded" unless @llm
      formatted = @llm.apply_chat_template(messages_without_system)
      expect(formatted).to include("Hello")
      expect(formatted).to include("[INST]")
    end

    it "handles multi-turn conversations" do
      skip "Model not loaded" unless @llm
      formatted = @llm.apply_chat_template(multi_turn_messages)
      expect(formatted).to include("Hi")
      expect(formatted).to include("Hello!")
      expect(formatted).to include("How are you?")
    end

    it "generates coherent text via chat" do
      skip "Model not loaded" unless @llm
      config = Candle::GenerationConfig.deterministic(max_length: 30)
      result = @llm.chat(messages_with_system, config: config)

      # Should not contain template artifacts in the output
      expect(result).not_to include("<<SYS>>")
      expect(result).not_to include("[INST]")
      # Should produce actual text, not garbage
      expect(result.length).to be > 2
    end
  end
end
