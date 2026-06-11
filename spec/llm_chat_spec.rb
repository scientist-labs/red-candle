# frozen_string_literal: true

require "spec_helper"

# Exercises the pure Ruby wiring behind LLM#chat (response assembly, tool
# execution, thinking resolution) without loading a model. The model-dependent
# behavior is covered by the spec/llm/* suites.
RSpec.describe "Candle::LLM chat wiring" do
  let(:echo) do
    Candle::Tool.new(
      name: "echo",
      description: "Echoes its input",
      parameters: { type: "object", properties: { text: { type: "string" } } }
    ) { |args| { echoed: args["text"] } }
  end

  describe "Candle::ChatResponse.from_raw without tools" do
    it "returns plain content for a plain response" do
      response = Candle::ChatResponse.from_raw("Hello there")
      expect(response).to be_a(Candle::ChatResponse)
      expect(response.content).to eq("Hello there")
      expect(response.tool_calls).to be_empty
      expect(response.thinking).to be_nil
    end

    it "extracts thinking when a parser is supplied" do
      parser = Candle::ThinkingParser.new
      response = Candle::ChatResponse.from_raw("<think>reasoning</think>final", thinking_parser: parser)
      expect(response.thinking).to eq("reasoning")
      expect(response.content).to eq("final")
      expect(response.raw_response).to eq("<think>reasoning</think>final")
    end

    it "leaves thinking tags in place when no parser is given" do
      response = Candle::ChatResponse.from_raw("<think>x</think>y")
      expect(response.thinking).to be_nil
      expect(response.content).to eq("<think>x</think>y")
    end
  end

  describe "Candle::ChatResponse.from_raw with tools" do
    it "parses tool calls without executing by default" do
      raw = '<tool_call>{"name":"echo","arguments":{"text":"hi"}}</tool_call>'
      response = Candle::ChatResponse.from_raw(raw, tools: [echo])
      expect(response.tool_calls?).to be true
      tc = response.tool_calls.first
      expect(tc.name).to eq("echo")
      expect(tc.arguments).to eq({ "text" => "hi" })
      expect(tc.executed?).to be false
    end

    it "executes tools when execute: true and attaches results" do
      raw = '<tool_call>{"name":"echo","arguments":{"text":"hi"}}</tool_call>'
      response = Candle::ChatResponse.from_raw(raw, tools: [echo], execute: true)
      tc = response.tool_calls.first
      expect(tc.result).to eq({ echoed: "hi" })
      expect(tc.error).to be_nil
      expect(tc.success?).to be true
    end

    it "filters out tool calls that are not available tools" do
      raw = '<tool_call>{"name":"missing","arguments":{}}</tool_call>'
      response = Candle::ChatResponse.from_raw(raw, tools: [echo], execute: true)
      expect(response.tool_calls).to be_empty
    end

    it "records an error when the tool raises" do
      boom = Candle::Tool.new(name: "boom", description: "always fails") { raise "kaboom" }
      raw = '<tool_call>{"name":"boom","arguments":{}}</tool_call>'
      response = Candle::ChatResponse.from_raw(raw, tools: [boom], execute: true)
      tc = response.tool_calls.first
      expect(tc.error).to eq("kaboom")
      expect(tc.success?).to be false
    end

    it "separates thinking, content, and tool calls together" do
      raw = "<think>plan</think>Calling now.<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"x\"}}</tool_call>"
      response = Candle::ChatResponse.from_raw(
        raw, thinking_parser: Candle::ThinkingParser.new, tools: [echo], execute: true
      )
      expect(response.thinking).to eq("plan")
      expect(response.content).to eq("Calling now.")
      expect(response.tool_calls.first.result).to eq({ echoed: "x" })
    end
  end

  describe "Candle::ThinkingParser.coerce" do
    it "returns nil for nil or false" do
      expect(Candle::ThinkingParser.coerce(nil)).to be_nil
      expect(Candle::ThinkingParser.coerce(false)).to be_nil
    end

    it "builds a parser from an [open, close] array" do
      parser = Candle::ThinkingParser.coerce(["<r>", "</r>"])
      expect(parser.open_tag).to eq("<r>")
      expect(parser.close_tag).to eq("</r>")
    end

    it "builds a parser from a Regexp" do
      parser = Candle::ThinkingParser.coerce(/\[t\](.*?)\[\/t\]/m)
      expect(parser.parse("[t]a[/t]b").thinking).to eq("a")
    end

    it "passes a ThinkingParser through unchanged" do
      original = Candle::ThinkingParser.new
      expect(Candle::ThinkingParser.coerce(original)).to be(original)
    end

    it "raises on an invalid spec" do
      expect { Candle::ThinkingParser.coerce(42) }.to raise_error(ArgumentError)
    end
  end

  describe "Candle::LLM.guess_thinking_tags" do
    it "infers <think> tags for Qwen3 models" do
      expect(Candle::LLM.guess_thinking_tags("Qwen/Qwen3-8B")).to eq(["<think>", "</think>"])
    end

    it "infers <think> tags for DeepSeek-R1 models" do
      expect(Candle::LLM.guess_thinking_tags("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"))
        .to eq(["<think>", "</think>"])
    end

    it "returns nil for models without thinking blocks" do
      expect(Candle::LLM.guess_thinking_tags("meta-llama/Llama-2-7b-chat-hf")).to be_nil
      expect(Candle::LLM.guess_thinking_tags(nil)).to be_nil
    end
  end
end
