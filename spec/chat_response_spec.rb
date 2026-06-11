# frozen_string_literal: true

require "spec_helper"

RSpec.describe Candle::ChatResponse do
  it "exposes content, thinking, tool_calls and raw_response" do
    tc = Candle::ToolCall.new(name: "x", arguments: {})
    response = described_class.new(
      content: "answer",
      thinking: "reasoning",
      tool_calls: [tc],
      raw_response: "<think>reasoning</think>answer"
    )

    expect(response.content).to eq("answer")
    expect(response.thinking).to eq("reasoning")
    expect(response.tool_calls).to eq([tc])
    expect(response.raw_response).to eq("<think>reasoning</think>answer")
  end

  it "defaults tool_calls to an empty array" do
    expect(described_class.new(content: "hi").tool_calls).to eq([])
    expect(described_class.new(content: "hi", tool_calls: nil).tool_calls).to eq([])
  end

  describe "#tool_calls?" do
    it "is false with no tool calls" do
      expect(described_class.new(content: "hi").tool_calls?).to be false
    end

    it "is true when tool calls are present" do
      response = described_class.new(content: nil, tool_calls: [Candle::ToolCall.new(name: "x", arguments: {})])
      expect(response.tool_calls?).to be true
    end
  end

  describe "#thinking?" do
    it "is false when thinking is nil or empty" do
      expect(described_class.new(content: "hi").thinking?).to be false
      expect(described_class.new(content: "hi", thinking: "").thinking?).to be false
    end

    it "is true when thinking is present" do
      expect(described_class.new(content: "hi", thinking: "hmm").thinking?).to be true
    end
  end

  describe "string coercion (backward compatibility)" do
    it "to_s returns the content" do
      expect(described_class.new(content: "the answer").to_s).to eq("the answer")
    end

    it "to_s returns empty string when content is nil" do
      expect(described_class.new(content: nil).to_s).to eq("")
    end

    it "behaves like a string via to_str" do
      response = described_class.new(content: "world")
      expect("hello " + response).to eq("hello world")
    end

    it "works with puts/print" do
      response = described_class.new(content: "printed")
      expect { print response }.to output("printed").to_stdout
    end
  end
end
