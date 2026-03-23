# frozen_string_literal: true

require "spec_helper"

RSpec.describe Candle::ToolCallParser do
  describe ".parse" do
    it "parses a single tool call" do
      output = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
      result = Candle::ToolCallParser.parse(output)

      expect(result.has_tool_calls?).to be true
      expect(result.tool_calls.length).to eq(1)
      expect(result.tool_calls.first.name).to eq("get_weather")
      expect(result.tool_calls.first.arguments).to eq({ "city" => "Paris" })
    end

    it "parses multiple tool calls" do
      output = "<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n</tool_call>\n" \
               "<tool_call>\n{\"name\": \"b\", \"arguments\": {\"y\": 2}}\n</tool_call>"
      result = Candle::ToolCallParser.parse(output)

      expect(result.tool_calls.length).to eq(2)
      expect(result.tool_calls.map(&:name)).to eq(["a", "b"])
    end

    it "returns text response when no tool calls present" do
      result = Candle::ToolCallParser.parse("The weather is sunny today.")

      expect(result.has_tool_calls?).to be false
      expect(result.text_response).to eq("The weather is sunny today.")
    end

    it "extracts text around tool calls" do
      output = "Let me check that.\n<tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</tool_call>\nDone."
      result = Candle::ToolCallParser.parse(output)

      expect(result.has_tool_calls?).to be true
      expect(result.text_response).to include("Let me check that.")
      expect(result.text_response).to include("Done.")
      expect(result.text_response).not_to include("tool_call")
    end

    it "handles malformed JSON gracefully" do
      output = "<tool_call>\n{broken json here\n</tool_call>"
      result = Candle::ToolCallParser.parse(output)

      expect(result.has_tool_calls?).to be false
    end

    it "handles tool call with parameters key instead of arguments" do
      output = "<tool_call>\n{\"name\": \"test\", \"parameters\": {\"a\": 1}}\n</tool_call>"
      result = Candle::ToolCallParser.parse(output)

      expect(result.has_tool_calls?).to be true
      expect(result.tool_calls.first.arguments).to eq({ "a" => 1 })
    end

    it "skips tool calls without a name" do
      output = "<tool_call>\n{\"arguments\": {\"a\": 1}}\n</tool_call>"
      result = Candle::ToolCallParser.parse(output)

      expect(result.has_tool_calls?).to be false
    end

    it "filters by available tools when specified" do
      tools = [
        Candle::Tool.new(name: "allowed", description: "ok") { "ok" }
      ]
      output = "<tool_call>\n{\"name\": \"allowed\", \"arguments\": {}}\n</tool_call>\n" \
               "<tool_call>\n{\"name\": \"forbidden\", \"arguments\": {}}\n</tool_call>"
      result = Candle::ToolCallParser.parse(output, available_tools: tools)

      expect(result.tool_calls.length).to eq(1)
      expect(result.tool_calls.first.name).to eq("allowed")
    end

    it "returns nil text_response when only tool calls are present" do
      output = "<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>"
      result = Candle::ToolCallParser.parse(output)

      expect(result.text_response).to be_nil
    end

    it "handles whitespace variations in tool call tags" do
      output = "<tool_call>  {\"name\": \"x\", \"arguments\": {}}  </tool_call>"
      result = Candle::ToolCallParser.parse(output)

      expect(result.has_tool_calls?).to be true
      expect(result.tool_calls.first.name).to eq("x")
    end
  end
end
