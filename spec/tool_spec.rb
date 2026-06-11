# frozen_string_literal: true

require "spec_helper"

RSpec.describe Candle::Tool do
  let(:weather_tool) do
    Candle::Tool.new(
      name: "get_weather",
      description: "Get the current weather for a city",
      parameters: {
        type: "object",
        properties: { city: { type: "string", description: "City name" } },
        required: ["city"]
      }
    ) { |args| { temperature: 22, condition: "sunny", city: args["city"] } }
  end

  describe "#initialize" do
    it "creates a tool with name, description, and parameters" do
      expect(weather_tool.name).to eq("get_weather")
      expect(weather_tool.description).to eq("Get the current weather for a city")
      expect(weather_tool.parameters).to have_key(:type)
    end

    it "creates a tool without parameters" do
      tool = Candle::Tool.new(name: "noop", description: "Does nothing") { "done" }
      expect(tool.parameters).to eq({})
    end
  end

  describe "#call" do
    it "invokes the block with arguments" do
      result = weather_tool.call({ "city" => "Paris" })
      expect(result[:city]).to eq("Paris")
      expect(result[:temperature]).to eq(22)
    end

    it "handles errors in the callable" do
      tool = Candle::Tool.new(name: "fail", description: "Always fails") do
        raise "intentional error"
      end
      expect { tool.call({}) }.to raise_error(RuntimeError, "intentional error")
    end
  end

  describe "#to_tool_definition" do
    it "returns OpenAI/HuggingFace format" do
      defn = weather_tool.to_tool_definition
      expect(defn["type"]).to eq("function")
      expect(defn["function"]["name"]).to eq("get_weather")
      expect(defn["function"]["description"]).to include("weather")
      expect(defn["function"]["parameters"]).to have_key(:type)
    end
  end
end

RSpec.describe Candle::ToolCall do
  it "creates a tool call with name and arguments" do
    tc = Candle::ToolCall.new(name: "get_weather", arguments: { "city" => "Paris" })
    expect(tc.name).to eq("get_weather")
    expect(tc.arguments).to eq({ "city" => "Paris" })
  end

  it "is not executed before a result or error is attached" do
    tc = Candle::ToolCall.new(name: "x", arguments: {})
    expect(tc.executed?).to be false
    expect(tc.success?).to be false
  end

  it "reports success once a result is attached" do
    tc = Candle::ToolCall.new(name: "x", arguments: {}, result: { ok: true })
    expect(tc.executed?).to be true
    expect(tc.success?).to be true
  end

  it "reports failure once an error is attached" do
    tc = Candle::ToolCall.new(name: "x", arguments: {}, error: "boom")
    expect(tc.executed?).to be true
    expect(tc.success?).to be false
  end
end
