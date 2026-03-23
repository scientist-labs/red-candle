# frozen_string_literal: true

require "spec_helper"

RSpec.describe Candle::Agent do
  let(:counter) { { calls: 0 } }

  let(:echo_tool) do
    Candle::Tool.new(
      name: "echo",
      description: "Echoes back the input",
      parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] }
    ) { |args| { echoed: args["text"] } }
  end

  describe "#initialize" do
    it "creates an agent with llm and tools" do
      mock_llm = double("llm")
      agent = Candle::Agent.new(mock_llm, tools: [echo_tool])

      expect(agent.llm).to eq(mock_llm)
      expect(agent.tools.length).to eq(1)
      expect(agent.max_iterations).to eq(Candle::Agent::MAX_ITERATIONS)
    end

    it "accepts custom max_iterations" do
      mock_llm = double("llm")
      agent = Candle::Agent.new(mock_llm, tools: [], max_iterations: 3)

      expect(agent.max_iterations).to eq(3)
    end

    it "accepts a system prompt" do
      mock_llm = double("llm")
      agent = Candle::Agent.new(mock_llm, tools: [], system_prompt: "Be helpful")

      expect(agent.system_prompt).to eq("Be helpful")
    end
  end

  describe "#run" do
    it "returns text response when no tool calls" do
      mock_llm = double("llm")
      allow(mock_llm).to receive(:chat_with_tools).and_return(
        Candle::ToolCallResult.new(
          tool_calls: [],
          tool_results: [],
          text_response: "Hello!",
          raw_response: "Hello!"
        )
      )

      agent = Candle::Agent.new(mock_llm, tools: [echo_tool])
      result = agent.run("Hi")

      expect(result.response).to eq("Hello!")
      expect(result.iterations).to eq(1)
      expect(result.tool_calls_made).to eq(0)
    end

    it "executes tool calls and continues" do
      mock_llm = double("llm")
      call_count = 0

      allow(mock_llm).to receive(:chat_with_tools) do |_messages, **_opts|
        call_count += 1
        if call_count == 1
          Candle::ToolCallResult.new(
            tool_calls: [Candle::ToolCall.new(name: "echo", arguments: { "text" => "test" })],
            tool_results: [{ tool_call: nil, result: { echoed: "test" }, error: nil }],
            text_response: nil,
            raw_response: "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"test\"}}</tool_call>"
          )
        else
          Candle::ToolCallResult.new(
            tool_calls: [],
            tool_results: [],
            text_response: "The echo returned: test",
            raw_response: "The echo returned: test"
          )
        end
      end

      agent = Candle::Agent.new(mock_llm, tools: [echo_tool])
      result = agent.run("Echo test")

      expect(result.response).to eq("The echo returned: test")
      expect(result.iterations).to eq(2)
      expect(result.tool_calls_made).to eq(1)
    end

    it "raises error when max iterations exceeded" do
      mock_llm = double("llm")
      allow(mock_llm).to receive(:chat_with_tools).and_return(
        Candle::ToolCallResult.new(
          tool_calls: [Candle::ToolCall.new(name: "echo", arguments: {})],
          tool_results: [{ tool_call: nil, result: "ok", error: nil }],
          text_response: nil,
          raw_response: "<tool_call>{\"name\":\"echo\",\"arguments\":{}}</tool_call>"
        )
      )

      agent = Candle::Agent.new(mock_llm, tools: [echo_tool], max_iterations: 2)

      expect { agent.run("Loop forever") }.to raise_error(Candle::AgentMaxIterationsError)
    end
  end
end
