# frozen_string_literal: true

require "json"

module Candle
  class Agent
    MAX_ITERATIONS = 10

    attr_reader :llm, :tools, :system_prompt, :max_iterations

    def initialize(llm, tools:, system_prompt: nil, max_iterations: MAX_ITERATIONS)
      @llm = llm
      @tools = tools
      @system_prompt = system_prompt
      @max_iterations = max_iterations
    end

    def run(user_message, **options)
      messages = []
      messages << { role: "system", content: @system_prompt } if @system_prompt
      messages << { role: "user", content: user_message }

      iterations = 0
      loop do
        iterations += 1
        if iterations > @max_iterations
          raise AgentMaxIterationsError,
            "Agent exceeded maximum iterations (#{@max_iterations})"
        end

        result = @llm.chat_with_tools(messages, tools: @tools, **options)

        if result.has_tool_calls?
          # If the model produced a substantial text answer alongside tool calls,
          # treat it as a final response (model is done, trailing tool calls are noise)
          if result.text_response && result.text_response.length > 50
            return AgentResult.new(
              response: result.text_response,
              messages: messages,
              iterations: iterations,
              tool_calls_made: messages.count { |m| m[:role] == "tool" }
            )
          end

          messages << { role: "assistant", content: result.raw_response }

          result.tool_results.each do |tr|
            tool_name = tr[:tool_call]&.name || "unknown"
            tool_output = tr[:error] ? "Error: #{tr[:error]}" : JSON.generate(tr[:result])
            messages << { role: "tool", content: "[#{tool_name}] #{tool_output}" }
          end
        else
          return AgentResult.new(
            response: result.text_response || result.raw_response,
            messages: messages,
            iterations: iterations,
            tool_calls_made: messages.count { |m| m[:role] == "tool" }
          )
        end
      end
    end
  end

  AgentResult = Struct.new(:response, :messages, :iterations, :tool_calls_made, keyword_init: true)
  AgentMaxIterationsError = Class.new(StandardError)
end
