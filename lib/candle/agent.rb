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

        result = @llm.chat(messages, tools: @tools, execute: true, **options)

        if result.tool_calls?
          # If the model produced a substantial text answer alongside tool calls,
          # treat it as a final response (model is done, trailing tool calls are noise).
          # content already has thinking blocks stripped by ChatResponse.
          final_text = result.content&.strip
          if final_text && final_text.length > 50
            return AgentResult.new(
              response: result.content,
              messages: messages,
              iterations: iterations,
              tool_calls_made: messages.count { |m| m[:role] == "tool" }
            )
          end

          messages << { role: "assistant", content: result.raw_response }

          result.tool_calls.each do |tool_call|
            tool_output = tool_call.error ? "Error: #{tool_call.error}" : JSON.generate(tool_call.result)
            messages << { role: "tool", content: "[#{tool_call.name}] #{tool_output}" }
          end
        else
          return AgentResult.new(
            response: result.content || result.raw_response,
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
