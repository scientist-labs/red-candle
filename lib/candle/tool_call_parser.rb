# frozen_string_literal: true

require "json"

module Candle
  class ToolCallParser
    TOOL_CALL_PATTERN = /<tool_call>\s*(.*?)\s*<\/tool_call>/m

    ParseResult = Struct.new(:text_response, :tool_calls, keyword_init: true) do
      def has_tool_calls?
        tool_calls && !tool_calls.empty?
      end
    end

    def self.parse(text, available_tools: [])
      tool_calls = []

      text.scan(TOOL_CALL_PATTERN) do |match|
        json_str = match[0].strip
        begin
          parsed = JSON.parse(json_str)
          name = parsed["name"]
          arguments = parsed["arguments"] || parsed["parameters"] || {}

          next unless name
          if available_tools.empty? || available_tools.any? { |t| t.name == name }
            tool_calls << ToolCall.new(name: name, arguments: arguments)
          end
        rescue JSON::ParserError
          # Skip malformed tool calls
        end
      end

      # Deduplicate identical tool calls (models sometimes repeat the same call)
      tool_calls.uniq! { |tc| [tc.name, tc.arguments] }

      remaining_text = text.gsub(TOOL_CALL_PATTERN, "").strip
      # Strip <think> blocks from the text response
      remaining_text = remaining_text.gsub(/<think>.*?<\/think>/m, "").strip
      remaining_text = nil if remaining_text.empty?

      ParseResult.new(
        text_response: remaining_text,
        tool_calls: tool_calls
      )
    end
  end
end
