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

      remaining_text = text.gsub(TOOL_CALL_PATTERN, "").strip
      remaining_text = nil if remaining_text.empty?

      ParseResult.new(
        text_response: remaining_text,
        tool_calls: tool_calls
      )
    end
  end
end
