# frozen_string_literal: true

require "json"

module Candle
  class ToolCallParser
    DEFAULT_PATTERN = /<tool_call>\s*(.*?)\s*<\/tool_call>/m

    attr_reader :pattern

    def initialize(pattern: DEFAULT_PATTERN)
      @pattern = pattern
    end

    ParseResult = Struct.new(:text_response, :tool_calls, keyword_init: true) do
      def has_tool_calls?
        tool_calls && !tool_calls.empty?
      end
    end

    def parse(text, available_tools: [])
      tool_calls = []

      text.scan(@pattern) do |match|
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

      remaining_text = text.gsub(@pattern, "").strip
      remaining_text = nil if remaining_text.empty?

      ParseResult.new(
        text_response: remaining_text,
        tool_calls: tool_calls
      )
    end

    # Convenience class method using the default pattern
    def self.parse(text, available_tools: [])
      new.parse(text, available_tools: available_tools)
    end
  end
end
