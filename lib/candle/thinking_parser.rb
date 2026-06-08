# frozen_string_literal: true

module Candle
  # Extracts a model's "thinking" / reasoning span from generated text.
  #
  # Reasoning models wrap their chain-of-thought in a tagged block — most use
  # <think>...</think>, some use <thinking>...</thinking>, and many models emit
  # none at all. The parser is configured with an open/close tag pair (which
  # also drives streaming segmentation) and can additionally take an explicit
  # regex for batch parsing.
  #
  #   parser = Candle::ThinkingParser.new
  #   result = parser.parse("<think>hmm</think>The answer is 42.")
  #   result.thinking  # => "hmm"
  #   result.content   # => "The answer is 42."
  class ThinkingParser
    DEFAULT_OPEN_TAG = "<think>"
    DEFAULT_CLOSE_TAG = "</think>"

    Result = Struct.new(:thinking, :content, keyword_init: true)

    # Coerce a thinking spec into a parser (or nil to disable extraction).
    # Accepts a ThinkingParser, an [open_tag, close_tag] Array, a Regexp, or
    # nil/false. Does not handle the "use the model default" case — callers
    # resolve that before coercing.
    def self.coerce(spec)
      case spec
      when nil, false then nil
      when ThinkingParser then spec
      when Array then new(open_tag: spec[0], close_tag: spec[1])
      when Regexp then new(pattern: spec)
      else
        raise ArgumentError,
          "thinking must be a ThinkingParser, [open_tag, close_tag], Regexp, or false"
      end
    end

    attr_reader :open_tag, :close_tag, :pattern

    def initialize(open_tag: DEFAULT_OPEN_TAG, close_tag: DEFAULT_CLOSE_TAG, pattern: nil)
      @open_tag = open_tag
      @close_tag = close_tag
      @pattern = pattern || build_pattern(open_tag, close_tag)
    end

    # Split text into its extracted thinking and the remaining content.
    # Returns a Result; thinking is nil when no thinking block is present.
    def parse(text)
      return Result.new(thinking: nil, content: text) if text.nil? || @pattern.nil?

      thoughts = []
      text.scan(@pattern) do |match|
        captured = match.is_a?(Array) ? match[0] : match
        thoughts << captured.strip if captured
      end

      content = text.gsub(@pattern, "").strip
      content = nil if content.empty?

      thinking = thoughts.empty? ? nil : thoughts.join("\n\n")
      thinking = nil if thinking && thinking.empty?

      Result.new(thinking: thinking, content: content)
    end

    private

    def build_pattern(open_tag, close_tag)
      return nil if open_tag.nil? || close_tag.nil?
      /#{Regexp.escape(open_tag)}(.*?)#{Regexp.escape(close_tag)}/m
    end
  end
end
