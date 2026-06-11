# frozen_string_literal: true

require "json"

module Candle
  # A single typed event emitted while streaming a chat response.
  #
  # type is one of:
  #   :content    - a chunk of answer text (delta is the text)
  #   :thinking   - a chunk of reasoning text (delta is the text)
  #   :tool_call  - the model emitted a complete tool call (tool_call is set)
  #   :done       - the stream finished (delta and tool_call are nil)
  StreamEvent = Struct.new(:type, :delta, :tool_call, keyword_init: true) do
    def content?
      type == :content
    end

    def thinking?
      type == :thinking
    end

    def tool_call?
      type == :tool_call
    end

    def done?
      type == :done
    end
  end

  # Turns a raw token stream into typed StreamEvents.
  #
  # Tokens arrive one at a time and tag boundaries (<think>, <tool_call>, ...)
  # can fall in the middle of a token, so this maintains a small buffer and a
  # mode (:content / :thinking / :tool_call), only emitting text once it is
  # certain that text is not the prefix of an upcoming marker.
  class StreamProcessor
    TOOL_OPEN_TAG = "<tool_call>"
    TOOL_CLOSE_TAG = "</tool_call>"

    def initialize(thinking_parser: nil, tools: nil, &emit)
      @think_open = thinking_parser&.open_tag
      @think_close = thinking_parser&.close_tag
      @tools = tools
      @emit = emit
      @buffer = +""
      @mode = :content
    end

    # Feed the next raw token from the underlying generator.
    def process(token)
      return if token.nil? || token.empty?
      @buffer << token
      pump
    end

    # Flush any buffered text and emit the terminal :done event.
    def finish
      unless @buffer.empty?
        case @mode
        when :thinking
          emit(:thinking, @buffer)
        when :tool_call
          # Unterminated tool call — surface the raw text rather than drop it.
          emit(:content, "#{TOOL_OPEN_TAG}#{@buffer}")
        else
          emit(:content, @buffer)
        end
        @buffer = +""
      end
      @emit.call(StreamEvent.new(type: :done, delta: nil, tool_call: nil))
    end

    private

    def pump
      loop do
        case @mode
        when :content
          break unless handle_content
        when :thinking
          break unless handle_thinking
        when :tool_call
          break unless handle_tool_call
        end
      end
    end

    # Returns true if it made progress (loop again), false to wait for more input.
    def handle_content
      markers = [@think_open, TOOL_OPEN_TAG].compact
      idx, marker = first_marker(markers)

      if idx.nil?
        emit_safe_text(:content, markers)
        return false
      end

      emit(:content, @buffer[0...idx]) if idx > 0
      @buffer = @buffer[(idx + marker.length)..] || +""
      @mode = (marker == @think_open ? :thinking : :tool_call)
      true
    end

    def handle_thinking
      idx = @buffer.index(@think_close)
      if idx.nil?
        emit_safe_text(:thinking, [@think_close])
        return false
      end

      emit(:thinking, @buffer[0...idx]) if idx > 0
      @buffer = @buffer[(idx + @think_close.length)..] || +""
      @mode = :content
      true
    end

    def handle_tool_call
      idx = @buffer.index(TOOL_CLOSE_TAG)
      return false if idx.nil? # accumulate until the closing tag arrives

      json_str = @buffer[0...idx]
      @buffer = @buffer[(idx + TOOL_CLOSE_TAG.length)..] || +""
      emit_tool_call(json_str)
      @mode = :content
      true
    end

    # Find the earliest occurrence of any marker in the buffer.
    def first_marker(markers)
      best_idx = nil
      best_marker = nil
      markers.each do |marker|
        idx = @buffer.index(marker)
        next if idx.nil?
        if best_idx.nil? || idx < best_idx
          best_idx = idx
          best_marker = marker
        end
      end
      [best_idx, best_marker]
    end

    # Emit buffered text of the given type, holding back any trailing characters
    # that could be the start of one of the markers.
    def emit_safe_text(type, markers)
      hold = trailing_partial_marker_length(markers)
      return if hold >= @buffer.length
      emit(type, @buffer[0...(@buffer.length - hold)])
      @buffer = @buffer[(@buffer.length - hold)..] || +""
    end

    # Length of the longest buffer suffix that is a (strict) prefix of a marker.
    def trailing_partial_marker_length(markers)
      max = markers.map(&:length).max.to_i - 1
      max = @buffer.length if max > @buffer.length
      max.downto(1) do |n|
        suffix = @buffer[-n..]
        return n if markers.any? { |m| m.start_with?(suffix) }
      end
      0
    end

    def emit(type, text)
      return if text.nil? || text.empty?
      @emit.call(StreamEvent.new(type: type, delta: text, tool_call: nil))
    end

    def emit_tool_call(json_str)
      parsed = begin
        JSON.parse(json_str.strip)
      rescue JSON::ParserError
        nil
      end
      return if parsed.nil?

      name = parsed["name"]
      return if name.nil?
      return if @tools && !@tools.empty? && @tools.none? { |t| t.name == name }

      arguments = parsed["arguments"] || parsed["parameters"] || {}
      tool_call = ToolCall.new(name: name, arguments: arguments)
      @emit.call(StreamEvent.new(type: :tool_call, delta: nil, tool_call: tool_call))
    end
  end
end
