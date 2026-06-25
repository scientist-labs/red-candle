# frozen_string_literal: true

module Candle
  # Unified return type for LLM#chat and LLM#chat with tools.
  #
  # Regardless of whether the model produced plain text, reasoning, or tool
  # calls, chat always returns a ChatResponse with the same shape. This mirrors
  # frontier APIs (OpenAI, Anthropic) which return a single message object.
  #
  #   response = llm.chat(messages)
  #   response.content      # => "Paris is the capital of France."
  #   response.thinking     # => "The user is asking about..." (or nil)
  #   response.tool_calls   # => [] (or Array<ToolCall>)
  #   puts response         # => content (to_s/to_str return content)
  class ChatResponse
    # Text the model produced, with any thinking and tool-call markup removed.
    attr_reader :content
    # Reasoning extracted from a thinking block (e.g. <think>...</think>), or nil.
    attr_reader :thinking
    # Array of ToolCall the model requested (empty when none).
    attr_reader :tool_calls
    # The complete, unmodified text the model generated.
    attr_reader :raw_response

    def initialize(content:, thinking: nil, tool_calls: [], raw_response: nil)
      @content = content
      @thinking = thinking
      @tool_calls = tool_calls || []
      @raw_response = raw_response
    end

    # Assemble a ChatResponse from raw model output. Extracts thinking (when a
    # parser is given), parses any tool calls (when tools are given), and
    # optionally executes them, attaching result/error to each ToolCall.
    def self.from_raw(raw_response, thinking_parser: nil, tools: nil, execute: false)
      if thinking_parser
        parsed = thinking_parser.parse(raw_response)
        thinking_text = parsed.thinking
        body = parsed.content
      else
        thinking_text = nil
        body = raw_response
      end

      if tools && !tools.empty?
        result = ToolCallParser.new.parse(body.to_s, available_tools: tools)
        tool_calls = result.tool_calls
        content = result.text_response
        tool_calls.each { |tool_call| execute_tool_call(tool_call, tools) } if execute
      else
        tool_calls = []
        content = body
      end

      new(content: content, thinking: thinking_text, tool_calls: tool_calls, raw_response: raw_response)
    end

    # Run a single tool call, recording its result or error onto the ToolCall.
    def self.execute_tool_call(tool_call, tools)
      tool = tools.find { |t| t.name == tool_call.name }
      if tool.nil?
        tool_call.error = "Unknown tool: #{tool_call.name}"
        return
      end

      begin
        tool_call.result = tool.call(tool_call.arguments)
      rescue Exception => e
        tool_call.error = e.message
      end
    end

    # True when the model requested at least one tool call.
    def tool_calls?
      !@tool_calls.empty?
    end

    # True when reasoning was extracted from a thinking block.
    def thinking?
      !@thinking.nil? && !@thinking.empty?
    end

    # Backward compatibility: `puts llm.chat(messages)` and string coercion
    # behave as they did when chat returned a String.
    def to_s
      @content.to_s
    end
    alias to_str to_s

    def inspect
      parts = ["content=#{@content.inspect}"]
      parts << "thinking=#{truncate(@thinking).inspect}" if thinking?
      parts << "tool_calls=#{@tool_calls.size}" if tool_calls?
      "#<Candle::ChatResponse #{parts.join(" ")}>"
    end

    private

    def truncate(text, max = 60)
      return text if text.nil? || text.length <= max
      "#{text[0, max]}..."
    end
  end
end
