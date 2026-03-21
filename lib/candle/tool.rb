# frozen_string_literal: true

module Candle
  class Tool
    attr_reader :name, :description, :parameters

    def initialize(name:, description:, parameters: {}, &block)
      @name = name
      @description = description
      @parameters = parameters
      @callable = block
    end

    def call(arguments)
      @callable.call(arguments)
    end

    def to_tool_definition
      {
        "type" => "function",
        "function" => {
          "name" => @name,
          "description" => @description,
          "parameters" => @parameters
        }
      }
    end
  end

  ToolCall = Struct.new(:name, :arguments, keyword_init: true)

  ToolCallResult = Struct.new(
    :tool_calls,
    :tool_results,
    :text_response,
    :raw_response,
    keyword_init: true
  ) do
    def has_tool_calls?
      tool_calls && !tool_calls.empty?
    end

    def success?
      tool_results.all? { |r| r[:error].nil? }
    end
  end
end
