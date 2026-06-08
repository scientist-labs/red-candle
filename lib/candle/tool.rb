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

  # A tool call requested by the model.
  #
  # name and arguments describe what the model wants to invoke. When chat is
  # called with execute: true, result holds the tool's return value (or error
  # holds the failure message) after Red Candle runs the tool.
  ToolCall = Struct.new(:name, :arguments, :result, :error, keyword_init: true) do
    # True once the tool has been run (successfully or not).
    def executed?
      !result.nil? || !error.nil?
    end

    # True when the tool ran without raising.
    def success?
      executed? && error.nil?
    end
  end
end
