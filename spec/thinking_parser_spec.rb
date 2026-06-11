# frozen_string_literal: true

require "spec_helper"

RSpec.describe Candle::ThinkingParser do
  describe "#parse with default <think> tags" do
    subject(:parser) { described_class.new }

    it "splits thinking from content" do
      result = parser.parse("<think>let me reason</think>The answer is 42.")
      expect(result.thinking).to eq("let me reason")
      expect(result.content).to eq("The answer is 42.")
    end

    it "returns nil thinking when there is no thinking block" do
      result = parser.parse("Just a plain answer.")
      expect(result.thinking).to be_nil
      expect(result.content).to eq("Just a plain answer.")
    end

    it "handles multiline reasoning" do
      result = parser.parse("<think>line one\nline two</think>done")
      expect(result.thinking).to eq("line one\nline two")
      expect(result.content).to eq("done")
    end

    it "joins multiple thinking blocks" do
      result = parser.parse("<think>a</think>middle<think>b</think>end")
      expect(result.thinking).to eq("a\n\nb")
      expect(result.content).to eq("middleend")
    end

    it "returns nil content when only thinking is present" do
      result = parser.parse("<think>only reasoning</think>")
      expect(result.thinking).to eq("only reasoning")
      expect(result.content).to be_nil
    end

    it "handles nil input" do
      result = parser.parse(nil)
      expect(result.thinking).to be_nil
      expect(result.content).to be_nil
    end
  end

  describe "custom tags" do
    it "supports <thinking> style tags" do
      parser = described_class.new(open_tag: "<thinking>", close_tag: "</thinking>")
      result = parser.parse("<thinking>hmm</thinking>answer")
      expect(result.thinking).to eq("hmm")
      expect(result.content).to eq("answer")
    end
  end

  describe "explicit regex pattern" do
    it "uses the provided pattern" do
      parser = described_class.new(pattern: /\[reason\](.*?)\[\/reason\]/m)
      result = parser.parse("[reason]because[/reason]so")
      expect(result.thinking).to eq("because")
      expect(result.content).to eq("so")
    end
  end

  describe "disabled parser (nil tags)" do
    it "treats everything as content" do
      parser = described_class.new(open_tag: nil, close_tag: nil)
      result = parser.parse("<think>not stripped</think>text")
      expect(result.thinking).to be_nil
      expect(result.content).to eq("<think>not stripped</think>text")
    end
  end
end
