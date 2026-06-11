# frozen_string_literal: true

require "spec_helper"

RSpec.describe Candle::StreamProcessor do
  let(:thinking_parser) { Candle::ThinkingParser.new }

  # Feed a sequence of tokens through a processor and collect emitted events.
  def run(tokens, tools: nil, parser: thinking_parser)
    events = []
    processor = described_class.new(thinking_parser: parser, tools: tools) { |e| events << e }
    tokens.each { |t| processor.process(t) }
    processor.finish
    events
  end

  def text_of(events, type)
    events.select { |e| e.type == type }.map(&:delta).join
  end

  describe "plain content" do
    it "emits content events and a terminal done event" do
      events = run(["Hello", ", ", "world"])
      expect(text_of(events, :content)).to eq("Hello, world")
      expect(events.last).to have_attributes(type: :done)
    end
  end

  describe "thinking extraction" do
    it "separates thinking from content when tags land on token boundaries" do
      events = run(["<think>", "reasoning", "</think>", "answer"])
      expect(text_of(events, :thinking)).to eq("reasoning")
      expect(text_of(events, :content)).to eq("answer")
    end

    it "handles tags split across multiple tokens" do
      events = run(["<th", "ink>re", "ason</thi", "nk>ans", "wer"])
      expect(text_of(events, :thinking)).to eq("reason")
      expect(text_of(events, :content)).to eq("answer")
    end

    it "handles thinking and content in a single token" do
      events = run(["<think>quick</think>done"])
      expect(text_of(events, :thinking)).to eq("quick")
      expect(text_of(events, :content)).to eq("done")
    end

    it "does not emit a content marker prefix prematurely" do
      # The leading '<' must be held back until we know it isn't '<think>'.
      events = run(["a", "<", "think>x</think>", "b"])
      expect(text_of(events, :content)).to eq("ab")
      expect(text_of(events, :thinking)).to eq("x")
    end

    it "treats a lone angle bracket as content when no tag follows" do
      events = run(["1 < 2 is true"])
      expect(text_of(events, :content)).to eq("1 < 2 is true")
      expect(text_of(events, :thinking)).to eq("")
    end
  end

  describe "tool calls" do
    let(:echo) { Candle::Tool.new(name: "echo", description: "e", parameters: {}) { |a| a } }

    it "emits a tool_call event with a parsed ToolCall" do
      tokens = ['<tool_call>{"name":"echo",', '"arguments":{"text":"hi"}}', "</tool_call>"]
      events = run(tokens, tools: [echo])
      tool_events = events.select(&:tool_call?)
      expect(tool_events.size).to eq(1)
      expect(tool_events.first.tool_call.name).to eq("echo")
      expect(tool_events.first.tool_call.arguments).to eq({ "text" => "hi" })
    end

    it "emits surrounding content alongside the tool call" do
      tokens = ["Let me check. ", '<tool_call>{"name":"echo","arguments":{}}</tool_call>', " Done."]
      events = run(tokens, tools: [echo])
      expect(text_of(events, :content)).to eq("Let me check.  Done.")
      expect(events.count(&:tool_call?)).to eq(1)
    end

    it "ignores tool calls not in the available tools" do
      tokens = ['<tool_call>{"name":"unknown","arguments":{}}</tool_call>']
      events = run(tokens, tools: [echo])
      expect(events.count(&:tool_call?)).to eq(0)
    end

    it "surfaces an unterminated tool call as content" do
      events = run(['<tool_call>{"name":"echo"'], tools: [echo])
      expect(text_of(events, :content)).to include("<tool_call>")
    end
  end

  describe "without a thinking parser" do
    it "treats think tags as ordinary content" do
      events = run(["<think>x</think>y"], parser: nil)
      expect(text_of(events, :content)).to eq("<think>x</think>y")
      expect(text_of(events, :thinking)).to eq("")
    end
  end

  it "always ends with exactly one done event" do
    events = run(["<think>a</think>", "b", '<tool_call>{"name":"echo","arguments":{}}</tool_call>'],
                 tools: [Candle::Tool.new(name: "echo", description: "e") { {} }])
    expect(events.count(&:done?)).to eq(1)
    expect(events.last).to be_done
  end
end
