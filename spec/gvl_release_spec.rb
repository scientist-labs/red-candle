# frozen_string_literal: true

require "spec_helper"

RSpec.describe "GVL release during inference" do
  let(:llm) do
    @llm ||= Candle::LLM.from_pretrained(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
      device: "cpu"
    )
  end

  let(:config) { Candle::GenerationConfig.deterministic(max_length: 30) }

  after(:all) do
    @llm = nil
    GC.start
  end

  it "releases the GVL during LLM.generate so other threads can run" do
    main_ticks = 0
    done = false

    inference_thread = Thread.new do
      llm.generate("What is the meaning of life?", config: config)
    ensure
      done = true
    end

    # Main thread: count ticks while inference runs
    while !done
      main_ticks += 1
      sleep 0.005 # 5ms ticks
    end

    inference_thread.join

    # If the GVL is released, the main thread should get many ticks
    # during inference. If held, it gets 0-1 ticks.
    expect(main_ticks).to be > 5,
      "Expected main thread to run during inference (got #{main_ticks} ticks). " \
      "GVL may not be released."
  end

  it "still returns correct generation results" do
    result = Thread.new { llm.generate("Hello", config: config) }.value
    expect(result).to be_a(String)
    expect(result.length).to be > 0
  end
end

RSpec.describe "GVL release during reranking" do
  let(:reranker) do
    @reranker ||= Candle::Reranker.from_pretrained(
      "cross-encoder/ms-marco-MiniLM-L-12-v2",
      device: "cpu"
    )
  end

  let(:docs) do
    # Use enough documents to make reranking take measurable time
    base = ["Ruby is a programming language", "Python is a snake", "Java is an island"]
    base * 10 # 30 documents
  end

  after(:all) do
    @reranker = nil
    GC.start
  end

  it "releases the GVL during reranking so other threads can run" do
    main_ticks = 0
    done = false

    rerank_thread = Thread.new do
      reranker.rerank("What is Ruby?", docs)
    ensure
      done = true
    end

    while !done
      main_ticks += 1
      sleep 0.001 # 1ms ticks
    end

    rerank_thread.join

    # Reranking 30 docs should take enough time for main thread to tick
    expect(main_ticks).to be > 2,
      "Expected main thread to run during reranking (got #{main_ticks} ticks). " \
      "GVL may not be released."
  end

  it "still returns correct reranking results from a thread" do
    results = Thread.new {
      reranker.rerank("What is Ruby?", ["Ruby is a language", "Cats are cute"])
    }.value

    expect(results.length).to eq(2)
    expect(results[0][:text]).to include("Ruby")
  end
end
