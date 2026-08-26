require "spec_helper"
require "timeout"

RSpec.describe "Reranker concurrent calls through one instance" do
  # This is the shape Benchmate needs: one memoized singleton reranker, shared across
  # concurrent Puma threads. BERT-family rerankers have no interior mutability in their
  # forward pass (weights are immutable, candle's Tensor storage is Arc<RwLock<..>> by
  # design for concurrent reads), so a single instance should be safe to call from many
  # threads at once with no native locking, and results should be deterministic.
  let(:model_id) { "cross-encoder/ms-marco-MiniLM-L6-v2" }
  let(:pinned_revision) { "233902d25c440f23af6f7d6e94d2946bac0bee0a" }

  let(:reranker) do
    @reranker ||= Candle::Reranker.from_pretrained(
      model_id,
      device: Candle::Device.cpu,
      revision: pinned_revision,
      batch_size: 8
    )
  end

  let(:query) { "What are the health benefits of drinking water?" }
  let(:documents) do
    [
      "Drinking water helps regulate body temperature and supports metabolic function.",
      "The Eiffel Tower is located in Paris, France.",
      "Staying hydrated improves cognitive performance and energy levels throughout the day.",
      "Basketball is a popular sport played by two teams of five players.",
      "Water intake supports kidney function and helps flush out toxins from the body."
    ]
  end

  after(:all) do
    @reranker = nil
    GC.start
  end

  it "returns identical, deterministic results from every thread with no deadlock or corruption" do
    reference = reranker.rerank(query, documents)

    thread_count = 8
    calls_per_thread = 10

    results = Timeout.timeout(120) do
      threads = Array.new(thread_count) do
        Thread.new do
          Array.new(calls_per_thread) { reranker.rerank(query, documents) }
        end
      end
      threads.map(&:value)
    end

    expect(results.length).to eq(thread_count)

    results.flatten(1).each do |thread_result|
      expect(thread_result.map { |r| r[:doc_id] }).to eq(reference.map { |r| r[:doc_id] })
      thread_result.zip(reference).each do |actual, expected|
        expect(actual[:score]).to eq(expected[:score])
        expect(actual[:text]).to eq(expected[:text])
      end
    end
  end

  it "lets other Ruby threads make progress while many reranking calls run concurrently" do
    main_ticks = 0
    done = false

    worker_threads = Array.new(4) do
      Thread.new do
        5.times { reranker.rerank(query, documents) }
      end
    end

    monitor = Thread.new do
      worker_threads.each(&:join)
    ensure
      done = true
    end

    while !done
      main_ticks += 1
      sleep 0.001
    end
    monitor.join

    expect(main_ticks).to be > 2,
      "Expected the main thread to keep ticking while 4 threads reranked concurrently " \
      "(got #{main_ticks} ticks). GVL may not be released, or calls may be serializing."
  end
end
