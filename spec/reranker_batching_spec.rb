require "spec_helper"

RSpec.describe "Reranker bounded micro-batching" do
  let(:model_id) { "cross-encoder/ms-marco-MiniLM-L6-v2" }
  let(:pinned_revision) { "233902d25c440f23af6f7d6e94d2946bac0bee0a" }
  let(:query) { "What are the benefits of exercise?" }

  # 100 documents of deliberately varying length (some short, some long) so a
  # micro-batch mixes sequence lengths — the exact scenario the right-padding fix targets.
  let(:documents) do
    topics = [
      "Exercise improves cardiovascular health and reduces the risk of heart disease.",
      "Cats sleep for most of the day.",
      "Regular physical activity strengthens muscles and bones over time.",
      "The stock market fluctuates based on many economic factors.",
      "Exercise releases endorphins which can improve mood and reduce stress significantly over the following hours.",
      "Rain is expected later this week across the region."
    ]
    (0...100).map { |i| "#{topics[i % topics.length]} (document #{i})" }
  end

  def batch_size_reranker(batch_size)
    Candle::Reranker.from_pretrained(
      model_id,
      device: Candle::Device.cpu,
      revision: pinned_revision,
      batch_size: batch_size
    )
  end

  describe "batch_size validation" do
    it "rejects a zero batch_size" do
      expect {
        batch_size_reranker(0)
      }.to raise_error(ArgumentError, /batch_size/)
    end

    it "rejects a negative batch_size" do
      expect {
        batch_size_reranker(-1)
      }.to raise_error(ArgumentError, /batch_size/)
    end

    it "accepts the default batch_size of 16" do
      expect(Candle::Reranker::DEFAULT_BATCH_SIZE).to eq(16)
      reranker = batch_size_reranker(16)
      expect(reranker.options["batch_size"]).to eq(16)
    end
  end

  describe "batch_size does not change reranking results" do
    let(:small_batch_reranker) { @small ||= batch_size_reranker(8) }
    let(:large_batch_reranker) { @large ||= batch_size_reranker(100) }

    after(:all) do
      @small = nil
      @large = nil
      GC.start
    end

    it "produces the same ranking (order + doc_id) regardless of batch_size" do
      small_results = small_batch_reranker.rerank(query, documents)
      large_results = large_batch_reranker.rerank(query, documents)

      expect(small_results.map { |r| r[:doc_id] }).to eq(large_results.map { |r| r[:doc_id] })
    end

    it "produces near-identical scores regardless of batch_size" do
      small_results = small_batch_reranker.rerank(query, documents).sort_by { |r| r[:doc_id] }
      large_results = large_batch_reranker.rerank(query, documents).sort_by { |r| r[:doc_id] }

      small_results.zip(large_results).each do |small, large|
        expect(small[:score]).to be_within(1e-4).of(large[:score])
      end
    end

    it "preserves each document's original global doc_id" do
      results = small_batch_reranker.rerank(query, documents)
      results.each do |result|
        expect(documents[result[:doc_id]]).to include("(document #{result[:doc_id]})")
        expect(result[:text]).to eq(documents[result[:doc_id]])
      end
    end
  end

  describe "global sort with stable tie-break on original index" do
    it "orders exact-tie scores by ascending original index" do
      # All three docs identical and kept within a single micro-batch (batch_size 8 >
      # 3 docs), so they get byte-identical tensors and an exact-equal score — no
      # cross-chunk floating point drift to muddy the tie-break assertion.
      reranker = batch_size_reranker(8)
      tied_docs = ["identical document text", "identical document text", "identical document text"]

      results = reranker.rerank("some query", tied_docs)

      expect(results.map { |r| r[:score] }.uniq.length).to eq(1)
      expect(results.map { |r| r[:doc_id] }).to eq([0, 1, 2])
    end
  end

  describe "raw logits vs sigmoid, preserved under batching" do
    it "keeps raw-logit and sigmoid modes both working with a small batch_size" do
      reranker = batch_size_reranker(4)

      raw = reranker.rerank(query, documents.first(10), apply_sigmoid: false)
      sigmoid = reranker.rerank(query, documents.first(10), apply_sigmoid: true)

      # Sigmoid squashes every score into (0, 1); raw logits are unbounded and should
      # disagree with the sigmoid-activated score for the same ranking.
      expect(sigmoid.map { |r| r[:score] }).to all(be_between(0.0, 1.0))
      raw_by_id = raw.to_h { |r| [r[:doc_id], r[:score]] }
      sigmoid_by_id = sigmoid.to_h { |r| [r[:doc_id], r[:score]] }
      raw_by_id.each do |doc_id, raw_score|
        expect(sigmoid_by_id[doc_id]).to be_within(1e-6).of(1.0 / (1.0 + Math.exp(-raw_score)))
      end
    end
  end
end
