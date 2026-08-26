require "spec_helper"

RSpec.describe "Reranker revision pinning" do
  # Benchmate's target model + immutable revision.
  let(:model_id) { "cross-encoder/ms-marco-MiniLM-L6-v2" }
  let(:pinned_revision) { "233902d25c440f23af6f7d6e94d2946bac0bee0a" }

  describe "loading a pinned revision" do
    let(:reranker) do
      @reranker ||= Candle::Reranker.from_pretrained(
        model_id,
        device: Candle::Device.cpu,
        revision: pinned_revision
      )
    end

    after(:all) do
      @reranker = nil
      GC.start
    end

    it "loads successfully on CPU" do
      expect(reranker).to be_a(Candle::Reranker)
    end

    it "surfaces the pinned revision in options" do
      expect(reranker.options["revision"]).to eq(pinned_revision)
    end

    it "surfaces the pinned revision in #inspect" do
      expect(reranker.inspect).to include(pinned_revision)
    end

    it "still reports the correct model_id" do
      expect(reranker.model_id).to eq(model_id)
    end

    it "ranks a relevant passage above an irrelevant one (known-query smoke test)" do
      query = "How many people live in London?"
      documents = [
        "The weather in London is often rainy.",
        "Around 9 million people live in London."
      ]

      results = reranker.rerank(query, documents)

      expect(results.first[:text]).to eq("Around 9 million people live in London.")
      expect(results.first[:score]).to be > results.last[:score]
    end
  end

  describe "omitting revision" do
    it "preserves prior behavior by tracking the default branch" do
      reranker = Candle::Reranker.from_pretrained(model_id, device: Candle::Device.cpu)
      expect(reranker.options["revision"]).to be_nil
      expect(reranker.inspect).to include("revision=main")
    end
  end

  describe "an invalid revision" do
    it "raises an error naming the model and the requested revision, without falling back to main" do
      expect {
        Candle::Reranker.from_pretrained(
          model_id,
          device: Candle::Device.cpu,
          revision: "not-a-real-revision-abc123"
        )
      }.to raise_error(RuntimeError) { |error|
        expect(error.message).to include(model_id)
        expect(error.message).to include("not-a-real-revision-abc123")
      }
    end
  end
end
