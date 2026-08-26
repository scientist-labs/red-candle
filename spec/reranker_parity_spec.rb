require "spec_helper"
require "json"

RSpec.describe "Reranker parity with Python Sentence Transformers" do
  # See docs/RERANKER_PARITY.md for how this fixture was generated and what it proves.
  fixture_path = File.join(__dir__, "fixtures", "reranker_parity", "ms_marco_minilm_l6.json")
  fixture = JSON.parse(File.read(fixture_path))

  # Measured max deviation on this machine (macOS arm64, Accelerate backend) across the
  # fixture's 7 pairs is ~8e-6 on raw logits and ~0 (rounds away at 1e-6) on sigmoid scores
  # — see docs/RERANKER_PARITY.md. These tolerances are ~100x that measured ceiling, to
  # absorb backend/platform floating-point differences (candle's CPU gemm vs PyTorch's,
  # different BLAS implementations across CI/production CPUs) without masking a real
  # architecture or pooling regression.
  RAW_LOGIT_TOLERANCE = 1e-3
  SIGMOID_TOLERANCE = 1e-4

  let(:reranker) do
    @reranker ||= Candle::Reranker.from_pretrained(
      fixture["model_id"],
      device: Candle::Device.cpu,
      revision: fixture["revision"],
      max_length: 512
    )
  end

  after(:all) do
    @reranker = nil
    GC.start
  end

  it "loaded the same model_id and revision the fixture was generated from" do
    expect(reranker.model_id).to eq(fixture["model_id"])
    expect(reranker.options["revision"]).to eq(fixture["revision"])
  end

  fixture["pairs"].each_with_index do |pair, i|
    it "matches Python raw logits and sigmoid scores for pair #{i} (#{pair['token_count']} tokens)" do
      # Batch of one — matches how the fixture was generated (see script docstring) and
      # sidesteps the reranker's padding behavior, which spec/reranker_batching_spec.rb
      # covers separately.
      raw = reranker.rerank(pair["query"], [pair["document"]], apply_sigmoid: false)
      sigmoid = reranker.rerank(pair["query"], [pair["document"]], apply_sigmoid: true)

      expect(raw.first[:score]).to be_within(RAW_LOGIT_TOLERANCE).of(pair["raw_logit"])
      expect(sigmoid.first[:score]).to be_within(SIGMOID_TOLERANCE).of(pair["sigmoid_score"])
    end
  end
end
