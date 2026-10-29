require "spec_helper"

RSpec.describe "EmbeddingModel max_length configuration" do
  let(:long_text) { "Artificial intelligence and machine learning. " * 200 }

  after(:all) do
    @model = nil
    @default_model = nil
    GC.start
  end

  describe "with max_length" do
    let(:model) do
      @model ||= Candle::EmbeddingModel.from_pretrained(device: Candle::Device.cpu, max_length: 128)
    end

    it "truncates inputs longer than max_length" do
      expect(model.embeddings(long_text).shape[1]).to eq(128)
    end

    it "embeds inputs shorter than max_length as the default model does" do
      @default_model ||= Candle::EmbeddingModel.from_pretrained(device: Candle::Device.cpu)

      expect(model.embedding("Hi there").values).to eq(@default_model.embedding("Hi there").values)
    end

    it "includes max_length in options" do
      expect(model.options).to include("max_length" => 128)
    end
  end

  describe "backward compatibility" do
    it "accepts max_length on the deprecated .new method" do
      expect {
        model = Candle::EmbeddingModel.new(device: Candle::Device.cpu, max_length: 128)
        expect(model.embeddings(long_text).shape[1]).to eq(128)
      }.to output(/DEPRECATION/).to_stderr
    end
  end
end
