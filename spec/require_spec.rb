require "spec_helper"

# These specs verify that the native extension loads correctly via $LOAD_PATH
# resolution (using `require`) rather than relative path resolution
# (using `require_relative`).
#
# Background (Issue #75):
# RubyGems installs native extensions into a separate extensions directory
# (e.g., ~/.gem/ruby/3.4.0/extensions/...) and adds that directory to
# $LOAD_PATH. Using `require_relative` bypasses $LOAD_PATH and looks only
# in the gem's lib/ directory, where the compiled .so/.bundle file does not
# exist. Using `require` resolves via $LOAD_PATH and finds the extension
# in the correct location.
RSpec.describe "Native extension loading" do
  it "loads the Candle module successfully" do
    expect(defined?(Candle)).to eq("constant")
  end

  it "makes Candle::Tensor available" do
    expect(defined?(Candle::Tensor)).to eq("constant")
  end

  it "makes Candle::EmbeddingModel available" do
    expect(defined?(Candle::EmbeddingModel)).to eq("constant")
  end

  it "makes Candle::DType available" do
    expect(defined?(Candle::DType)).to eq("constant")
  end

  it "makes Candle::Device available" do
    expect(defined?(Candle::Device)).to eq("constant")
  end

  it "can create a tensor (proves native extension is functional)" do
    t = Candle::Tensor.new([1.0, 2.0, 3.0], :f32)
    expect(t).to be_a(Candle::Tensor)
    expect(t.shape).to eq([3])
  end

  it "loads candle/candle via require (not require_relative)" do
    # Verify that lib/candle.rb uses `require` for the native extension.
    # This is critical because RubyGems places compiled extensions in the
    # extensions directory, not in the gem's lib/ directory.
    candle_rb = File.read(File.expand_path("../lib/candle.rb", __dir__))
    expect(candle_rb).to include('require "candle/candle"')
    expect(candle_rb).not_to include('require_relative "candle/candle"')
  end
end
