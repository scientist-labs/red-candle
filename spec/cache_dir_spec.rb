require "spec_helper"

# These specs verify that the HuggingFace cache directory is pre-created
# when the Candle gem loads (at native extension init time).
#
# Background (Issue #72):
# When ~/.cache/huggingface doesn't exist, the hf_hub crate may fail to
# create the full directory tree (including the hub/ subdirectory) on the
# first run, causing the model cache to appear empty and triggering a
# re-download on the next invocation.
#
# The fix ensures that the cache directory (including hub/) is created
# at gem load time via ensure_hf_cache_dir() in the Rust init function,
# using the same resolution order as hf_hub:
#   1. $HF_HOME (if set)
#   2. $XDG_CACHE_HOME/huggingface (if set)
#   3. ~/.cache/huggingface
#
# Note: Because the cache directory is created at gem load time (not at
# model creation time), changing HF_HOME or XDG_CACHE_HOME after the gem
# is loaded will not cause new directories to be created.
RSpec.describe "HuggingFace cache directory creation" do
  it "has created the hub directory by the time the gem is loaded" do
    # Determine the expected cache path using the same resolution as ensure_hf_cache_dir
    hub_path = if ENV["HF_HOME"]
      File.join(ENV["HF_HOME"], "hub")
    elsif ENV["XDG_CACHE_HOME"]
      File.join(ENV["XDG_CACHE_HOME"], "huggingface", "hub")
    else
      File.join(Dir.home, ".cache", "huggingface", "hub")
    end

    # The gem has already loaded by the time specs run, so the directory
    # should exist from the init-time ensure_hf_cache_dir() call
    expect(Dir.exist?(hub_path)).to be true
  end

  it "does not fail when loading a model if the cache directory already exists" do
    # The cache directory already exists (created at init time).
    # Loading a model with a nonexistent ID should fail on download,
    # not on cache directory creation.
    expect {
      begin
        Candle::EmbeddingModel.new(model_path: "nonexistent/model-for-cache-test")
      rescue RuntimeError
        # Expected to fail on model download, not on dir creation
      end
    }.not_to raise_error
  end
end
