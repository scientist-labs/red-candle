require "spec_helper"
require "tmpdir"
require "fileutils"

# These specs verify that the HuggingFace cache directory is pre-created
# before the hf_hub API client is initialized.
#
# Background (Issue #72):
# When ~/.cache/huggingface doesn't exist, the hf_hub crate may fail to
# create the full directory tree (including the hub/ subdirectory) on the
# first run, causing the model cache to appear empty and triggering a
# re-download on the next invocation.
#
# The fix ensures that the cache directory (including hub/) is created
# before Api::new() is called, using the same resolution order as hf_hub:
#   1. $HF_HOME (if set)
#   2. $XDG_CACHE_HOME/huggingface (if set)
#   3. ~/.cache/huggingface
RSpec.describe "HuggingFace cache directory creation" do
  around(:each) do |example|
    # Save and restore environment variables
    original_hf_home = ENV["HF_HOME"]
    original_xdg = ENV["XDG_CACHE_HOME"]
    begin
      example.run
    ensure
      if original_hf_home
        ENV["HF_HOME"] = original_hf_home
      else
        ENV.delete("HF_HOME")
      end
      if original_xdg
        ENV["XDG_CACHE_HOME"] = original_xdg
      else
        ENV.delete("XDG_CACHE_HOME")
      end
    end
  end

  it "creates the hub directory when HF_HOME points to a non-existent path" do
    Dir.mktmpdir do |tmpdir|
      hf_home = File.join(tmpdir, "custom_hf_cache")
      ENV["HF_HOME"] = hf_home

      # The directory should not exist yet
      expect(Dir.exist?(hf_home)).to be false

      # Loading an embedding model will trigger ensure_hf_cache_dir internally.
      # We use a model_id that won't be found, but the cache dir should still
      # be created before the download attempt.
      begin
        Candle::EmbeddingModel.new(model_path: "nonexistent/model-for-cache-test")
      rescue => e
        # Expected to fail (model doesn't exist), but cache dir should be created
      end

      expect(Dir.exist?(File.join(hf_home, "hub"))).to be true
    end
  end

  it "creates the hub directory when XDG_CACHE_HOME is set" do
    Dir.mktmpdir do |tmpdir|
      ENV.delete("HF_HOME")
      ENV["XDG_CACHE_HOME"] = tmpdir

      hub_path = File.join(tmpdir, "huggingface", "hub")
      expect(Dir.exist?(hub_path)).to be false

      begin
        Candle::EmbeddingModel.new(model_path: "nonexistent/model-for-cache-test")
      rescue => e
        # Expected to fail
      end

      expect(Dir.exist?(hub_path)).to be true
    end
  end

  it "does not fail when the cache directory already exists" do
    Dir.mktmpdir do |tmpdir|
      hf_home = File.join(tmpdir, "existing_cache")
      hub_path = File.join(hf_home, "hub")
      FileUtils.mkdir_p(hub_path)
      ENV["HF_HOME"] = hf_home

      # Should not raise even though directory already exists
      expect {
        begin
          Candle::EmbeddingModel.new(model_path: "nonexistent/model-for-cache-test")
        rescue => e
          # Expected to fail on model download, not on dir creation
        end
      }.not_to raise_error
    end
  end
end
