require_relative "candle/logger"

# Load the compiled Rust extension. Precompiled (platform) gems install it into a
# Ruby-ABI-versioned subdir (lib/candle/<major.minor>/candle.{so,bundle}) so a single
# fat gem can carry a binary per Ruby version; source/dev builds place it flat at
# lib/candle/candle.{so,bundle}. Try the versioned path first, fall back to the flat
# one. Resolution goes through $LOAD_PATH (`require`, never `require_relative`) because
# RubyGems installs native extensions outside the gem's lib/ dir — see
# spec/require_spec.rb and Issue #75.
begin
  RUBY_VERSION =~ /(\d+\.\d+)/
  require "candle/#{Regexp.last_match(1)}/candle"
rescue LoadError
  require "candle/candle"
end

require_relative "candle/tensor"
require_relative "candle/device_utils"
require_relative "candle/embedding_model_type"
require_relative "candle/embedding_model"
require_relative "candle/reranker"
require_relative "candle/tool"
require_relative "candle/tool_call_parser"
require_relative "candle/thinking_parser"
require_relative "candle/chat_response"
require_relative "candle/stream_processor"
require_relative "candle/llm"
require_relative "candle/agent"
require_relative "candle/tokenizer"
require_relative "candle/ner"
require_relative "candle/vlm"
require_relative "candle/build_info"
