#!/usr/bin/env ruby
# frozen_string_literal: true

# Isolated repeated-call memory growth check for Candle::Reranker — run standalone (own
# process, nothing else loaded first) so RSS deltas reflect only this reranker's behavior,
# not GC cleanup of unrelated objects from a prior sweep. See docs/RERANKER_BENCHMARKS.md.
#
#   bundle exec ruby bench/reranker_memory_growth.rb

require "bundler/setup"
require "candle"

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
ITERATIONS = 50

def rss_kb
  `ps -o rss= -p #{Process.pid}`.to_i
end

QUERY = "What are the benefits of regular exercise for cardiovascular health?"
documents = (0...100).map { |i| "Regular exercise improves cardiovascular health and reduces disease risk. (candidate #{i})" }

reranker = Candle::Reranker.from_pretrained(
  MODEL_ID, device: Candle::Device.cpu, revision: REVISION, max_length: 512, batch_size: 16
)
reranker.rerank(QUERY, documents) # warm up
GC.start(full_mark: true, immediate_sweep: true)
rss_after_load = rss_kb

samples = [rss_after_load]
ITERATIONS.times do |i|
  reranker.rerank(QUERY, documents)
  samples << rss_kb if (i + 1) % 10 == 0
end
GC.start(full_mark: true, immediate_sweep: true)
rss_final = rss_kb

puts "RSS after load + warmup: #{rss_after_load}kB"
puts "RSS every 10 calls: #{samples.join(' -> ')}kB"
puts "RSS after #{ITERATIONS} calls (pre-GC): #{samples.last}kB"
puts "RSS after #{ITERATIONS} calls (post-GC.start): #{rss_final}kB"
puts "Net growth (post-GC): #{rss_final - rss_after_load}kB " \
     "(#{((rss_final - rss_after_load).to_f / ITERATIONS).round(2)}kB/call)"
