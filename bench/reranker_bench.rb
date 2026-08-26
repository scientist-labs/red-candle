#!/usr/bin/env ruby
# frozen_string_literal: true

# Reproducible reranker benchmark for Candle::Reranker, sized for Benchmate's target
# model/workload: cross-encoder/ms-marco-MiniLM-L6-v2, up to 100 candidates per call,
# CPU-only inference.
#
# Run on Linux amd64 CPU (the production target) with:
#
#   bundle exec ruby bench/reranker_bench.rb
#
# See docs/RERANKER_BENCHMARKS.md for measured results and how to interpret them —
# including which platform they were measured on. Writes bench/reranker_bench_results.csv.
#
# Flags:
#   --quick   Small matrix + few iterations, for a fast sanity check of the script itself.

require "bundler/setup"
require "candle"
require "benchmark"

QUICK = ARGV.include?("--quick")

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

CANDIDATE_COUNTS = QUICK ? [10, 100] : [10, 25, 50, 100]
MAX_LENGTHS      = QUICK ? [128, 512] : [128, 256, 512]
BATCH_SIZES      = QUICK ? [16] : [8, 16, 32]
WARM_ITERATIONS  = QUICK ? 5 : 15
CONCURRENCY_LEVELS = [1, 4]
CONCURRENT_CALLS_PER_THREAD = QUICK ? 3 : 6
MEMORY_GROWTH_ITERATIONS = QUICK ? 10 : 50

QUERY = "What are the benefits of regular exercise for cardiovascular health?"

# Synthetic documents of varying, realistic length so max_length truncation and batching
# both get meaningfully exercised. Deterministic (no RNG) so runs are comparable.
TOPICS = [
  "Regular exercise improves cardiovascular health and reduces the risk of heart disease over time.",
  "The stock market fluctuates based on interest rates, earnings reports, and investor sentiment.",
  "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen in plants.",
  "The Roman Empire spanned three continents at its height and lasted for over a thousand years.",
  "Deep learning models require large datasets and significant compute for effective training.",
  "Coral reefs support roughly a quarter of all marine species despite covering a tiny fraction of the ocean floor."
].freeze

def build_documents(count)
  (0...count).map do |i|
    base = TOPICS[i % TOPICS.length]
    # Vary length: every third document gets padded out with extra sentences, to mix
    # short and long sequences within a micro-batch (the case the padding-direction fix
    # in ext/candle/src/ruby/reranker.rs targets).
    extra = (i % 3 == 0) ? " #{base} #{base}" : ""
    "#{base}#{extra} (candidate #{i})"
  end
end

def rss_kb(pid = Process.pid)
  `ps -o rss= -p #{pid}`.to_i
end

def percentile(sorted_values, pct)
  return sorted_values.first if sorted_values.length == 1
  idx = ((pct / 100.0) * (sorted_values.length - 1)).round
  sorted_values[idx]
end

results = []

puts "=" * 100
puts "Candle::Reranker benchmark — #{MODEL_ID} @ #{REVISION}"
puts "Platform: #{RUBY_PLATFORM}, Ruby #{RUBY_VERSION}"
puts "=" * 100

MAX_LENGTHS.each do |max_length|
  BATCH_SIZES.each do |batch_size|
    print "\nLoading reranker (max_length=#{max_length}, batch_size=#{batch_size})... "
    cold_load_s = Benchmark.realtime do
      $reranker = Candle::Reranker.from_pretrained(
        MODEL_ID, device: Candle::Device.cpu, revision: REVISION,
        max_length: max_length, batch_size: batch_size
      )
    end
    reranker = $reranker
    puts "done in #{(cold_load_s * 1000).round(1)}ms"

    CANDIDATE_COUNTS.each do |count|
      documents = build_documents(count)

      # Warm up (exclude from timing — page cache, tokenizer JIT-ish paths, etc.)
      2.times { reranker.rerank(QUERY, documents) }

      latencies_ms = []
      rss_before = rss_kb
      WARM_ITERATIONS.times do
        t = Benchmark.realtime { reranker.rerank(QUERY, documents) }
        latencies_ms << (t * 1000)
      end
      rss_after = rss_kb

      sorted = latencies_ms.sort
      p50 = percentile(sorted, 50)
      p95 = percentile(sorted, 95)
      throughput = count / (p50 / 1000.0)

      row = {
        max_length: max_length,
        batch_size: batch_size,
        candidate_count: count,
        cold_load_ms: (cold_load_s * 1000).round(1),
        warm_p50_ms: p50.round(2),
        warm_p95_ms: p95.round(2),
        throughput_docs_per_sec: throughput.round(1),
        rss_before_kb: rss_before,
        rss_after_kb: rss_after,
        rss_growth_kb: rss_after - rss_before,
        concurrency_1_p50_ms: nil,
        concurrency_4_p50_ms: nil
      }

      # Single- vs multi-threaded caller latency, only for the "realistic" combo (largest
      # candidate count) to keep total runtime bounded — this is the shape that matters for
      # Benchmate's concurrent-Puma-threads use case.
      if count == CANDIDATE_COUNTS.max
        CONCURRENCY_LEVELS.each do |n_threads|
          per_thread_latencies = Array.new(n_threads) { [] }
          Benchmark.realtime do
            threads = Array.new(n_threads) do |ti|
              Thread.new do
                CONCURRENT_CALLS_PER_THREAD.times do
                  t = Benchmark.realtime { reranker.rerank(QUERY, documents) }
                  per_thread_latencies[ti] << (t * 1000)
                end
              end
            end
            threads.each(&:join)
          end
          all_latencies = per_thread_latencies.flatten.sort
          row[:"concurrency_#{n_threads}_p50_ms"] = percentile(all_latencies, 50).round(2)
        end
      end

      results << row
      printf(
        "  count=%3d  p50=%7.2fms  p95=%7.2fms  throughput=%7.1f docs/s  rss_growth=%+dkB\n",
        count, row[:warm_p50_ms], row[:warm_p95_ms], row[:throughput_docs_per_sec], row[:rss_growth_kb]
      )
    end
  end
end

# Repeated-call memory growth: hammer one representative config many times, check RSS
# before vs after to catch leaks (mmap'd weights should be shared/reused, not re-allocated).
puts "\n" + "=" * 100
puts "Repeated-call memory growth (#{MEMORY_GROWTH_ITERATIONS} calls, max_length=512, batch_size=16, 100 docs)"
reranker = Candle::Reranker.from_pretrained(
  MODEL_ID, device: Candle::Device.cpu, revision: REVISION, max_length: 512, batch_size: 16
)
documents = build_documents(100)
reranker.rerank(QUERY, documents) # warm up
GC.start
rss_start = rss_kb
MEMORY_GROWTH_ITERATIONS.times { reranker.rerank(QUERY, documents) }
GC.start
rss_end = rss_kb
puts "RSS before: #{rss_start}kB, RSS after: #{rss_end}kB, growth: #{rss_end - rss_start}kB " \
     "(#{((rss_end - rss_start).to_f / MEMORY_GROWTH_ITERATIONS).round(2)}kB/call)"

csv_path = File.join(__dir__, "reranker_bench_results.csv")
File.open(csv_path, "w") do |f|
  f.puts results.first.keys.join(",")
  results.each { |r| f.puts r.values.map { |v| v.nil? ? "" : v }.join(",") }
end
puts "\nWrote #{csv_path}"
puts "\nPeak RSS observed: #{results.map { |r| r[:rss_after_kb] }.max}kB " \
     "(~#{(results.map { |r| r[:rss_after_kb] }.max / 1024.0 / 1024.0).round(3)} GiB)"
