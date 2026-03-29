# frozen_string_literal: true

require_relative "smoke_test_helper"

docs = ["Ruby is a programming language", "Python is a snake", "Java is an island"]

# BERT
section "Reranker (BERT)"

reranker = test("load (BERT)") do
  Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2", device: $device)
end

if reranker
  test("rerank (pooler)") { reranker.rerank("What is Ruby?", docs, pooling_method: "pooler") }
  test("rerank (cls)") { reranker.rerank("What is Ruby?", docs, pooling_method: "cls") }
  test("rerank (mean)") { reranker.rerank("What is Ruby?", docs, pooling_method: "mean") }
  test("inspect") { reranker.inspect }

  reranker = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

# XLM-RoBERTa / BGE
section "Reranker (XLM-RoBERTa / BGE)"

bge_reranker = test("load (BGE)") do
  Candle::Reranker.from_pretrained("BAAI/bge-reranker-base", device: $device)
end

if bge_reranker
  test("rerank (bge)") do
    results = bge_reranker.rerank("What is Ruby?", docs)
    raise "wrong top result" unless results[0][:text].include?("Ruby")
    results
  end
  test("model_id") { bge_reranker.model_id }
  test("tokenizer access") { bge_reranker.tokenizer }
  test("inspect") { bge_reranker.inspect }

  bge_reranker = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

# DeBERTa
section "Reranker (DeBERTa)"

deberta_reranker = test("load (mxbai)") do
  Candle::Reranker.from_pretrained("mixedbread-ai/mxbai-rerank-base-v1", device: $device)
end

if deberta_reranker
  test("rerank (deberta)") do
    results = deberta_reranker.rerank("What is Ruby?", docs)
    raise "wrong top result" unless results[0][:text].include?("Ruby")
    results
  end
  test("inspect") { deberta_reranker.inspect }

  deberta_reranker = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

# ModernBERT
section "Reranker (ModernBERT)"

modernbert_reranker = test("load (gte)") do
  Candle::Reranker.from_pretrained("Alibaba-NLP/gte-reranker-modernbert-base", device: $device)
end

if modernbert_reranker
  test("rerank (modernbert)") do
    results = modernbert_reranker.rerank("What is Ruby?", docs)
    raise "wrong top result" unless results[0][:text].include?("Ruby")
    results
  end
  test("inspect") { modernbert_reranker.inspect }

  modernbert_reranker = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

# Qwen3
section "Reranker (Qwen3)"

qwen3_reranker = test("load (Qwen3)") do
  Candle::Reranker.from_pretrained("Qwen/Qwen3-Reranker-0.6B", device: $device)
end

if qwen3_reranker
  test("rerank (qwen3)") do
    results = qwen3_reranker.rerank("What is Ruby?", docs)
    raise "wrong top result" unless results[0][:text].include?("Ruby")
    raise "score not in 0-1 range" unless results[0][:score] >= 0.0 && results[0][:score] <= 1.0
    results
  end
  test("inspect") { qwen3_reranker.inspect }

  qwen3_reranker = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

smoke_test_summary
