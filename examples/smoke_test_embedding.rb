# frozen_string_literal: true

require_relative "smoke_test_helper"

section "Embedding Model"

emb = test("load") do
  Candle::EmbeddingModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", device: $device)
end

if emb
  test("embedding (pooled_normalized)") do
    e = emb.embedding("Hello, world!")
    raise "empty embedding" if e.values.empty?
    e
  end
  test("embedding (cls)") { emb.embedding("Hello", pooling_method: "cls") }
  test("embedding (pooled)") { emb.embedding("Hello", pooling_method: "pooled") }
  test("embeddings (batch)") do
    results = emb.embeddings("Hello, world!")
    raise "empty" if results.values.empty?
    results
  end
  test("inspect") { emb.inspect }
end

smoke_test_summary
