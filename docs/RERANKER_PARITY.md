# Reranker parity with Python Sentence Transformers

`spec/reranker_parity_spec.rb` proves that `Candle::Reranker`'s BERT cross-encoder path
(`RerankerModel::Bert` in `ext/candle/src/ruby/reranker.rs`) computes the same scores as
Python `transformers`/`sentence-transformers`, for the exact model and pinned revision
Benchmate uses: `cross-encoder/ms-marco-MiniLM-L6-v2` @
`233902d25c440f23af6f7d6e94d2946bac0bee0a`.

## What it proves

- **Architecture match**: the fixture generator asserts the model's `state_dict` has
  `bert.pooler.dense.{weight,bias}` and `classifier.{weight,bias}` — exactly the layers
  `RerankerModel::Bert` loads (`vb.pp("bert.pooler.dense")`, `vb.pp("classifier")`) and
  runs through pooler (dense + tanh) → classifier, matching `pooling_method: "pooler"`
  (the default). If this model's architecture ever changed shape, the generator script
  fails loudly instead of silently producing a fixture that no longer represents it.
- **Numerical match**: for 7 query/document pairs (a clear match, a clear mismatch,
  near-duplicate phrasing, and a document long enough to force truncation at
  `max_length: 512`), Red Candle's raw logits and sigmoid-activated scores are compared
  against Python's within a documented tolerance.

## Why pairs are scored one at a time

Both the fixture generator and `reranker_parity_spec.rb` rerank **one document per call**
(no padding). This isolates model/pooler/classifier correctness from padding behavior —
Red Candle pads micro-batches with more than one document to the **right**
(`ext/candle/src/ruby/reranker.rs`, scoped to the reranker only), which combined with
candle-transformers' un-offset absolute position ids (`0..seq_len` regardless of padding —
see the `// TODO: Proper absolute positions?` in `candle-transformers`' `bert.rs`) is
required to keep shorter documents in a mixed-length batch getting correct position
embeddings. `spec/reranker_batching_spec.rb` separately proves multi-document micro-batches
are internally consistent (same ranking regardless of `batch_size`); this fixture stays
scoped to proving the single-pair math is right.

## Tolerance

Measured max deviation on macOS arm64 (Accelerate backend) across the 7 fixture pairs:

| | max |diff| |
|---|---|
| raw logit | ~8e-6 |
| sigmoid score | ~0 (rounds away at 1e-6) |

`spec/reranker_parity_spec.rb` uses `RAW_LOGIT_TOLERANCE = 1e-3` and
`SIGMOID_TOLERANCE = 1e-4` — roughly 100x the measured ceiling, to absorb backend
floating-point differences (candle's CPU gemm vs. PyTorch's, different BLAS
implementations across development/CI/production CPUs) without being loose enough to mask
a real architecture or pooling regression.

## Regenerating the fixture

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
python3 scripts/generate_reranker_parity_fixtures.py
```

Writes `spec/fixtures/reranker_parity/ms_marco_minilm_l6.json`, which records the exact
`torch`/`transformers`/`sentence_transformers` versions used, alongside the scores. Commit
the regenerated fixture if you intentionally change `MODEL_ID`, `REVISION`, or the `PAIRS`
list in `scripts/generate_reranker_parity_fixtures.py`.
