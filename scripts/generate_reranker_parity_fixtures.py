#!/usr/bin/env python3
"""Generate reference reranker scores from Python Sentence Transformers for parity testing.

Produces spec/fixtures/reranker_parity/ms_marco_minilm_l6.json, which
spec/reranker_parity_spec.rb compares Red Candle's Candle::Reranker output against.

Each query/document pair is tokenized and scored ONE AT A TIME (batch of one, no
padding). This deliberately matches Red Candle's batch-of-one code path exactly, so the
fixture isolates model/pooler/classifier correctness from any batching or padding
behavior (those are covered separately by spec/reranker_batching_spec.rb).

Usage:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers sentence-transformers
    python3 scripts/generate_reranker_parity_fixtures.py

See docs/RERANKER_PARITY.md for what this fixture proves and how to interpret it.
"""

import json
import pathlib
import sys

import torch
import transformers
import sentence_transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MAX_LENGTH = 512

# Query/document pairs chosen to cover: a clearly relevant match, a clearly irrelevant
# match, near-duplicate phrasing, and a long document that exercises truncation at
# MAX_LENGTH.
PAIRS = [
    {
        "query": "What is the capital of France?",
        "document": "The capital of France is Paris.",
    },
    {
        "query": "What is the capital of France?",
        "document": "Berlin is the capital of Germany.",
    },
    {
        "query": "How many people live in London?",
        "document": "Around 9 million people live in London.",
    },
    {
        "query": "How many people live in London?",
        "document": "The weather in London is often rainy.",
    },
    {
        "query": "What are the benefits of regular exercise?",
        "document": (
            "Regular exercise improves cardiovascular health, strengthens muscles and "
            "bones, boosts mood through endorphin release, and helps maintain a healthy "
            "body weight over time."
        ),
    },
    {
        "query": "What are the benefits of regular exercise?",
        "document": "The train departs from platform nine at noon.",
    },
    {
        # Exercises truncation: this document is long enough to exceed MAX_LENGTH tokens.
        "query": "What is machine learning?",
        "document": (
            "Machine learning is a subfield of artificial intelligence. " * 80
        ).strip(),
    },
]


def main():
    print(f"Loading {MODEL_ID} @ {REVISION} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=REVISION)
    model.eval()

    # Sanity-check the architecture assumption RerankerModel::Bert makes in
    # ext/candle/src/ruby/reranker.rs: a BERT encoder + pooler (dense+tanh on CLS) +
    # a single-logit classifier head, loaded from `bert.pooler.dense.*` / `classifier.*`.
    state_dict_keys = set(model.state_dict().keys())
    expected_keys = {
        "bert.pooler.dense.weight",
        "bert.pooler.dense.bias",
        "classifier.weight",
        "classifier.bias",
    }
    missing = expected_keys - state_dict_keys
    if missing:
        raise RuntimeError(
            f"Model state_dict is missing expected keys {missing} — "
            "RerankerModel::Bert's architecture assumption does not hold for this revision."
        )
    print(f"Confirmed state_dict has {sorted(expected_keys)}", file=sys.stderr)

    results = []
    with torch.no_grad():
        for pair in PAIRS:
            encoded = tokenizer(
                pair["query"],
                pair["document"],
                truncation="longest_first",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            token_count = encoded["input_ids"].shape[1]
            logits = model(**encoded).logits
            raw_logit = logits.squeeze().item()
            sigmoid_score = torch.sigmoid(logits).squeeze().item()

            results.append({
                "query": pair["query"],
                "document": pair["document"],
                "max_length": MAX_LENGTH,
                "token_count": token_count,
                "raw_logit": raw_logit,
                "sigmoid_score": sigmoid_score,
            })
            print(
                f"  raw={raw_logit:+.6f} sigmoid={sigmoid_score:.6f} "
                f"tokens={token_count} doc={pair['document'][:50]!r}",
                file=sys.stderr,
            )

    fixture = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "pooling_method": "pooler",
        "generated_with": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
        "note": (
            "Each pair tokenized and scored individually (batch of one, no padding) — "
            "matches Red Candle's batch-of-one path exactly. See docs/RERANKER_PARITY.md."
        ),
        "pairs": results,
    }

    out_path = pathlib.Path(__file__).resolve().parent.parent / "spec" / "fixtures" / "reranker_parity" / "ms_marco_minilm_l6.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
