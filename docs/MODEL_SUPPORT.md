# Model Support in Red-Candle

## LLM Models

### Supported Architectures

| Architecture | Format | Module |
|:-------------|:-------|:-------|
| Llama / TinyLlama / Yi / SmolLM2 | safetensors, GGUF | `llama` |
| Mistral | GGUF | `mistral` |
| Gemma 3 | GGUF | `gemma` |
| Qwen 2 / 2.5 | GGUF | `qwen` |
| Qwen 3 | safetensors, GGUF | `qwen3` |
| Phi 2 / 3 / 4 | safetensors, GGUF | `phi` |
| GLM-4 | safetensors, GGUF | `glm4` |
| Granite 7B | safetensors | `granite` |
| Granite 4.0 | safetensors | `granitemoehybrid` |

GGUF models auto-detect architecture from metadata. Any GGUF model using the above architectures should work.

### Known Working Models

| Model | Model Path | Format | Initializer |
|:------|:-----------|:-------|:------------|
| Mistral v0.2 | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` | GGUF | `Candle::LLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", gguf_file: "mistral-7b-instruct-v0.2.Q4_K_M.gguf")` |
| Mistral v0.3 | `MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF` | GGUF | `Candle::LLM.from_pretrained("MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF", gguf_file: "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", tokenizer: "mistralai/Mistral-7B-Instruct-v0.3")` |
| TinyLlama 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | safetensors | `Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")` |
| TinyLlama 1.1B | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | GGUF | `Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")` |
| Gemma 3 4B | `google/gemma-3-4b-it-qat-q4_0-gguf` | GGUF | `Candle::LLM.from_pretrained("google/gemma-3-4b-it-qat-q4_0-gguf", gguf_file: "gemma-3-4b-it-q4_0.gguf", tokenizer: "google/gemma-3-4b-it")` |
| Qwen 2.5 1.5B | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | GGUF | `Candle::LLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct-GGUF", gguf_file: "qwen2.5-1.5b-instruct-q4_k_m.gguf")` |
| Qwen3 0.6B | `Qwen/Qwen3-0.6B` | safetensors | `Candle::LLM.from_pretrained("Qwen/Qwen3-0.6B")` |
| Qwen3 0.6B | `MaziyarPanahi/Qwen3-0.6B-GGUF` | GGUF | `Candle::LLM.from_pretrained("MaziyarPanahi/Qwen3-0.6B-GGUF", gguf_file: "Qwen3-0.6B.Q4_K_M.gguf", tokenizer: "Qwen/Qwen3-0.6B")` |
| Qwen3 1.7B | `MaziyarPanahi/Qwen3-1.7B-GGUF` | GGUF | `Candle::LLM.from_pretrained("MaziyarPanahi/Qwen3-1.7B-GGUF", gguf_file: "Qwen3-1.7B.Q4_K_M.gguf", tokenizer: "Qwen/Qwen3-1.7B")` |
| SmolLM2 360M | `HuggingFaceTB/SmolLM2-360M-Instruct` | safetensors | `Candle::LLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")` |
| SmolLM2 360M | `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF` | GGUF | `Candle::LLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", gguf_file: "smollm2-360m-instruct-q8_0.gguf")` |
| Phi-2 | `microsoft/phi-2` | safetensors | `Candle::LLM.from_pretrained("microsoft/phi-2")` |
| Phi-2 | `TheBloke/phi-2-GGUF` | GGUF | `Candle::LLM.from_pretrained("TheBloke/phi-2-GGUF", gguf_file: "phi-2.Q4_K_M.gguf")` |
| Phi-3 mini 4k | `microsoft/Phi-3-mini-4k-instruct` | safetensors | `Candle::LLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")` |
| Phi-3 mini 128k | `microsoft/Phi-3-mini-128k-instruct` | safetensors | `Candle::LLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct")` |
| Phi-3 mini 4k | `microsoft/Phi-3-mini-4k-instruct-gguf` | GGUF | `Candle::LLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct-gguf", gguf_file: "Phi-3-mini-4k-instruct-q4.gguf")` |
| Phi-4 | `microsoft/phi-4-gguf` | GGUF | `Candle::LLM.from_pretrained("microsoft/phi-4-gguf", gguf_file: "phi-4-Q4_K_S.gguf")` |
| Yi-1.5 6B | `01-ai/Yi-1.5-6B-Chat` | safetensors | `Candle::LLM.from_pretrained("01-ai/Yi-1.5-6B-Chat")` |
| Yi-1.5 6B | `bartowski/Yi-1.5-6B-Chat-GGUF` | GGUF | `Candle::LLM.from_pretrained("bartowski/Yi-1.5-6B-Chat-GGUF", gguf_file: "Yi-1.5-6B-Chat-Q4_K_M.gguf", tokenizer: "01-ai/Yi-1.5-6B-Chat")` |
| GLM-4 9B | `THUDM/GLM-4-9B-0414` | safetensors | `Candle::LLM.from_pretrained("THUDM/GLM-4-9B-0414")` |
| GLM-4 9B | `bartowski/THUDM_GLM-4-9B-0414-GGUF` | GGUF | `Candle::LLM.from_pretrained("bartowski/THUDM_GLM-4-9B-0414-GGUF", gguf_file: "THUDM_GLM-4-9B-0414-Q4_K_M.gguf", tokenizer: "THUDM/GLM-4-9B-0414")` |
| Granite 7B | `ibm-granite/granite-7b-instruct` | safetensors | `Candle::LLM.from_pretrained("ibm-granite/granite-7b-instruct")` |
| Granite 4.0 Micro | `ibm-granite/granite-4.0-micro` | safetensors | `Candle::LLM.from_pretrained("ibm-granite/granite-4.0-micro")` |

### Partially Working LLM Models

| Model | Issue |
|:------|:------|
| `bartowski/glm-4-9b-chat-1m-GGUF` | Older GLM-4 ships `tokenizer.model` (tiktoken) not `tokenizer.json`. Use GLM-4-9B-0414 instead. |
| `ibm-granite/granite-3.3-2b-instruct` | Garbage output. Upstream candle-transformers lacks Granite 3.x scaling multipliers. |
| `Qwen/Qwen2.5-0.5B-Instruct` (safetensors) | Garbage output via safetensors, upstream candle bug [#2295](https://github.com/huggingface/candle/issues/2295). GGUF version works fine. |
| `MaziyarPanahi/Mistral-Nemo-Instruct-2407-GGUF` | Shape mismatch in GGUF loader (different GQA head configuration). |
| `mistralai/Ministral-8B-Instruct-2410` | Cannot find embed_tokens weight (sharded safetensors layout issue). |

---

## Embedding Models

### Supported Architectures

| Architecture | Type | Module |
|:-------------|:-----|:-------|
| JinaBert | Encoder | `jina_bert` |
| BERT | Encoder | `bert` |
| DistilBERT | Encoder | `distilbert` |
| MiniLM | Encoder (BERT-based) | `bert` |

### Pooling Methods

| Method | Description |
|:-------|:------------|
| `pooled_normalized` | Mean pooling with L2 normalization (default, recommended for similarity) |
| `pooled` | Mean pooling without normalization |
| `cls` | CLS token extraction |

### Known Working Models

| Model | Params | Dimensions | Initializer |
|:------|:-------|:-----------|:------------|
| Jina Embeddings v2 Base EN | 137M | 768 | `Candle::EmbeddingModel.from_pretrained("jinaai/jina-embeddings-v2-base-en")` |
| Jina Embeddings v2 Small EN | 33M | 512 | `Candle::EmbeddingModel.from_pretrained("jinaai/jina-embeddings-v2-small-en")` |

Any BERT-based or DistilBERT-based embedding model from HuggingFace should work. Specify the model type if auto-detection fails:

```ruby
model = Candle::EmbeddingModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", model_type: "minilm")
```

---

## Reranker Models

### Supported Architectures

| Architecture | Module | Notes |
|:-------------|:-------|:------|
| BERT | `bert` | Manual pooler + classifier layers. Pooling method configurable (`pooler`, `cls`, `mean`). |
| XLM-RoBERTa | `xlm_roberta` | Built-in classification head via `XLMRobertaForSequenceClassification`. Pooling method ignored. |
| DeBERTa v2/v3 | `debertav2` | Context pooler + classifier. Used by Mixedbread mxbai-rerank models (BEIR leaderboard leaders). |
| ModernBERT | `modernbert` | CLS pooling + head + classifier. Best accuracy-per-parameter ratio. |
| Qwen3 (decoder) | `qwen3` | Decoder-based yes/no scoring. Scores are P(yes) from logits. Top of MTEB multilingual leaderboard. |

Model type is auto-detected from `config.json` `model_type` field.

### Known Working Models

| Model | Architecture | Params | Quality | Initializer |
|:------|:-------------|:-------|:--------|:------------|
| BGE Reranker Base | XLM-RoBERTa | 278M | Strong quality | `Candle::Reranker.from_pretrained("BAAI/bge-reranker-base")` |
| BGE Reranker Large | XLM-RoBERTa | 560M | Best BGE quality | `Candle::Reranker.from_pretrained("BAAI/bge-reranker-large")` |
| BGE Reranker v2 M3 | XLM-RoBERTa | 568M | Multilingual | `Candle::Reranker.from_pretrained("BAAI/bge-reranker-v2-m3")` |
| Jina Reranker v2 | XLM-RoBERTa | 278M | Multilingual | `Candle::Reranker.from_pretrained("jinaai/jina-reranker-v2-base-multilingual")` |
| mxbai Rerank Base v1 | DeBERTa v3 | 184M | BEIR leader | `Candle::Reranker.from_pretrained("mixedbread-ai/mxbai-rerank-base-v1")` |
| mxbai Rerank Large v1 | DeBERTa v3 | 435M | BEIR leader (large) | `Candle::Reranker.from_pretrained("mixedbread-ai/mxbai-rerank-large-v1")` |
| GTE Reranker ModernBERT | ModernBERT | 149M | Best accuracy/size ratio | `Candle::Reranker.from_pretrained("Alibaba-NLP/gte-reranker-modernbert-base")` |
| Qwen3 Reranker 0.6B | Qwen3 | 600M | MTEB multilingual leader | `Candle::Reranker.from_pretrained("Qwen/Qwen3-Reranker-0.6B")` |
| Qwen3 Reranker 4B | Qwen3 | 4B | Higher quality | `Candle::Reranker.from_pretrained("Qwen/Qwen3-Reranker-4B")` |
| MiniLM L-12 v2 | BERT | 33M | Lightweight | `Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")` |
| MiniLM L-6 v2 | BERT | 22M | Fastest | `Candle::Reranker.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")` |

### Reranker Roadmap (Not Yet Supported)

These architectures are available in `candle-transformers` and could be added:

| Architecture | Models | Priority | Notes |
|:-------------|:-------|:---------|:------|
| Gemma2 (decoder) | `BAAI/bge-reranker-v2.5-gemma2-lightweight` | Medium | Decoder-based reranker, similar prompt-based scoring pattern to Qwen3. `gemma2` module exists in candle. |

---

## NER Models (Named Entity Recognition)

### Supported Architectures

| Architecture | Module | Notes |
|:-------------|:-------|:------|
| BERT (token classification) | `bert` | BertModel with classification head over each token |

### Known Working Models

| Model | Entities | Initializer |
|:------|:---------|:------------|
| dslim/bert-base-NER | PER, LOC, ORG, MISC | `Candle::NER.from_pretrained("dslim/bert-base-NER", tokenizer: "bert-base-cased")` |
| Babelscape/wikineural-multilingual-ner | PER, LOC, ORG, MISC | `Candle::NER.from_pretrained("Babelscape/wikineural-multilingual-ner")` |

### Pattern & Gazetteer Recognizers (No Model Required)

```ruby
# Pattern-based (regex)
rec = Candle::PatternEntityRecognizer.new("EMAIL", [/\b[\w.]+@[\w.]+\.\w+\b/])

# Dictionary-based
rec = Candle::GazetteerEntityRecognizer.new("LANG", ["Ruby", "Python", "Rust"])

# Hybrid (combines ML model + patterns + gazetteers)
hybrid = Candle::HybridNER.new("dslim/bert-base-NER", tokenizer: "bert-base-cased")
hybrid.add_pattern_recognizer("EMAIL", [/\b[\w.]+@[\w.]+\.\w+\b/])
hybrid.add_gazetteer_recognizer("LANG", ["Ruby", "Python"])
```

---

## Vision-Language Models (VLM)

### Supported Architectures

| Architecture | Module | Notes |
|:-------------|:-------|:------|
| LLaVA-Next | `llava` + `clip` + `llama` | CLIP vision encoder + MM projector + Llama LLM. Auto-regressive image-to-text generation. |

### Known Working Models

| Model | LLM Backend | Size | Initializer |
|:------|:-----------|:-----|:------------|
| LLaVA-Next Vicuna 7B | Llama (Vicuna) | 13GB | `Candle::VLM.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")` |

### API

```ruby
vlm = Candle::VLM.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
vlm.describe("photo.jpg")                                    # describe image
vlm.ask("photo.jpg", "What color is the car?")               # ask a question
vlm.ask("photo.jpg", "Count the people", max_length: 50)     # limit output
```

---

## Tokenizers

Standalone tokenizer access for any HuggingFace tokenizer:

```ruby
tokenizer = Candle::Tokenizer.from_pretrained("bert-base-uncased")
tokenizer = Candle::Tokenizer.from_file("/path/to/tokenizer.json")
```

All model types (LLM, EmbeddingModel, Reranker, NER) expose their tokenizer via `.tokenizer`.

---

## Device Support

All model types support CPU, Metal (macOS), and CUDA (NVIDIA):

```ruby
model = Candle::LLM.from_pretrained("model-id", device: "cpu")
model = Candle::LLM.from_pretrained("model-id", device: "metal")
model = Candle::LLM.from_pretrained("model-id", device: "cuda")
```

Omit `device:` to auto-select the best available device.
