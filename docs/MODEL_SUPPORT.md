# Model Support in red-candle

## Known Working Models

| Model | Model Path | GGUF File | Tokenizer | Initializer |
| :----- | :---- | :---- | :---- | :---- |
| Mistral v0.2 | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` | `mistral-7b-instruct-v0.2.Q4_K_M.gguf` | | `llm = Candle::LLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", gguf_file: "mistral-7b-instruct-v0.2.Q4_K_M.gguf")` |
| Mistral v0.3 | `MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF` | `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` | `mistralai/Mistral-7B-Instruct-v0.3` | `llm = Candle::LLM.from_pretrained("MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF", gguf_file: "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", tokenizer: "mistralai/Mistral-7B-Instruct-v0.3")` |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | | | `llm = Candle::LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")` |
| TinyLlama (GGUF) | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` | | `llm = Candle::LLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")` |
| Gemma 3 | `google/gemma-3-4b-it-qat-q4_0-gguf` | `gemma-3-4b-it-q4_0.gguf` | `google/gemma-3-4b-it` | `llm = Candle::LLM.from_pretrained("google/gemma-3-4b-it-qat-q4_0-gguf", gguf_file: "gemma-3-4b-it-q4_0.gguf", tokenizer: "google/gemma-3-4b-it")` |
| Qwen-2.5 (GGUF) | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | | `llm = Candle::LLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct-GGUF", gguf_file: "qwen2.5-1.5b-instruct-q4_k_m.gguf")` |
| Qwen3 | `Qwen/Qwen3-0.6B` | | | `llm = Candle::LLM.from_pretrained("Qwen/Qwen3-0.6B")` |
| Qwen3 | `MaziyarPanahi/Qwen3-0.6B-GGUF` | `Qwen3-0.6B.Q4_K_M.gguf` | `Qwen/Qwen3-0.6B` | `llm = Candle::LLM.from_pretrained("MaziyarPanahi/Qwen3-0.6B-GGUF", gguf_file: "Qwen3-0.6B.Q4_K_M.gguf", tokenizer: "Qwen/Qwen3-0.6B")` |
| Qwen3 | `MaziyarPanahi/Qwen3-1.7B-GGUF` | `Qwen3-1.7B.Q4_K_M.gguf` | `Qwen/Qwen3-1.7B` | `llm = Candle::LLM.from_pretrained("MaziyarPanahi/Qwen3-1.7B-GGUF", gguf_file: "Qwen3-1.7B.Q4_K_M.gguf", tokenizer: "Qwen/Qwen3-1.7B")` |
| SmolLM2 360M | `HuggingFaceTB/SmolLM2-360M-Instruct` | | | `llm = Candle::LLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")` |
| SmolLM2 360M (GGUF) | `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF` | `smollm2-360m-instruct-q8_0.gguf` | | `llm = Candle::LLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", gguf_file: "smollm2-360m-instruct-q8_0.gguf")` |
| Phi-2 | `microsoft/phi-2` | | | `llm = Candle::LLM.from_pretrained("microsoft/phi-2")` |
| Phi-2 | `TheBloke/phi-2-GGUF` | `phi-2.Q4_K_M.gguf` | | `llm = Candle::LLM.from_pretrained("TheBloke/phi-2-GGUF", gguf_file: "phi-2.Q4_K_M.gguf")` |
| Phi-3 | `microsoft/Phi-3-mini-4k-instruct` | | | `llm = Candle::LLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")` |
| Phi-3 | `microsoft/Phi-3-mini-128k-instruct` | | | `llm = Candle::LLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct")` |
| Yi-1.5 | `01-ai/Yi-1.5-6B-Chat` | | | `llm = Candle::LLM.from_pretrained("01-ai/Yi-1.5-6B-Chat")` |
| Yi-1.5 (GGUF) | `bartowski/Yi-1.5-6B-Chat-GGUF` | `Yi-1.5-6B-Chat-Q4_K_M.gguf` | `01-ai/Yi-1.5-6B-Chat` | `llm = Candle::LLM.from_pretrained("bartowski/Yi-1.5-6B-Chat-GGUF", gguf_file: "Yi-1.5-6B-Chat-Q4_K_M.gguf", tokenizer: "01-ai/Yi-1.5-6B-Chat")` |
| GLM-4 | `THUDM/GLM-4-9B-0414` | | | `llm = Candle::LLM.from_pretrained("THUDM/GLM-4-9B-0414")` |
| GLM-4 (GGUF) | `bartowski/THUDM_GLM-4-9B-0414-GGUF` | `THUDM_GLM-4-9B-0414-Q4_K_M.gguf` | `THUDM/GLM-4-9B-0414` | `llm = Candle::LLM.from_pretrained("bartowski/THUDM_GLM-4-9B-0414-GGUF", gguf_file: "THUDM_GLM-4-9B-0414-Q4_K_M.gguf", tokenizer: "THUDM/GLM-4-9B-0414")` |
| Granite 7B | `ibm-granite/granite-7b-instruct` | | | `llm = Candle::LLM.from_pretrained("ibm-granite/granite-7b-instruct")` |
| Phi-3 (GGUF) | `microsoft/Phi-3-mini-4k-instruct-gguf` | `Phi-3-mini-4k-instruct-q4.gguf` | | `llm = Candle::LLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct-gguf", gguf_file: "Phi-3-mini-4k-instruct-q4.gguf")` |
| Phi-4 (GGUF) | `microsoft/phi-4-gguf` | `phi-4-Q4_K_S.gguf` | | `llm = Candle::LLM.from_pretrained("microsoft/phi-4-gguf", gguf_file: "phi-4-Q4_K_S.gguf")` |

## ⚠️ Partially Working Models

| Model | Model Path | GGUF File | Tokenizer | Initializer | Status |
| :----- | :---- | :---- | :---- | :---- | :---- |
| GLM-4 (older) | `bartowski/glm-4-9b-chat-1m-GGUF` | `glm-4-9b-chat-1m-Q4_K_M.gguf` | `THUDM/glm-4-9b-chat` | `llm = Candle::LLM.from_pretrained("bartowski/glm-4-9b-chat-1m-GGUF", gguf_file: "glm-4-9b-chat-1m-Q4_K_M.gguf", tokenizer: "THUDM/glm-4-9b-chat")` | Older GLM-4 only ships `tokenizer.model` (tiktoken), not `tokenizer.json`. Use GLM-4-9B-0414 instead. |
| Granite 3.3 2B | `ibm-granite/granite-3.3-2b-instruct` | | | `llm = Candle::LLM.from_pretrained("ibm-granite/granite-3.3-2b-instruct")` | Produces garbage output. Upstream candle-transformers 0.9.2 `granite.rs` does not implement Granite 3.x scaling multipliers (`embedding_multiplier`, `attention_multiplier`, `residual_multiplier`, `logits_scaling`). The `granitemoehybrid.rs` has these but targets Granite 4.0, not 3.x. |
| Qwen-2.5 (safetensors) | `Qwen/Qwen2.5-0.5B-Instruct` | | | `llm = Candle::LLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")` | Safetensors loading produces garbage output, upstream candle bug [#2295](https://github.com/huggingface/candle/issues/2295). GGUF version works fine. |
| Mistral-Nemo | `MaziyarPanahi/Mistral-Nemo-Instruct-2407-GGUF` | `Mistral-Nemo-Instruct-2407.Q4_K_M.gguf` | `mistralai/Mistral-Nemo-Instruct-2407` | `llm = Candle::LLM.from_pretrained("MaziyarPanahi/Mistral-Nemo-Instruct-2407-GGUF", gguf_file: "Mistral-Nemo-Instruct-2407.Q4_K_M.gguf", tokenizer: "mistralai/Mistral-Nemo-Instruct-2407")` | Shape mismatch in GGUF loader (different GQA head configuration). |
| Ministral-8B | `mistralai/Ministral-8B-Instruct-2410` | | | `llm = Candle::LLM.from_pretrained("mistralai/Ministral-8B-Instruct-2410")` | Cannot find embed_tokens weight (sharded safetensors layout issue). |
