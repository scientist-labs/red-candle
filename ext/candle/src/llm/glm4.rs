use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::models::glm4_new::{Config, ModelForCausalLM as Glm4Model};
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;

use crate::llm::{GenerationConfig, TextGeneration, TextGenerator, TokenizerWrapper};

#[derive(Debug)]
pub struct Glm4 {
    model: Glm4Model,
    tokenizer: TokenizerWrapper,
    device: Device,
    model_id: String,
    eos_token_id: u32,
}

impl Glm4 {
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn tokenizer(&self) -> &TokenizerWrapper {
        &self.tokenizer
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    pub async fn from_pretrained_with_tokenizer(model_id: &str, device: Device, tokenizer_source: Option<&str>) -> CandleResult<Self> {
        let api = Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;

        let repo = api.model(model_id.to_string());

        let config_filename = repo.get("config.json").await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download config: {}", e)))?;
        let config_str = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;

        let tokenizer = if let Some(tokenizer_id) = tokenizer_source {
            let tokenizer_repo = api.model(tokenizer_id.to_string());
            let tokenizer_filename = tokenizer_repo.get("tokenizer.json").await
                .map_err(|e| candle_core::Error::Msg(format!("Failed to download tokenizer from {}: {}", tokenizer_id, e)))?;
            Tokenizer::from_file(tokenizer_filename)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?
        } else {
            let tokenizer_filename = repo.get("tokenizer.json").await
                .map_err(|e| candle_core::Error::Msg(format!("Failed to download tokenizer: {}", e)))?;
            Tokenizer::from_file(tokenizer_filename)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?
        };

        let vocab = tokenizer.get_vocab(true);
        let eos_token_id = vocab.get("<|endoftext|>")
            .or_else(|| vocab.get("<|user|>"))
            .or_else(|| vocab.get("</s>"))
            .copied()
            .unwrap_or(151329);

        let mut filenames = vec![];
        let num_shards = if model_id.contains("9b") || model_id.contains("9B") { 4 } else { 1 };

        if num_shards == 1 {
            let filename = repo.get("model.safetensors").await
                .map_err(|e| candle_core::Error::Msg(format!("Failed to download model weights: {}", e)))?;
            filenames.push(filename);
        } else {
            for shard_idx in 1..=num_shards {
                let filename = repo.get(&format!("model-{:05}-of-{:05}.safetensors", shard_idx, num_shards)).await
                    .map_err(|e| candle_core::Error::Msg(format!("Failed to download shard {}: {}", shard_idx, e)))?;
                filenames.push(filename);
            }
        }

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)?
        };

        let model = Glm4Model::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer: TokenizerWrapper::new(tokenizer),
            device,
            model_id: model_id.to_string(),
            eos_token_id,
        })
    }

    pub async fn from_pretrained(model_id: &str, device: Device) -> CandleResult<Self> {
        Self::from_pretrained_with_tokenizer(model_id, device, None).await
    }

    pub fn apply_chat_template(&self, messages: &[serde_json::Value]) -> CandleResult<String> {
        let mut prompt = String::new();

        prompt.push_str("[gMASK]<sop>");

        for message in messages {
            let role = message["role"].as_str().unwrap_or("");
            let content = message["content"].as_str().unwrap_or("");

            match role {
                "system" => {
                    prompt.push_str(&format!("<|system|>\n{}", content));
                }
                "user" => {
                    prompt.push_str(&format!("<|user|>\n{}", content));
                }
                "assistant" => {
                    prompt.push_str(&format!("<|assistant|>\n{}", content));
                }
                _ => {}
            }
        }

        prompt.push_str("<|assistant|>\n");

        Ok(prompt)
    }

    fn generate_tokens(
        &mut self,
        prompt_tokens: Vec<u32>,
        config: &GenerationConfig,
        mut callback: Option<impl FnMut(&str)>,
    ) -> CandleResult<Vec<u32>> {
        let mut text_gen = TextGeneration::new(config);
        text_gen.set_eos_token_id(self.eos_token_id);
        text_gen.set_tokens(prompt_tokens.clone());

        let mut all_tokens = prompt_tokens.clone();
        let start_gen = all_tokens.len();

        for index in 0..config.max_length {
            let context_size = if index > 0 { 1 } else { all_tokens.len() };
            let start_pos = all_tokens.len().saturating_sub(context_size);
            let ctxt = &all_tokens[start_pos..];

            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?;

            let logits = if logits.dims().len() == 2 {
                let seq_len = logits.dim(0)?;
                logits.narrow(0, seq_len - 1, 1)?.squeeze(0)?
            } else {
                logits
            };

            let logits = logits.to_dtype(DType::F32)?;

            let next_token = text_gen.sample_next_token(&logits)?;

            all_tokens.push(next_token);

            if let Some(ref mut cb) = callback {
                if config.debug_tokens {
                    let token_piece = self.tokenizer.token_to_piece(next_token)?;
                    cb(&format!("[{}:{}]", next_token, token_piece));
                } else {
                    let decoded_text = self.tokenizer.decode_incremental(&all_tokens, all_tokens.len() - 1)?;
                    cb(&decoded_text);
                }
            }

            if text_gen.should_stop(next_token, config.max_length) {
                break;
            }

            if config.stop_on_constraint_satisfaction {
                let satisfied = if config.stop_on_match {
                    text_gen.is_constraint_satisfied_stop_on_match()
                } else {
                    text_gen.is_constraint_satisfied()
                };
                if satisfied {
                    break;
                }
            }

            let generated_text = self.tokenizer.decode(&all_tokens[start_gen..], true)?;
            if text_gen.check_stop_sequences(&generated_text, &config.stop_sequences) {
                break;
            }
        }

        Ok(if config.include_prompt {
            all_tokens
        } else {
            all_tokens[start_gen..].to_vec()
        })
    }
}

impl TextGenerator for Glm4 {
    fn generate(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> CandleResult<String> {
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        let output_tokens = self.generate_tokens(prompt_tokens, config, None::<fn(&str)>)?;

        if config.debug_tokens {
            self.tokenizer.format_tokens_with_debug(&output_tokens)
        } else {
            self.tokenizer.decode(&output_tokens, true)
        }
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        mut callback: impl FnMut(&str),
    ) -> CandleResult<String> {
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        let output_tokens = self.generate_tokens(prompt_tokens, config, Some(&mut callback))?;
        self.tokenizer.decode(&output_tokens, true)
    }

    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn clear_cache(&mut self) {
        self.clear_kv_cache();
    }
}
