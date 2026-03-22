use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::granitemoehybrid::{
    GraniteMoeHybrid as GraniteMoeHybridModel, GraniteMoeHybridCache,
    GraniteMoeHybridConfig, GraniteMoeHybridInternalConfig,
};
use hf_hub::{api::tokio::Api, Repo};
use tokenizers::Tokenizer;

use super::{GenerationConfig, TextGeneration, TextGenerator, TokenizerWrapper};

#[derive(Debug)]
pub struct GraniteMoeHybrid {
    model: GraniteMoeHybridModel,
    tokenizer: TokenizerWrapper,
    device: Device,
    model_id: String,
    eos_token_id: u32,
    cache: GraniteMoeHybridCache,
    config: GraniteMoeHybridInternalConfig,
}

impl GraniteMoeHybrid {
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn clear_kv_cache(&mut self) {
        if let Ok(new_cache) =
            GraniteMoeHybridCache::new(self.cache.use_kv_cache, DType::F32, &self.config, &self.device)
        {
            self.cache = new_cache;
        }
    }

    pub fn tokenizer(&self) -> &TokenizerWrapper {
        &self.tokenizer
    }

    pub async fn from_pretrained_with_tokenizer(
        model_id: &str,
        device: Device,
        tokenizer_source: Option<&str>,
    ) -> CandleResult<Self> {
        let api = Api::new()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create HF API: {}", e)))?;

        let repo = api.repo(Repo::model(model_id.to_string()));

        let config_filename = repo
            .get("config.json")
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download config: {}", e)))?;

        let config_str = std::fs::read_to_string(config_filename)?;
        let granite_config: GraniteMoeHybridConfig = serde_json::from_str(&config_str)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;
        let config = granite_config.into_config(false);

        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config JSON: {}", e)))?;
        let tie_word_embeddings = config_json
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let tokenizer = if let Some(tokenizer_id) = tokenizer_source {
            let tokenizer_repo = api.repo(Repo::model(tokenizer_id.to_string()));
            let tokenizer_filename = tokenizer_repo
                .get("tokenizer.json")
                .await
                .map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "Failed to download tokenizer from {}: {}",
                        tokenizer_id, e
                    ))
                })?;
            Tokenizer::from_file(tokenizer_filename)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?
        } else {
            let tokenizer_filename = repo.get("tokenizer.json").await.map_err(|e| {
                candle_core::Error::Msg(format!("Failed to download tokenizer: {}", e))
            })?;
            Tokenizer::from_file(tokenizer_filename)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?
        };

        let vocab = tokenizer.get_vocab(true);
        let eos_token_id = vocab
            .get("<|end_of_text|>")
            .or_else(|| vocab.get("<|endoftext|>"))
            .or_else(|| vocab.get("</s>"))
            .copied()
            .unwrap_or(0);

        let weights_filenames = if let Ok(single_file) = repo.get("model.safetensors").await {
            vec![single_file]
        } else {
            let mut sharded_files = Vec::new();
            let mut index = 1;
            loop {
                let mut found = false;
                for total in [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30] {
                    let filename =
                        format!("model-{:05}-of-{:05}.safetensors", index, total);
                    if let Ok(file) = repo.get(&filename).await {
                        sharded_files.push(file);
                        found = true;
                        break;
                    }
                }
                if !found {
                    break;
                }
                index += 1;
            }

            if sharded_files.is_empty() {
                return Err(candle_core::Error::Msg(
                    "Could not find model weights. Tried: model.safetensors, model-*-of-*.safetensors".to_string(),
                ));
            }
            sharded_files
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights_filenames, DType::F32, &device)?
        };

        let vb = if tie_word_embeddings {
            vb.rename_f(|name: &str| {
                if name == "lm_head.weight" {
                    "model.embed_tokens.weight".to_string()
                } else {
                    name.to_string()
                }
            })
        } else {
            vb
        };

        let model = GraniteMoeHybridModel::load(vb, &config)?;
        let cache = GraniteMoeHybridCache::new(true, DType::F32, &config, &device)?;

        Ok(Self {
            model,
            tokenizer: TokenizerWrapper::new(tokenizer),
            device,
            model_id: model_id.to_string(),
            eos_token_id,
            cache,
            config,
        })
    }

    pub async fn from_pretrained(model_id: &str, device: Device) -> CandleResult<Self> {
        Self::from_pretrained_with_tokenizer(model_id, device, None).await
    }

    pub fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
    ) -> CandleResult<String> {
        let mut prompt = String::new();

        for message in messages {
            let role = message["role"].as_str().unwrap_or("");
            let content = message["content"].as_str().unwrap_or("");

            match role {
                "system" => {
                    prompt.push_str(&format!(
                        "<|start_of_role|>system<|end_of_role|>{}<|end_of_text|>\n",
                        content
                    ));
                }
                "user" => {
                    prompt.push_str(&format!(
                        "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n",
                        content
                    ));
                }
                "assistant" => {
                    prompt.push_str(&format!(
                        "<|start_of_role|>assistant<|end_of_role|>{}<|end_of_text|>\n",
                        content
                    ));
                }
                "tool" => {
                    prompt.push_str(&format!(
                        "<|start_of_role|>tool<|end_of_role|>{}<|end_of_text|>\n",
                        content
                    ));
                }
                _ => {}
            }
        }

        prompt.push_str("<|start_of_role|>assistant<|end_of_role|>");

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
            let input = input.contiguous()?;
            let logits = self.model.forward(&input, start_pos, &mut self.cache)?;

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
                    let decoded_text =
                        self.tokenizer
                            .decode_incremental(&all_tokens, all_tokens.len() - 1)?;
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

impl TextGenerator for GraniteMoeHybrid {
    fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> CandleResult<String> {
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
