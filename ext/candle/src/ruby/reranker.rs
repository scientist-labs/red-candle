use magnus::{function, method, prelude::*, Error, RModule, RArray, RHash, Ruby};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::xlm_roberta::{
    XLMRobertaForSequenceClassification, Config as XLMRobertaConfig,
};
use candle_transformers::models::debertav2::{
    DebertaV2Model, DebertaV2ContextPooler, Config as DebertaV2Config,
};
use candle_transformers::models::modernbert::{
    ModernBert, Config as ModernBertConfig,
};
use candle_transformers::models::qwen3::{
    ModelForCausalLM as Qwen3Model, Config as Qwen3Config,
};
use candle_core::{Device as CoreDevice, Tensor, IndexOp, DType};
use candle_nn::{VarBuilder, Linear, Module, ops::sigmoid};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{EncodeInput, Tokenizer};
use std::cell::RefCell;
use crate::ruby::{Device, Result};
use crate::tokenizer::{TokenizerWrapper, loader::TokenizerLoader};

enum RerankerModel {
    Bert {
        model: BertModel,
        pooler: Linear,
        classifier: Linear,
    },
    XLMRoberta {
        model: XLMRobertaForSequenceClassification,
        pad_token_id: u32,
    },
    DeBERTa {
        model: DebertaV2Model,
        pooler: DebertaV2ContextPooler,
        classifier: Linear,
        pad_token_id: u32,
    },
    ModernBert {
        model: ModernBert,
        head_dense: Linear,
        head_norm: candle_nn::LayerNorm,
        classifier: Linear,
        pad_token_id: u32,
    },
    Qwen3 {
        model: RefCell<Qwen3Model>,
        yes_token_id: u32,
        no_token_id: u32,
    },
}

#[magnus::wrap(class = "Candle::Reranker", free_immediately, size)]
pub struct Reranker {
    model: RerankerModel,
    tokenizer: TokenizerWrapper,
    device: CoreDevice,
    model_id: String,
}

impl Reranker {
    pub fn new(model_id: String, device: Option<Device>, max_length: Option<usize>) -> Result<Self> {
        let device = device.unwrap_or(Device::best()).as_device()?;
        let max_length = max_length.unwrap_or(512);  // Default to 512
        Self::new_with_core_device(model_id, device, max_length)
    }

    fn new_with_core_device(model_id: String, device: CoreDevice, max_length: usize) -> std::result::Result<Self, Error> {
        let ruby = Ruby::get().unwrap();
        let runtime_error = ruby.exception_runtime_error();

        let result = (|| -> std::result::Result<(RerankerModel, TokenizerWrapper), Box<dyn std::error::Error + Send + Sync>> {
            let api = Api::new()?;
            let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model));

            // Download model files
            let config_filename = repo.get("config.json")?;
            let tokenizer_filename = repo.get("tokenizer.json")?;
            let weights_filename = repo.get("model.safetensors")?;

            // Read raw config to detect model type
            let config_str = std::fs::read_to_string(&config_filename)?;
            let raw_config: serde_json::Value = serde_json::from_str(&config_str)?;
            let model_type = raw_config["model_type"].as_str().unwrap_or("bert");

            // Setup tokenizer with padding AND truncation
            let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
            let tokenizer = TokenizerLoader::with_padding(tokenizer, None);
            let tokenizer = TokenizerLoader::with_truncation(tokenizer, max_length);

            // Load model weights
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)?
            };

            let model = match model_type {
                "xlm-roberta" => {
                    let config: XLMRobertaConfig = serde_json::from_str(&config_str)?;
                    let pad_token_id = config.pad_token_id;
                    let model = XLMRobertaForSequenceClassification::new(1, &config, vb)?;
                    RerankerModel::XLMRoberta { model, pad_token_id }
                }
                "deberta-v2" => {
                    let config: DebertaV2Config = serde_json::from_str(&config_str)?;
                    let pad_token_id = config.pad_token_id.unwrap_or(0) as u32;
                    let model = DebertaV2Model::load(vb.pp("deberta"), &config)?;
                    let pooler = DebertaV2ContextPooler::load(vb.clone(), &config)?;
                    let pooler_hidden_size = config.pooler_hidden_size.unwrap_or(config.hidden_size);
                    let num_labels = config.id2label.as_ref().map_or(1, |m| m.len());
                    let classifier = candle_nn::linear(pooler_hidden_size, num_labels, vb.pp("classifier"))?;
                    RerankerModel::DeBERTa { model, pooler, classifier, pad_token_id }
                }
                "qwen3" => {
                    let config: Qwen3Config = serde_json::from_str(&config_str)?;
                    let model = Qwen3Model::new(&config, vb)?;

                    // Look up "yes" and "no" token IDs from the tokenizer
                    let yes_token_id: u32 = tokenizer
                        .encode("yes", false)
                        .ok()
                        .and_then(|enc| enc.get_ids().first().copied())
                        .unwrap_or(9693);
                    let no_token_id: u32 = tokenizer
                        .encode("no", false)
                        .ok()
                        .and_then(|enc| enc.get_ids().first().copied())
                        .unwrap_or(2152);

                    RerankerModel::Qwen3 {
                        model: RefCell::new(model),
                        yes_token_id,
                        no_token_id,
                    }
                }
                "modernbert" => {
                    let config: ModernBertConfig = serde_json::from_str(&config_str)?;
                    let pad_token_id = config.pad_token_id;
                    let model = ModernBert::load(vb.clone(), &config)?;
                    // ModernBertHead::load is private, so load the head layers manually
                    let head_vb = vb.pp("head");
                    let head_dense = candle_nn::linear_no_bias(config.hidden_size, config.hidden_size, head_vb.pp("dense"))?;
                    let head_norm = candle_nn::layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, head_vb.pp("norm"))?;
                    let classifier = candle_nn::linear(config.hidden_size, 1, vb.pp("classifier"))?;
                    RerankerModel::ModernBert { model, head_dense, head_norm, classifier, pad_token_id }
                }
                _ => {
                    let config: BertConfig = serde_json::from_str(&config_str)?;
                    let model = BertModel::load(vb.pp("bert"), &config)?;
                    let pooler = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("bert.pooler.dense"))?;
                    let classifier = candle_nn::linear(config.hidden_size, 1, vb.pp("classifier"))?;
                    RerankerModel::Bert { model, pooler, classifier }
                }
            };

            Ok((model, TokenizerWrapper::new(tokenizer)))
        })();

        match result {
            Ok((model, tokenizer)) => {
                Ok(Self { model, tokenizer, device, model_id })
            }
            Err(e) => Err(Error::new(runtime_error, format!("Failed to load model: {}", e))),
        }
    }

    /// Extract CLS embeddings from the model output, handling Metal device workarounds
    fn extract_cls_embeddings(&self, embeddings: &Tensor) -> std::result::Result<Tensor, Error> {
        let ruby = Ruby::get().unwrap();
        let runtime_error = ruby.exception_runtime_error();

        let cls_embeddings = if self.device.is_metal() {
            // Metal has issues with tensor indexing, use a different approach
            let (batch_size, seq_len, hidden_size) = embeddings.dims3()
                .map_err(|e| Error::new(runtime_error, format!("Failed to get dims: {}", e)))?;

            // Reshape to [batch * seq_len, hidden] then take first hidden vectors for each batch
            let reshaped = embeddings.reshape((batch_size * seq_len, hidden_size))
                .map_err(|e| Error::new(runtime_error, format!("Failed to reshape: {}", e)))?;

            // Extract CLS tokens (first token of each sequence)
            let mut cls_vecs = Vec::new();
            for i in 0..batch_size {
                let start_idx = i * seq_len;
                let cls_vec = reshaped.narrow(0, start_idx, 1)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to extract CLS: {}", e)))?;
                cls_vecs.push(cls_vec);
            }

            // Stack the CLS vectors
            Tensor::cat(&cls_vecs, 0)
                .map_err(|e| Error::new(runtime_error, format!("Failed to cat CLS tokens: {}", e)))?
        } else {
            embeddings.i((.., 0))
                .map_err(|e| Error::new(runtime_error, format!("Failed to extract CLS token: {}", e)))?
        };

        // Ensure tensor is contiguous for downstream operations
        cls_embeddings.contiguous()
            .map_err(|e| Error::new(runtime_error, format!("Failed to make CLS embeddings contiguous: {}", e)))
    }

    pub fn debug_tokenization(&self, query: String, document: String) -> std::result::Result<RHash, Error> {
        let ruby = Ruby::get().unwrap();
        let runtime_error = ruby.exception_runtime_error();

        // Create query-document pair for cross-encoder
        let query_doc_pair: EncodeInput = (query.clone(), document.clone()).into();

        // Tokenize using the inner tokenizer for detailed info
        let encoding = self.tokenizer.inner().encode(query_doc_pair, true)
            .map_err(|e| Error::new(runtime_error, format!("Tokenization failed: {}", e)))?;

        // Get token information
        let token_ids = encoding.get_ids().to_vec();
        let token_type_ids = encoding.get_type_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();
        let tokens = encoding.get_tokens().iter().map(|t| t.to_string()).collect::<Vec<_>>();

        // Create result hash
        let result = ruby.hash_new();
        result.aset("token_ids", ruby.ary_from_vec(token_ids.iter().map(|&id| id as i64).collect::<Vec<_>>()))?;
        result.aset("token_type_ids", ruby.ary_from_vec(token_type_ids.iter().map(|&id| id as i64).collect::<Vec<_>>()))?;
        result.aset("attention_mask", ruby.ary_from_vec(attention_mask.iter().map(|&mask| mask as i64).collect::<Vec<_>>()))?;
        result.aset("tokens", ruby.ary_from_vec(tokens))?;

        Ok(result)
    }

    pub fn rerank_with_options(&self, query: String, documents: RArray, pooling_method: String, apply_sigmoid: bool) -> std::result::Result<RArray, Error> {
        let ruby = Ruby::get().unwrap();
        let runtime_error = ruby.exception_runtime_error();
        let documents: Vec<String> = documents.to_vec()?;

        // Create query-document pairs for cross-encoder
        let query_and_docs: Vec<EncodeInput> = documents
            .iter()
            .map(|d| (query.clone(), d.clone()).into())
            .collect();

        // Tokenize batch using inner tokenizer for access to token type IDs
        let encodings = self.tokenizer.inner().encode_batch(query_and_docs, true)
            .map_err(|e| Error::new(runtime_error, format!("Tokenization failed: {}", e)))?;

        // Convert to tensors
        let token_ids = encodings
            .iter()
            .map(|e| e.get_ids().to_vec())
            .collect::<Vec<_>>();

        let token_type_ids = encodings
            .iter()
            .map(|e| e.get_type_ids().to_vec())
            .collect::<Vec<_>>();

        let token_ids = Tensor::new(token_ids, &self.device)
            .map_err(|e| Error::new(runtime_error, format!("Failed to create tensor: {}", e)))?;
        let token_type_ids = Tensor::new(token_type_ids, &self.device)
            .map_err(|e| Error::new(runtime_error, format!("Failed to create token type ids tensor: {}", e)))?;

        // Compute scores based on model type
        let scores = match &self.model {
            RerankerModel::Bert { model, pooler, classifier } => {
                let attention_mask = token_ids.ne(0u32)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to create attention mask: {}", e)))?;

                // Forward pass through BERT
                let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))
                    .map_err(|e| Error::new(runtime_error, format!("Model forward pass failed: {}", e)))?;

                // Apply pooling based on the specified method
                let pooled_embeddings = match pooling_method.as_str() {
                    "pooler" => {
                        let cls_embeddings = self.extract_cls_embeddings(&embeddings)?;
                        let pooled = pooler.forward(&cls_embeddings)
                            .map_err(|e| Error::new(runtime_error, format!("Pooler forward failed: {}", e)))?;
                        pooled.tanh()
                            .map_err(|e| Error::new(runtime_error, format!("Tanh activation failed: {}", e)))?
                    },
                    "cls" => {
                        self.extract_cls_embeddings(&embeddings)?
                    },
                    "mean" => {
                        let (_batch, seq_len, _hidden) = embeddings.dims3()
                            .map_err(|e| Error::new(runtime_error, format!("Failed to get tensor dimensions: {}", e)))?;
                        let sum = embeddings.sum(1)
                            .map_err(|e| Error::new(runtime_error, format!("Failed to sum embeddings: {}", e)))?;
                        (sum / (seq_len as f64))
                            .map_err(|e| Error::new(runtime_error, format!("Failed to compute mean: {}", e)))?
                    },
                    _ => return Err(Error::new(runtime_error,
                        format!("Unknown pooling method: {}. Use 'pooler', 'cls', or 'mean'", pooling_method)))
                };

                let pooled_embeddings = pooled_embeddings.contiguous()
                    .map_err(|e| Error::new(runtime_error, format!("Failed to make pooled_embeddings contiguous: {}", e)))?;
                let logits = classifier.forward(&pooled_embeddings)
                    .map_err(|e| Error::new(runtime_error, format!("Classifier forward failed: {}", e)))?;
                logits.squeeze(1)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to squeeze tensor: {}", e)))?
            }
            RerankerModel::XLMRoberta { model, pad_token_id } => {
                let attention_mask = token_ids.ne(*pad_token_id)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to create attention mask: {}", e)))?;

                // XLMRobertaForSequenceClassification returns logits directly
                let logits = model.forward(&token_ids, &attention_mask, &token_type_ids)
                    .map_err(|e| Error::new(runtime_error, format!("Model forward pass failed: {}", e)))?;
                logits.squeeze(1)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to squeeze tensor: {}", e)))?
            }
            RerankerModel::DeBERTa { model, pooler, classifier, pad_token_id } => {
                let attention_mask = token_ids.ne(*pad_token_id)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to create attention mask: {}", e)))?;

                // Forward through DeBERTa encoder
                let encoder_output = model.forward(&token_ids, Some(token_type_ids.clone()), Some(attention_mask))
                    .map_err(|e| Error::new(runtime_error, format!("Model forward pass failed: {}", e)))?;

                // Pool and classify
                let pooled = pooler.forward(&encoder_output)
                    .map_err(|e| Error::new(runtime_error, format!("Pooler forward failed: {}", e)))?;
                let logits = classifier.forward(&pooled)
                    .map_err(|e| Error::new(runtime_error, format!("Classifier forward failed: {}", e)))?;
                logits.squeeze(1)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to squeeze tensor: {}", e)))?
            }
            RerankerModel::ModernBert { model, head_dense, head_norm, classifier, pad_token_id } => {
                let attention_mask = token_ids.ne(*pad_token_id)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to create attention mask: {}", e)))?;
                let attention_mask_f32 = attention_mask.to_dtype(DType::F32)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to convert attention mask: {}", e)))?;

                // Forward through ModernBERT encoder
                let encoder_output = model.forward(&token_ids, &attention_mask_f32)
                    .map_err(|e| Error::new(runtime_error, format!("Model forward pass failed: {}", e)))?;

                // CLS pooling, then head (dense + GELU + norm) + classifier
                let cls = encoder_output.i((.., 0, ..))
                    .map_err(|e| Error::new(runtime_error, format!("Failed to extract CLS: {}", e)))?
                    .contiguous()
                    .map_err(|e| Error::new(runtime_error, format!("Failed to make contiguous: {}", e)))?;
                let hidden = head_dense.forward(&cls)
                    .map_err(|e| Error::new(runtime_error, format!("Head dense failed: {}", e)))?;
                let hidden = hidden.gelu_erf()
                    .map_err(|e| Error::new(runtime_error, format!("GELU activation failed: {}", e)))?;
                let hidden = head_norm.forward(&hidden)
                    .map_err(|e| Error::new(runtime_error, format!("Head norm failed: {}", e)))?;
                let logits = classifier.forward(&hidden)
                    .map_err(|e| Error::new(runtime_error, format!("Classifier forward failed: {}", e)))?;
                logits.squeeze(1)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to squeeze tensor: {}", e)))?
            }
            RerankerModel::Qwen3 { model, yes_token_id, no_token_id } => {
                // Qwen3 reranker: decoder-based yes/no scoring
                // Process each document individually (causal LM, not batch encoder)
                let mut scores_vec: Vec<f32> = Vec::with_capacity(documents.len());
                let mut model = model.borrow_mut();

                for doc in &documents {
                    // Build the Qwen3 reranker prompt
                    let prompt = format!(
                        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {}\n<Document>: {}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                        query, doc
                    );

                    // Tokenize the prompt
                    let encoding = self.tokenizer.inner().encode(prompt.as_str(), false)
                        .map_err(|e| Error::new(runtime_error, format!("Tokenization failed: {}", e)))?;
                    let input_ids: Vec<u32> = encoding.get_ids().to_vec();

                    // Clear KV cache for each document
                    model.clear_kv_cache();

                    // Forward pass — get logits for the last token position
                    let input_tensor = Tensor::new(&input_ids[..], &self.device)
                        .map_err(|e| Error::new(runtime_error, format!("Failed to create tensor: {}", e)))?
                        .unsqueeze(0)
                        .map_err(|e| Error::new(runtime_error, format!("Failed to unsqueeze: {}", e)))?;

                    let logits = model.forward(&input_tensor, 0)
                        .map_err(|e| Error::new(runtime_error, format!("Model forward pass failed: {}", e)))?;

                    // logits shape: [1, 1, vocab_size] → flatten to [vocab_size]
                    let logits = logits.flatten_all()
                        .map_err(|e| Error::new(runtime_error, format!("Failed to flatten: {}", e)))?
                        .to_dtype(DType::F32)
                        .map_err(|e| Error::new(runtime_error, format!("Failed to convert dtype: {}", e)))?;

                    // Extract yes/no logits and compute score
                    let yes_logit: f32 = logits.i(*yes_token_id as usize)
                        .map_err(|e| Error::new(runtime_error, format!("Failed to get yes logit: {}", e)))?
                        .to_scalar()
                        .map_err(|e| Error::new(runtime_error, format!("Failed to convert yes logit: {}", e)))?;
                    let no_logit: f32 = logits.i(*no_token_id as usize)
                        .map_err(|e| Error::new(runtime_error, format!("Failed to get no logit: {}", e)))?
                        .to_scalar()
                        .map_err(|e| Error::new(runtime_error, format!("Failed to convert no logit: {}", e)))?;

                    // softmax over [yes, no] → P(yes)
                    let max_logit = yes_logit.max(no_logit);
                    let yes_exp = (yes_logit - max_logit).exp();
                    let no_exp = (no_logit - max_logit).exp();
                    let score = yes_exp / (yes_exp + no_exp);

                    scores_vec.push(score);
                }

                // Build scores tensor for uniform handling below
                Tensor::new(scores_vec.as_slice(), &self.device)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to create scores tensor: {}", e)))?
            }
        };

        // Optionally apply sigmoid activation
        let scores = if apply_sigmoid {
            sigmoid(&scores)
                .map_err(|e| Error::new(runtime_error, format!("Sigmoid failed: {}", e)))?
        } else {
            scores
        };

        let scores_vec: Vec<f32> = scores.to_vec1()
            .map_err(|e| Error::new(runtime_error, format!("Failed to convert scores to vec: {}", e)))?;

        // Create tuples with document, score, and original index
        let mut ranked_docs: Vec<(String, f32, usize)> = documents
            .into_iter()
            .zip(scores_vec)
            .enumerate()
            .map(|(idx, (doc, score))| (doc, score, idx))
            .collect();

        // Sort documents by relevance score (descending)
        ranked_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Build result array with [doc, score, doc_id]
        let result_array = ruby.ary_new();
        for (doc, score, doc_id) in ranked_docs {
            let tuple = ruby.ary_new();
            tuple.push(doc)?;
            tuple.push(ruby.float_from_f64(score as f64))?;
            tuple.push(doc_id)?;
            result_array.push(tuple)?;
        }
        Ok(result_array)
    }

    /// Get the tokenizer used by this model
    pub fn tokenizer(&self) -> std::result::Result<crate::ruby::tokenizer::Tokenizer, Error> {
        Ok(crate::ruby::tokenizer::Tokenizer(self.tokenizer.clone()))
    }

    /// Get the model_id
    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    /// Get the device
    pub fn device(&self) -> Device {
        Device::from_device(&self.device)
    }

    /// Get all options as a hash
    pub fn options(&self) -> std::result::Result<RHash, Error> {
        let ruby = Ruby::get().unwrap();
        let hash = ruby.hash_new();
        hash.aset("model_id", self.model_id.clone())?;
        hash.aset("device", self.device().__str__())?;
        Ok(hash)
    }
}

pub fn init(rb_candle: RModule) -> std::result::Result<(), Error> {
    let ruby = Ruby::get().unwrap();
    let c_reranker = rb_candle.define_class("Reranker", ruby.class_object())?;
    c_reranker.define_singleton_method("_create", function!(Reranker::new, 3))?;
    c_reranker.define_method("rerank_with_options", method!(Reranker::rerank_with_options, 4))?;
    c_reranker.define_method("debug_tokenization", method!(Reranker::debug_tokenization, 2))?;
    c_reranker.define_method("tokenizer", method!(Reranker::tokenizer, 0))?;
    c_reranker.define_method("model_id", method!(Reranker::model_id, 0))?;
    c_reranker.define_method("device", method!(Reranker::device, 0))?;
    c_reranker.define_method("options", method!(Reranker::options, 0))?;
    Ok(())
}
