use magnus::{function, method, prelude::*, Error, RModule, Ruby};
use candle_transformers::models::llava::{
    config::{LLaVAConfig, HFLLaVAConfig, HFGenerationConfig, HFPreProcessorConfig},
    LLaVA,
};
use candle_transformers::models::llama::Cache;
use candle_core::{Device as CoreDevice, Tensor, DType};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use crate::ruby::{Device, Result};
use crate::tokenizer::TokenizerWrapper;

const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Vision-Language Model wrapping LLaVA for image understanding.
/// Uses CLIP vision encoder + MM projector + Llama LLM.
///
/// Note: LLaVA contains trait objects (dyn Module) that are !Send,
/// so we wrap it in an UnsafeCell. This is safe because Ruby's GVL
/// ensures single-threaded access to the model.
struct UnsafeSendSync<T>(T);
unsafe impl<T> Send for UnsafeSendSync<T> {}
unsafe impl<T> Sync for UnsafeSendSync<T> {}

#[magnus::wrap(class = "Candle::VLM", free_immediately, size)]
pub struct VLM {
    model: std::cell::RefCell<UnsafeSendSync<LLaVA>>,
    tokenizer: TokenizerWrapper,
    cache: std::cell::RefCell<UnsafeSendSync<Cache>>,
    config: LLaVAConfig,
    device: CoreDevice,
    model_id: String,
    image_size: usize,
    eos_token_id: u32,
}

impl VLM {
    pub fn new(model_id: String, device: Option<Device>) -> Result<Self> {
        let device = device.unwrap_or(Device::best()).as_device()?;
        Self::load_model(model_id, device)
    }

    fn load_model(model_id: String, device: CoreDevice) -> std::result::Result<Self, Error> {
        let ruby = Ruby::get().unwrap();
        let runtime_error = ruby.exception_runtime_error();

        let result = (|| -> std::result::Result<_, Box<dyn std::error::Error + Send + Sync>> {
            let api = Api::new()?;
            let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model));

            // Download config files
            let config_filename = repo.get("config.json")?;
            let gen_config_filename = repo.get("generation_config.json")?;
            let preproc_config_filename = repo.get("preprocessor_config.json")?;
            let tokenizer_filename = repo.get("tokenizer.json")?;

            // Read configs
            let config_str = std::fs::read_to_string(&config_filename)?;
            let gen_config_str = std::fs::read_to_string(&gen_config_filename)?;
            let preproc_config_str = std::fs::read_to_string(&preproc_config_filename)?;

            // Patch config: some models have null pad_token_id in text_config
            // but candle's HFLLaVATextConfig requires usize. Fix by defaulting to 0.
            let mut config_json: serde_json::Value = serde_json::from_str(&config_str)?;
            let top_pad_id = config_json.get("pad_token_id")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            // Patch missing image_grid_pinpoints for LLaVA 1.5
            if config_json.get("image_grid_pinpoints").map_or(true, |v| v.is_null()) {
                config_json["image_grid_pinpoints"] = serde_json::json!([[336, 672], [672, 336], [672, 672]]);
            }
            if let Some(text_config) = config_json.get_mut("text_config") {
                if text_config.get("pad_token_id").map_or(true, |v| v.is_null()) {
                    text_config["pad_token_id"] = serde_json::Value::Number(top_pad_id.into());
                }
            }
            let patched_config_str = serde_json::to_string(&config_json)?;
            let hf_config: HFLLaVAConfig = serde_json::from_str(&patched_config_str)?;
            let gen_config: HFGenerationConfig = serde_json::from_str(&gen_config_str)?;
            let preproc_config: HFPreProcessorConfig = serde_json::from_str(&preproc_config_str)?;

            let image_size = hf_config.vision_config.image_size;
            let eos_token_id = gen_config.eos_token_id as u32;

            let clip_vision_config = hf_config.to_clip_vision_config();
            let config = hf_config.to_llava_config(&gen_config, &preproc_config);

            // Load tokenizer
            let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

            // Download weight files (sharded)
            let weight_files = Self::download_weights(&repo)?;

            // Load model weights
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, &device)?
            };

            // Load LLaVA model with CLIP vision config
            let model = LLaVA::load(vb, &config, Some(clip_vision_config))?;

            // Create KV cache for the Llama LLM
            let llama_config = config.to_llama_config();
            let cache = Cache::new(true, DType::F32, &llama_config, &device)?;

            Ok((model, TokenizerWrapper::new(tokenizer), cache, config, image_size, eos_token_id))
        })();

        match result {
            Ok((model, tokenizer, cache, config, image_size, eos_token_id)) => {
                Ok(Self {
                    model: std::cell::RefCell::new(UnsafeSendSync(model)),
                    tokenizer,
                    cache: std::cell::RefCell::new(UnsafeSendSync(cache)),
                    config,
                    device,
                    model_id,
                    image_size,
                    eos_token_id,
                })
            }
            Err(e) => Err(Error::new(runtime_error, format!("Failed to load VLM: {}", e))),
        }
    }

    fn download_weights(
        repo: &hf_hub::api::sync::ApiRepo,
    ) -> std::result::Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error + Send + Sync>> {
        // Try single file first
        if let Ok(path) = repo.get("model.safetensors") {
            return Ok(vec![path]);
        }

        // Try to get the index file for sharded weights
        let index_path = repo.get("model.safetensors.index.json")?;
        let index_str = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_str)?;

        let weight_map = index["weight_map"].as_object()
            .ok_or("Missing weight_map in index")?;

        let mut filenames: Vec<String> = weight_map.values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        filenames.sort();
        filenames.dedup();

        let mut paths = Vec::new();
        for filename in &filenames {
            let path = repo.get(filename)?;
            paths.push(path);
        }

        Ok(paths)
    }

    /// Load and preprocess an image from a file path into a CLIP-ready tensor
    fn load_image(&self, image_path: &str) -> std::result::Result<Tensor, Error> {
        let ruby = Ruby::get().unwrap();
        let runtime_error = ruby.exception_runtime_error();

        let img = image::open(image_path)
            .map_err(|e| Error::new(runtime_error, format!("Failed to open image: {}", e)))?;

        // Resize to expected size
        let img = img.resize_exact(
            self.image_size as u32,
            self.image_size as u32,
            image::imageops::FilterType::Triangle,
        );

        let img = img.to_rgb8();
        let (width, height) = img.dimensions();
        let h = height as usize;
        let w = width as usize;

        // Convert to CHW format with CLIP normalization
        let mut chw = vec![0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel(x as u32, y as u32);
                chw[0 * h * w + y * w + x] = (p[0] as f32 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0];
                chw[1 * h * w + y * w + x] = (p[1] as f32 / 255.0 - CLIP_MEAN[1]) / CLIP_STD[1];
                chw[2 * h * w + y * w + x] = (p[2] as f32 / 255.0 - CLIP_MEAN[2]) / CLIP_STD[2];
            }
        }

        Tensor::from_vec(chw, (1, 3, h, w), &self.device)
            .map_err(|e| Error::new(runtime_error, format!("Failed to create image tensor: {}", e)))
    }

    /// Describe an image
    pub fn describe(&self, image_path: String, max_length: Option<usize>) -> std::result::Result<String, Error> {
        self.ask(image_path, "Describe this image in detail.".to_string(), max_length)
    }

    /// Ask a question about an image
    pub fn ask(&self, image_path: String, question: String, max_length: Option<usize>) -> std::result::Result<String, Error> {
        let ruby = Ruby::get().unwrap();
        let runtime_error = ruby.exception_runtime_error();
        let max_length = max_length.unwrap_or(256);

        // Load and preprocess image
        let image_tensor = self.load_image(&image_path)?;

        // Build prompt with image token placeholder
        // LLaVA 1.5 HF format: USER: <image>\n{question}\nASSISTANT:
        let prompt = format!("USER: <image>\n{}\nASSISTANT:", question);

        // Tokenize
        let encoding = self.tokenizer.inner().encode(prompt.as_str(), false)
            .map_err(|e| Error::new(runtime_error, format!("Tokenization failed: {}", e)))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();

        // LLaVA expects I64 input IDs
        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        let input_tensor = Tensor::new(&input_ids_i64[..], &self.device)
            .map_err(|e| Error::new(runtime_error, format!("Failed to create input tensor: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| Error::new(runtime_error, format!("Failed to unsqueeze: {}", e)))?;

        let mut model_ref = self.model.borrow_mut();
        let model = &mut model_ref.0;
        let mut cache_ref = self.cache.borrow_mut();
        let cache = &mut cache_ref.0;

        // Prepare multimodal input: merge image features with text embeddings
        let image_size = (self.image_size as u32, self.image_size as u32);
        let input_embeds = model.prepare_inputs_labels_for_multimodal(
            &input_tensor,
            &[image_tensor],
            &[image_size],
        ).map_err(|e| Error::new(runtime_error, format!("Failed to prepare multimodal input: {}", e)))?;

        // Generate tokens autoregressively
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut current_embeds = input_embeds;

        for i in 0..max_length {
            let logits = model.forward(&current_embeds, i, cache)
                .map_err(|e| Error::new(runtime_error, format!("Forward pass failed at step {}: {}", i, e)))?;

            // Get logits for last position
            let logits = logits.flatten_all()
                .map_err(|e| Error::new(runtime_error, format!("Failed to flatten: {}", e)))?;

            // Handle multi-dim logits (take last token if needed)
            let vocab_size = self.config.vocab_size;
            let logits = if logits.elem_count() > vocab_size {
                let n_tokens = logits.elem_count() / vocab_size;
                logits.reshape((n_tokens, vocab_size))
                    .map_err(|e| Error::new(runtime_error, format!("Failed to reshape logits: {}", e)))?
                    .narrow(0, n_tokens - 1, 1)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to narrow logits: {}", e)))?
                    .squeeze(0)
                    .map_err(|e| Error::new(runtime_error, format!("Failed to squeeze logits: {}", e)))?
            } else {
                logits
            };

            let logits = logits.to_dtype(DType::F32)
                .map_err(|e| Error::new(runtime_error, format!("Failed to convert dtype: {}", e)))?;

            // Greedy decoding
            let next_token = logits.argmax(0)
                .map_err(|e| Error::new(runtime_error, format!("Argmax failed: {}", e)))?
                .to_scalar::<u32>()
                .map_err(|e| Error::new(runtime_error, format!("Failed to get token: {}", e)))?;

            // Check for EOS
            if next_token == self.eos_token_id {
                break;
            }

            generated_tokens.push(next_token);

            // For subsequent tokens, embed directly through Llama's embedding layer
            let next_input = Tensor::new(&[next_token as i64], &self.device)
                .map_err(|e| Error::new(runtime_error, format!("Failed to create next input: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| Error::new(runtime_error, format!("Failed to unsqueeze next: {}", e)))?;

            current_embeds = model.llama.embed(&next_input)
                .map_err(|e| Error::new(runtime_error, format!("Failed to embed next token: {}", e)))?;
        }

        // Decode generated tokens
        let text = self.tokenizer.inner().decode(&generated_tokens, true)
            .map_err(|e| Error::new(runtime_error, format!("Decoding failed: {}", e)))?;

        Ok(text.trim().to_string())
    }

    pub fn model_id(&self) -> String {
        self.model_id.clone()
    }

    pub fn device(&self) -> Device {
        Device::from_device(&self.device)
    }

    pub fn tokenizer(&self) -> std::result::Result<crate::ruby::tokenizer::Tokenizer, Error> {
        Ok(crate::ruby::tokenizer::Tokenizer(self.tokenizer.clone()))
    }
}

pub fn init(rb_candle: RModule) -> std::result::Result<(), Error> {
    let ruby = Ruby::get().unwrap();
    let c_vlm = rb_candle.define_class("VLM", ruby.class_object())?;
    c_vlm.define_singleton_method("_create", function!(VLM::new, 2))?;
    c_vlm.define_method("describe", method!(VLM::describe, 2))?;
    c_vlm.define_method("ask", method!(VLM::ask, 3))?;
    c_vlm.define_method("model_id", method!(VLM::model_id, 0))?;
    c_vlm.define_method("device", method!(VLM::device, 0))?;
    c_vlm.define_method("tokenizer", method!(VLM::tokenizer, 0))?;
    Ok(())
}
