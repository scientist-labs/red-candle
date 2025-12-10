use magnus::{Error, Module, RModule, function, Object};
use std::sync::Arc;

use crate::structured::{SchemaProcessor, VocabularyAdapter, Index, Vocabulary};
use crate::ruby::{Result, tokenizer::Tokenizer};

/// Ruby wrapper for structured generation constraints
#[derive(Clone, Debug)]
#[magnus::wrap(class = "Candle::StructuredConstraint", mark, free_immediately)]
pub struct StructuredConstraint {
    pub(crate) index: Arc<Index>,
}

impl StructuredConstraint {
    /// Create a constraint from a JSON schema using a model ID
    /// This uses Vocabulary::from_pretrained which handles tokenizer byte encoding correctly
    pub fn from_schema_with_model(schema: String, model_id: String) -> Result<Self> {
        // Use tokio runtime for async vocabulary loading
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create runtime: {}", e)))?;

        let vocabulary = rt.block_on(async {
            Vocabulary::from_pretrained(&model_id, None)
        })
        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create vocabulary from model '{}': {:?}", model_id, e)))?;

        let processor = SchemaProcessor::new();
        let index = processor.process_schema(&schema, &vocabulary)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to process schema: {}", e)))?;

        Ok(Self { index })
    }

    /// Create a constraint from a regex pattern using a model ID
    pub fn from_regex_with_model(pattern: String, model_id: String) -> Result<Self> {
        // Use tokio runtime for async vocabulary loading
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create runtime: {}", e)))?;

        let vocabulary = rt.block_on(async {
            Vocabulary::from_pretrained(&model_id, None)
        })
        .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create vocabulary from model '{}': {:?}", model_id, e)))?;

        let processor = SchemaProcessor::new();
        let index = processor.process_regex(&pattern, &vocabulary)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to process regex: {}", e)))?;

        Ok(Self { index })
    }

    /// Create a constraint from a JSON schema (legacy method using tokenizer directly)
    /// Note: This may not handle all tokenizer byte encodings correctly
    pub fn from_schema(schema: String, tokenizer: &Tokenizer) -> Result<Self> {
        let vocabulary = VocabularyAdapter::from_tokenizer(&tokenizer.0)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create vocabulary: {}", e)))?;

        let processor = SchemaProcessor::new();
        let index = processor.process_schema(&schema, &vocabulary)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to process schema: {}", e)))?;

        Ok(Self { index })
    }

    /// Create a constraint from a regex pattern (legacy method using tokenizer directly)
    /// Note: This may not handle all tokenizer byte encodings correctly
    pub fn from_regex(pattern: String, tokenizer: &Tokenizer) -> Result<Self> {
        let vocabulary = VocabularyAdapter::from_tokenizer(&tokenizer.0)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to create vocabulary: {}", e)))?;

        let processor = SchemaProcessor::new();
        let index = processor.process_regex(&pattern, &vocabulary)
            .map_err(|e| Error::new(magnus::exception::runtime_error(), format!("Failed to process regex: {}", e)))?;

        Ok(Self { index })
    }
}

pub fn init_structured(rb_candle: RModule) -> Result<()> {
    let class = rb_candle.define_class("StructuredConstraint", magnus::class::object())?;

    // New methods using model_id for proper vocabulary loading
    class.define_singleton_method("from_schema_with_model", function!(StructuredConstraint::from_schema_with_model, 2))?;
    class.define_singleton_method("from_regex_with_model", function!(StructuredConstraint::from_regex_with_model, 2))?;

    // Legacy methods using tokenizer directly (may have byte encoding issues with some models)
    class.define_singleton_method("from_schema", function!(StructuredConstraint::from_schema, 2))?;
    class.define_singleton_method("from_regex", function!(StructuredConstraint::from_regex, 2))?;

    Ok(())
}