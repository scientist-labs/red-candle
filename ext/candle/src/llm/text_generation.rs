use candle_core::{Result as CandleResult, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::sync::Arc;

use super::GenerationConfig;
use crate::structured::Index;

/// Helper struct for text generation process
pub struct TextGeneration {
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    eos_token_id: Option<u32>,
    repetition_penalty: f32,
    repetition_penalty_last_n: usize,
    constraint: Option<Arc<Index>>,
    constraint_state: Option<u32>,
    constraint_completed: bool,
    tokens_since_constraint_start: usize,
}

impl TextGeneration {
    pub fn new(config: &GenerationConfig) -> Self {
        let logits_processor = LogitsProcessor::new(config.seed, Some(config.temperature), config.top_p);
        
        let mut text_gen = Self {
            logits_processor,
            tokens: Vec::new(),
            eos_token_id: None,
            repetition_penalty: config.repetition_penalty,
            repetition_penalty_last_n: config.repetition_penalty_last_n,
            constraint: None,
            constraint_state: None,
            constraint_completed: false,
            tokens_since_constraint_start: 0,
        };
        
        // Set constraint if provided
        if let Some(ref constraint) = config.constraint {
            text_gen.set_constraint(Arc::clone(constraint));
        }
        
        text_gen
    }

    pub fn set_eos_token_id(&mut self, eos_token_id: u32) {
        self.eos_token_id = Some(eos_token_id);
    }

    pub fn set_tokens(&mut self, tokens: Vec<u32>) {
        self.tokens = tokens;
    }

    pub fn get_tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn push_token(&mut self, token: u32) {
        self.tokens.push(token);
    }

    pub fn set_constraint(&mut self, constraint: Arc<Index>) {
        // Initialize with the first state
        self.constraint_state = Some(constraint.initial_state());
        self.constraint = Some(constraint);
        self.constraint_completed = false;
        self.tokens_since_constraint_start = self.tokens.len();
    }

    /// Apply constraints to logits by masking disallowed tokens
    fn apply_constraints(&self, logits: &mut Tensor) -> CandleResult<()> {
        if let (Some(ref constraint_index), Some(state)) = (&self.constraint, self.constraint_state) {
            let device = logits.device();
            let vocab_size = logits.dims1()?;
            
            // Get allowed tokens from the constraint index for current state
            if let Some(allowed_tokens) = constraint_index.allowed_tokens(&state) {
                // Create a mask where allowed tokens have value 0 and others have -inf
                let mut mask = vec![f32::NEG_INFINITY; vocab_size];
                for &token_id in &allowed_tokens {
                    if (token_id as usize) < vocab_size {
                        mask[token_id as usize] = 0.0;
                    }
                }
                
                // Apply mask to logits
                let mask_tensor = Tensor::from_vec(mask, vocab_size, device)?;
                *logits = logits.add(&mask_tensor)?;
            }
        }
        Ok(())
    }

    /// Apply repetition penalty to logits
    pub fn apply_repetition_penalty(
        &self,
        logits: &mut Tensor,
        penalty: f32,
        context_size: usize,
    ) -> CandleResult<()> {
        if penalty == 1.0 {
            return Ok(());
        }

        let device = logits.device();
        let vocab_size = logits.dims1()?;
        
        // Get the context tokens to apply penalty to
        let start = self.tokens.len().saturating_sub(context_size);
        let context_tokens = &self.tokens[start..];
        
        // Apply penalty to tokens that appear in the context
        let mut logits_vec = logits.to_vec1::<f32>()?;
        for &token in context_tokens {
            if (token as usize) < vocab_size {
                let idx = token as usize;
                if logits_vec[idx] > 0.0 {
                    logits_vec[idx] /= penalty;
                } else {
                    logits_vec[idx] *= penalty;
                }
            }
        }
        
        *logits = Tensor::from_vec(logits_vec, vocab_size, device)?;
        Ok(())
    }

    /// Sample next token from logits
    pub fn sample_next_token(
        &mut self,
        logits: &Tensor,
    ) -> CandleResult<u32> {
        let mut logits = logits.clone();
        
        // Apply repetition penalty using stored parameters
        if self.repetition_penalty != 1.0 {
            self.apply_repetition_penalty(&mut logits, self.repetition_penalty, self.repetition_penalty_last_n)?;
        }
        
        // Apply constraints if active
        self.apply_constraints(&mut logits)?;
        
        // Sample token
        let next_token = self.logits_processor.sample(&logits)?;
        self.tokens.push(next_token);
        
        // Update constraint state if active
        if let (Some(ref constraint_index), Some(current_state)) = (&self.constraint, self.constraint_state) {
            // Get the next state
            let next_state = constraint_index.next_state(&current_state, &next_token);

            // Check if we're transitioning to a state with no allowed tokens (completion)
            if !self.constraint_completed && self.tokens.len() > self.tokens_since_constraint_start {
                // Check if next state has no allowed tokens at all - this is definitive completion
                if let Some(next_state_val) = next_state {
                    if let Some(allowed) = constraint_index.allowed_tokens(&next_state_val) {
                        if allowed.is_empty() {
                            self.constraint_completed = true;
                        }
                        // Only mark as complete if ONLY EOS is allowed (not just if EOS is one of many options)
                        else if let Some(eos) = self.eos_token_id {
                            if allowed.len() == 1 && allowed.contains(&eos) {
                                self.constraint_completed = true;
                            }
                        }
                    } else {
                        // None means no tokens allowed - constraint is complete
                        self.constraint_completed = true;
                    }
                }
            }

            self.constraint_state = next_state;
        }
        
        Ok(next_token)
    }

    /// Check if the constraint is satisfied (reached a valid completion state)
    pub fn is_constraint_satisfied(&self) -> bool {
        // If we've explicitly marked the constraint as completed, return true
        if self.constraint_completed {
            return true;
        }

        // Also check the current state
        if let (Some(ref constraint_index), Some(state)) = (&self.constraint, self.constraint_state) {
            // Check if the constraint has reached a state where it MUST end
            // This happens when there are no more allowed tokens (constraint fully satisfied)
            if let Some(allowed) = constraint_index.allowed_tokens(&state) {
                // If no tokens are allowed, the constraint is fully satisfied
                if allowed.is_empty() {
                    return true;
                }

                // For JSON schemas, check if ONLY the EOS token is allowed
                // This means we've generated a complete, valid JSON structure
                // Don't treat EOS as a satisfaction signal if other tokens are also allowed
                if let Some(eos) = self.eos_token_id {
                    if allowed.len() == 1 && allowed.contains(&eos) {
                        return true;
                    }
                }
            } else {
                // None means no tokens allowed - constraint is satisfied
                return true;
            }
        }
        false
    }
    
    /// Check if the constraint is satisfied when stop_on_match is true
    /// NOTE: For JSON schemas, this should only return true when the JSON structure is complete,
    /// not just because we're in a state with many allowed tokens (like inside a string).
    pub fn is_constraint_satisfied_stop_on_match(&self) -> bool {
        // When stop_on_match is true, we stop as soon as the constraint is completed
        if self.constraint_completed {
            return true;
        }

        // For JSON and other structured outputs, don't use the "large allowed set" heuristic.
        // Instead, only consider the constraint satisfied when:
        // 1. There are no allowed tokens (definitive completion)
        // 2. Only EOS is allowed (completion with optional termination)
        if let (Some(ref constraint_index), Some(state)) = (&self.constraint, self.constraint_state) {
            if let Some(allowed) = constraint_index.allowed_tokens(&state) {
                // No more tokens allowed - definitely complete
                if allowed.is_empty() {
                    return true;
                }

                // Only EOS is allowed - complete JSON structure
                if let Some(eos) = self.eos_token_id {
                    if allowed.len() == 1 && allowed.contains(&eos) {
                        return true;
                    }
                }
            } else {
                // None means no tokens allowed - constraint is complete
                return true;
            }
        }

        false
    }

    /// Check if we should stop generation
    pub fn should_stop(&self, token: u32, max_length: usize) -> bool {
        if self.tokens.len() >= max_length {
            return true;
        }

        if let Some(eos) = self.eos_token_id {
            if token == eos {
                return true;
            }
        }

        // Check if we've reached a final state in constraint
        // A state is considered final if it has no allowed tokens
        if let (Some(ref constraint_index), Some(state)) = (&self.constraint, self.constraint_state) {
            if let Some(allowed) = constraint_index.allowed_tokens(&state) {
                if allowed.is_empty() {
                    return true;
                }
            } else {
                // None means no tokens allowed - we're done
                return true;
            }
        }

        false
    }

    /// Check if the generated text ends with any stop sequence
    pub fn check_stop_sequences(&self, text: &str, stop_sequences: &[String]) -> bool {
        for seq in stop_sequences {
            if text.ends_with(seq) {
                return true;
            }
        }
        false
    }
}