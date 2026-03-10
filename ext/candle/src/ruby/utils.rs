use magnus::{function, Module, Object};

use ::candle_core::Tensor as CoreTensor;

use crate::ruby::Result;

/// Ensures the HuggingFace cache directory exists before Api::new() is called.
///
/// The hf_hub crate stores downloaded models in a "hub" subdirectory under the
/// cache root. When the parent directory doesn't exist, hf_hub may fail to
/// create the full path or silently produce an empty cache. This function
/// pre-creates the directory tree to avoid the race condition described in
/// issue #72.
///
/// Resolution order for the cache root:
///   1. $HF_HOME (if set)
///   2. $XDG_CACHE_HOME/huggingface (if XDG_CACHE_HOME is set)
///   3. ~/.cache/huggingface
pub fn ensure_hf_cache_dir() {
    let cache_root = if let Ok(hf_home) = std::env::var("HF_HOME") {
        std::path::PathBuf::from(hf_home)
    } else if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        std::path::PathBuf::from(xdg).join("huggingface")
    } else if let Ok(home) = std::env::var("HOME") {
        std::path::PathBuf::from(home).join(".cache").join("huggingface")
    } else {
        return;
    };
    let hub_dir = cache_root.join("hub");
    let _ = std::fs::create_dir_all(hub_dir);
}

pub fn actual_index(t: &CoreTensor, dim: usize, index: i64) -> candle_core::Result<usize> {
    let dim = t.dim(dim)?;
    if 0 <= index {
        let index = index as usize;
        if dim <= index {
            candle_core::bail!("index {index} is too large for tensor dimension {dim}")
        }
        Ok(index)
    } else {
        if (dim as i64) < -index {
            candle_core::bail!("index {index} is too low for tensor dimension {dim}")
        }
        Ok((dim as i64 + index) as usize)
    }
}

pub fn actual_dim(t: &CoreTensor, dim: i64) -> candle_core::Result<usize> {
    let rank = t.rank();
    if 0 <= dim {
        let dim = dim as usize;
        if rank <= dim {
            candle_core::bail!("dimension index {dim} is too large for tensor rank {rank}")
        }
        Ok(dim)
    } else {
        if (rank as i64) < -dim {
            candle_core::bail!("dimension index {dim} is too low for tensor rank {rank}")
        }
        Ok((rank as i64 + dim) as usize)
    }
}

/// Returns true if the 'cuda' backend is available.
/// &RETURNS&: bool
fn cuda_is_available() -> bool {
    candle_core::utils::cuda_is_available()
}

/// Returns true if candle was compiled with 'accelerate' support.
/// &RETURNS&: bool
fn has_accelerate() -> bool {
    candle_core::utils::has_accelerate()
}

/// Returns true if candle was compiled with MKL support.
/// &RETURNS&: bool
fn has_mkl() -> bool {
    candle_core::utils::has_mkl()
}

/// Returns the number of threads used by the candle.
/// &RETURNS&: int
fn get_num_threads() -> usize {
    candle_core::utils::get_num_threads()
}

pub fn candle_utils(rb_candle: magnus::RModule) -> Result<()> {
    let rb_utils = rb_candle.define_module("Utils")?;
    rb_utils.define_singleton_method("cuda_is_available", function!(cuda_is_available, 0))?;
    rb_utils.define_singleton_method("get_num_threads", function!(get_num_threads, 0))?;
    rb_utils.define_singleton_method("has_accelerate", function!(has_accelerate, 0))?;
    rb_utils.define_singleton_method("has_mkl", function!(has_mkl, 0))?;
    Ok(())
}
