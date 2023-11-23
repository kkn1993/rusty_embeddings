// mod alibi;
#[cfg(feature = "cuda")]
mod compute_cap;
#[cfg(feature = "cuda")]
mod flash_attn;
mod layers;
mod models;

#[cfg(feature = "cuda")]
use crate::compute_cap::{incompatible_compute_cap, COMPILE_COMPUTE_CAP, RUNTIME_COMPUTE_CAP};
#[cfg(feature = "cuda")]
use crate::models::FlashBertModel;
use crate::models::{BertModel, Model, PositionEmbeddingType};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use models::Config;
use std::path::PathBuf;
use backend_core::{Backend, BackendError, Batch, Embedding, ModelType};

pub struct CandleBackend {
    model: Box<dyn Model + Send>,
}


pub trait WrapErr<O> {
    fn s(self) -> Result<O, BackendError>;
    fn e(self) -> Result<O, BackendError>;
}

impl<O> WrapErr<O> for Result<O, candle_core::Error> {
    fn s(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Start(e.to_string()))
    }
    fn e(self) -> Result<O, BackendError> {
        self.map_err(|e| BackendError::Inference(e.to_string()))
    }
}