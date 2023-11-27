#[cfg(any(feature = "mkl", feature = "mkl-dynamic"))]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod bert;
mod roberta; 

pub use bert::{BertModel, Config, PositionEmbeddingType};
use candle_core::{Result, Tensor};
// pub use jina::JinaBertModel;
use backend_core::Batch;

#[cfg(feature = "cuda")]
mod flash_bert;
// mod jina;

#[cfg(feature = "cuda")]
pub use flash_bert::FlashBertModel;

pub(crate) trait Model {
    fn embed(&self, _batch: Batch) -> Result<Tensor> {
        candle_core::bail!("`embed` is not implemented for this model");
    }

    fn predict(&self, _batch: Batch) -> Result<Tensor> {
        candle_core::bail!("`predict is not implemented for this model");
    }
}