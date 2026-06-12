//! [`BitBuffer`](crate::BitBuffer) for sequences where the items only have a
//! runtime-known length.

mod non_uniform;
mod uniform;

pub use non_uniform::NonUniformSequence;
pub use uniform::{NonMatchingItemError, UniformSequence};
