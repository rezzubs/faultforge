//! Generic encodable bit buffers with fault injection support
//!
//! At the core of the library is the [`BitBuffer`] trait which gives bit-level
//! access to many common types. There are two additional supertraits for
//! extended functionality:
//!
//! - [`SizedBitBuffer`] for types which have a compile-time known length
//! - [`ByteBuffer`] for buffers which can be split at byte boundaries.
//!
//! # Trait Support table
//!
//! | Type | [`BitBuffer`] | [`SizedBitBuffer`] | [`ByteBuffer`] |
//! | - | - | - | - |
//! | numeric primitives (`u8`–`u128`, `i8`–`i128`, `f32`, `f64`) | ✓ | ✓ | ✓ |
//! | `[T; N]` where `T: SizedBitBuffer` | ✓ | ✓ | ✓† |
//! | `[T]` where `T: SizedBitBuffer` | ✓ | | ✓† |
//! | `Vec<T>` where `T: SizedBitBuffer` | ✓ | | ✓† |
//!
//! † [`ByteBuffer`] on sequence types additionally requires `T: ByteBuffer`.
//!
//! Types in the [`sequence`] module can be used to drop the `SizedBitBuffer`
//! requirement on `T`.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![warn(clippy::must_use_candidate)]

mod bit_buffer;
mod byte_buffer;
pub mod chunks;
pub mod encoding;
mod fault;
mod limited;
pub mod sequence;

pub use bit_buffer::{
    Bit, BitBuffer, Bits, CopyIntoResult, CopyIntoStatus, OutOfBoundsError, SizedBitBuffer,
};
pub use byte_buffer::ByteBuffer;
pub use fault::Fault;
pub use limited::Limited;
