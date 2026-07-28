#![forbid(unsafe_code)]
//! Checked physical boundaries for coherent scalar wave-field studies.
//!
//! This crate owns the dependency-free quantity vocabulary used by the
//! interference family. It deliberately does not define propagation, solving,
//! tensor storage, runtime bindings, or presentation.

mod error;
mod quantity;

pub use error::InterferenceError;
pub use quantity::{
    FieldAmplitude, Hertz, Metres, MetresPerSecond, NepersPerMetre, PositiveMetres, Radians,
};
