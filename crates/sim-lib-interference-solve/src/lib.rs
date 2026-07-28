#![forbid(unsafe_code)]
//! Deterministic CPU `f64` reference solving for coherent scalar wave fields.
//!
//! [`ReferencePhasorSolver`] consumes the checked physical model and preflight
//! vocabulary from `sim-lib-interference-core`. Successful solves return a
//! [`HostPhasorField`] with separate row-major real and imaginary component
//! planes plus immutable [`SolveEvidence`]. The crate has no SIM runtime,
//! tensor, compute-provider, or presentation dependency.

mod complex;
mod error;
mod field;
mod reference;

pub use error::ReferenceSolveError;
pub use field::HostPhasorField;
pub use reference::{ReferencePhasorSolver, SolveEvidence};
