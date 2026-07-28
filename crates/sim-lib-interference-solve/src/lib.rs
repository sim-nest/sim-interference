#![forbid(unsafe_code)]
//! Deterministic CPU `f64` reference solving for coherent scalar wave fields.
//!
//! [`ReferencePhasorSolver`] consumes the checked physical model and preflight
//! vocabulary from `sim-lib-interference-core`. Successful solves return a
//! [`HostPhasorField`] with separate row-major real and imaginary component
//! planes plus immutable [`SolveEvidence`].
//!
//! [`verify_reference_solver`] independently checks closed-form propagation
//! identities, true metamorphic field laws, the outgoing time sign, and
//! second-order convergence of a seven-point Helmholtz residual over point,
//! plane, mixed, attenuating, and multi-source fixtures. The crate has no SIM
//! runtime, tensor, compute-provider, or presentation dependency.

mod analytic;
mod complex;
mod error;
mod field;
mod helmholtz;
mod reference;
mod verification;
mod verify;

pub use error::ReferenceSolveError;
pub use field::HostPhasorField;
pub use reference::{ReferencePhasorSolver, SolveEvidence};
pub use verification::{
    MAX_ANALYTIC_RELATIVE_ERROR, MAX_METAMORPHIC_RELATIVE_ERROR, VerificationError,
    VerificationReport,
};
pub use verify::verify_reference_solver;
