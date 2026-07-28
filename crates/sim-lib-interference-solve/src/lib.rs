#![forbid(unsafe_code)]
//! Deterministic CPU `f64` solving and certified observation of coherent scalar fields.
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
//!
//! [`project`] derives real, imaginary, amplitude, wrapped phase,
//! squared-magnitude, and instantaneous scalar fields without rerunning
//! propagation. [`reduce_for_view`] applies an explicit detector rule over an
//! exact partition of the source grid. Undefined phase is
//! [`ScalarSample::Masked`], detail reduction is fail-closed, and every result
//! carries a [`ProjectionCertificate`] retaining source sampling evidence.
//! [`analyze_fringes`] then derives deterministic statistics, strict local
//! node/antinode candidates, and Michelson contrast without dropping either
//! sampling evidence or projection identity.

mod analysis;
mod analytic;
mod complex;
mod error;
mod field;
mod helmholtz;
mod observable;
mod projection;
mod reduce;
mod reference;
mod verification;
mod verify;

pub use analysis::{
    AnalysisError, Extremum, ExtremumKind, FieldStats, FringeReport, analyze_fringes,
};
pub use error::ReferenceSolveError;
pub use field::HostPhasorField;
pub use observable::Observable;
pub use projection::{
    DetectorFootprint, GridDimensions, LossClass, ProjectionCertificate, ProjectionError,
    ProjectionIdentity, ScalarProjection, ScalarSample, project,
};
pub use reduce::{ReductionRule, reduce_for_view};
pub use reference::{ReferencePhasorSolver, SolveEvidence};
pub use verification::{
    MAX_ANALYTIC_RELATIVE_ERROR, MAX_METAMORPHIC_RELATIVE_ERROR, VerificationError,
    VerificationReport,
};
pub use verify::verify_reference_solver;
