#![forbid(unsafe_code)]
//! Checked physical boundaries for coherent scalar wave-field studies.
//!
//! This crate owns the dependency-free quantity and source vocabulary used by
//! the interference family. It deliberately does not define grid solving,
//! tensor storage, runtime bindings, or presentation.
//!
//! # Governing convention
//!
//! The real field is `u(x, t) = Re{U(x) * exp(-i * omega * t)}`. In a
//! homogeneous scalar medium,
//! `k_tilde = omega / c + i * alpha` and, away from point sources,
//! `laplacian(U) + k_tilde^2 * U = 0`.
//!
//! A point emitter contributes
//! `A * (R_ref / r) * exp(i * k_tilde * r + i * phase)`, where
//! [`POINT_SOURCE_REFERENCE_DISTANCE_METRES`] defines `R_ref`. A forward-plane
//! emitter contributes `A * exp(i * k_tilde * s + i * phase)` only for
//! non-negative signed distance `s`. The positive propagation sign is outgoing
//! under the `exp(-i * omega * t)` time convention, and the imaginary part of
//! `k_tilde` gives `exp(-alpha * distance)` attenuation.

mod emitter;
mod error;
mod geometry;
mod medium;
mod problem;
mod propagation;
mod quantity;

pub use emitter::{Emitter, SourceSet};
pub use error::InterferenceError;
pub use geometry::{Point3M, UnitVector3};
pub use medium::{ScalarMedium, WaveNumber};
pub use problem::{InterferenceProblem, POINT_SOURCE_REFERENCE_DISTANCE_METRES};
pub use propagation::{contribution_at, forward_plane_contribution_at, point_contribution_at};
pub use quantity::{
    FieldAmplitude, Hertz, Metres, MetresPerSecond, NepersPerMetre, PositiveMetres, Radians,
};
