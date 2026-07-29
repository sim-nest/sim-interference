#![forbid(unsafe_code)]
//! Deterministic CPU `f64` solving and certified observation of coherent scalar fields.
//!
//! [`ReferencePhasorSolver`] consumes the checked physical model and preflight
//! vocabulary from `sim-lib-interference-core`. Successful solves return a
//! [`HostPhasorField`] with separate row-major real and imaginary component
//! planes plus immutable [`SolveEvidence`].
//!
//! The solved model is homogeneous, isotropic, three-dimensional, scalar, and
//! free-field. It does not model polarization, impedance, interfaces,
//! obstacles, or diffraction. Amplitude-squared projections and detector
//! reductions are normalized observables, not physical intensity, power, or
//! energy.
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
//! [`ScenarioBuilder`] constructs bounded two-point, counter-propagating,
//! phased-array, and rectangular discrete-aperture problems. Array amplitudes
//! are normalized and wavelength-relative element spacing remains explicit.
//! [`ToneStudy`] seals one independently solved frequency with its exact
//! problem, plane, positive composition weight, and solve evidence.
//! [`MultiToneStudy`] admits only certified tones on identical physical sample
//! geometry. [`ToneCombination::IncoherentMagnitudeSquared`] sums weighted
//! component squared magnitudes and [`ToneCombination::Instant`] sums
//! weighted real fields at one shared clock time; unlike phasors are never
//! added. Every [`MultiToneProjection`] carries [`MultiToneCertificate`] with
//! the highest-frequency [`MultiToneSamplingRequirements`] and all component
//! [`ToneCertificate`] records.

mod analysis;
mod analytic;
mod complex;
mod error;
mod field;
mod helmholtz;
mod multitone;
mod multitone_error;
mod observable;
mod projection;
mod reduce;
mod reference;
mod scenario;
mod scenario_admission;
mod scenario_error;
mod verification;
mod verify;

pub use analysis::{
    AnalysisError, Extremum, ExtremumKind, FieldStats, FringeReport, analyze_fringes,
};
pub use error::ReferenceSolveError;
pub use field::{HostPhasorField, HostPhasorFieldError};
pub use multitone::{
    MultiToneCertificate, MultiToneProjection, MultiToneSamplingRequirements, MultiToneStudy,
    ToneCertificate, ToneCombination, ToneStudy,
};
pub use multitone_error::MultiToneError;
pub use observable::Observable;
pub use projection::{
    DetectorFootprint, GridDimensions, LossClass, ProjectionCertificate, ProjectionError,
    ProjectionIdentity, ScalarProjection, ScalarSample, project,
};
pub use reduce::{ReductionRule, reduce_for_view};
pub use reference::{ReferencePhasorSolver, SolveEvidence};
pub use scenario::{
    ABSOLUTE_MAX_GENERATED_ID_BYTES, ABSOLUTE_MAX_SCENARIO_SOURCES, ABSOLUTE_MAX_TOTAL_ID_BYTES,
    AperturePolicy, ElementSpacingWavelengths, NamedScenario, STRICT_MAX_SPACING_WAVELENGTHS,
    ScenarioBuilder, ScenarioCertificate, ScenarioKind, ScenarioLimits,
};
pub use scenario_error::ScenarioError;
pub use verification::{
    MAX_ANALYTIC_RELATIVE_ERROR, MAX_METAMORPHIC_RELATIVE_ERROR, VerificationError,
    VerificationReport,
};
pub use verify::verify_reference_solver;
