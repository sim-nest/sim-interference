#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Tensor-backed runtime records for coherent interference studies.
//!
//! Domain values cross this boundary as validated Citizen read-constructs.
//! Phasors reuse two canonical [`sim_lib_numbers_tensor::Tensor`] values, scalar
//! projections reuse one, and registered Shapes admit public runtime calls.

#[macro_use]
mod citizen;
mod evidence;
mod projection_records;
mod records;
mod sampling;
mod shapes;
mod tensor_bridge;

pub use citizen::{
    InterferenceRecordsLib, install_interference_records, interference_citizen_registry,
};
pub use evidence::{
    SamplingCertificateDescriptor, StudyDescriptor, StudyEvidenceDescriptor, WorkEstimateDescriptor,
};
pub use projection_records::{
    ProjectionCertificateDescriptor, ProjectionRequestDescriptor, ScalarProjectionDescriptor,
};
pub use records::{EmitterDescriptor, MediumDescriptor, PlaneDescriptor, ProblemDescriptor};
pub use shapes::{
    interference_shape_symbols, plane_shape_symbol, problem_shape_symbol,
    projection_request_shape_symbol, projection_shape_symbol, study_shape_symbol,
};
pub use tensor_bridge::PhasorFieldDescriptor;

#[cfg(test)]
mod tests;
