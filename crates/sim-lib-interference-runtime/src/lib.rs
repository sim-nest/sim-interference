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
mod lib_impl;
mod ops;
mod ops_outputs;
mod ops_values;
mod projection_records;
mod records;
mod sampling;
mod shapes;
mod solver;
mod tensor_bridge;

pub use citizen::{
    InterferenceRecordsLib, install_interference_records, interference_citizen_registry,
};
pub use evidence::{
    SamplingCertificateDescriptor, StudyDescriptor, StudyEvidenceDescriptor, WorkEstimateDescriptor,
};
pub use lib_impl::{InterferenceLib, interference_lib_symbol};
pub use ops::{
    analyze_function_symbol, multitone_function_symbol, problem_function_symbol,
    project_function_symbol, sampling_plane_function_symbol, scenarios_function_symbol,
    solve_function_symbol,
};
pub use projection_records::{
    ProjectionCertificateDescriptor, ProjectionRequestDescriptor, ScalarProjectionDescriptor,
};
pub use records::{EmitterDescriptor, MediumDescriptor, PlaneDescriptor, ProblemDescriptor};
pub use shapes::{
    interference_shape_symbols, plane_shape_symbol, problem_shape_symbol,
    projection_request_shape_symbol, projection_shape_symbol, study_shape_symbol,
};
pub use solver::{
    InterferenceStudy, ReferenceStudySolver, SolveRequest, SolverProvider, StudySolver,
    resolve_study_solver, study_solver_symbol, tensor_study_solver_symbol,
};
pub use tensor_bridge::PhasorFieldDescriptor;

/// Checked CPU interference recipes embedded with the loadable library.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod runtime_tests;
#[cfg(test)]
mod tests;
