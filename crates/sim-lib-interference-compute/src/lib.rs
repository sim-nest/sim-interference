#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Normalized tile-local `f32` lowering over the canonical Tensor executor.
//!
//! The host keeps physical geometry and absolute phase in `f64`. Each prepared
//! tile exposes only bounded offsets and residual phase to ordinary open Tensor
//! operations, so no executor needs interference-specific behavior.

mod constants;
mod coordinates;
mod dense;
mod diff;
mod lower;
mod preflight;
mod tile;

pub use constants::{
    PlaneTileConstants, PointTileConstants, SourcePhaseEstimate, SourceTileConstants,
};
pub use dense::{DenseExecutionEvidence, DenseF32Field, solve_dense_f32_cpu};
pub use diff::{
    ConformanceMetric, DifferentialError, DifferentialMaximum, DifferentialReport,
    DifferentialTolerances, ScalarTolerance, compare_dense_to_reference,
};
pub use lower::{LoweredTile, LoweringPlan};
pub use preflight::{
    LoweringError, PhaseBudget, PreflightCheck, REQUIRED_TENSOR_OPERATION_NAMES, TileProfile,
};
pub use tile::{PlaneTile, TilePlan};

#[cfg(test)]
mod tests;
