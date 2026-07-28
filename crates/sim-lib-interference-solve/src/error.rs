//! Stable diagnostics for reference-solver admission and execution.

use std::fmt;

use sim_lib_interference_core::InterferenceError;

/// A reference solve failed without returning a partial field.
#[derive(Clone, Debug, PartialEq)]
pub enum ReferenceSolveError {
    /// Sampling, work, or another request-wide preflight check failed.
    Request {
        /// Exact core diagnostic that refused the request.
        cause: Box<InterferenceError>,
    },
    /// A sampling cell could not be represented by the checked plane.
    CellGeometry {
        /// Zero-based row.
        row: usize,
        /// Zero-based column.
        column: usize,
        /// Exact geometry diagnostic.
        cause: Box<InterferenceError>,
    },
    /// One canonical source could not be evaluated at one sampling cell.
    SourceAtCell {
        /// Stable source identity.
        source_id: String,
        /// Zero-based row.
        row: usize,
        /// Zero-based column.
        column: usize,
        /// Exact propagation diagnostic.
        cause: Box<InterferenceError>,
    },
    /// Compensated component accumulation became non-finite.
    NonFiniteAccumulation {
        /// Stable source identity after whose contribution the failure arose.
        source_id: String,
        /// Zero-based row.
        row: usize,
        /// Zero-based column.
        column: usize,
        /// Component plane (`real` or `imaginary`).
        component: &'static str,
        /// Rejected accumulated value.
        value: f64,
    },
    /// Host storage could not be reserved for a complete component plane.
    AllocationFailed {
        /// Component plane whose reservation failed.
        component: &'static str,
        /// Requested cell count.
        cells: usize,
    },
}

impl From<InterferenceError> for ReferenceSolveError {
    fn from(cause: InterferenceError) -> Self {
        Self::Request {
            cause: Box::new(cause),
        }
    }
}

impl fmt::Display for ReferenceSolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request { cause } => {
                write!(formatter, "reference solve request refused: {cause}")
            }
            Self::CellGeometry { row, column, cause } => write!(
                formatter,
                "sampling cell ({row}, {column}) has invalid geometry: {cause}"
            ),
            Self::SourceAtCell {
                source_id,
                row,
                column,
                cause,
            } => write!(
                formatter,
                "source `{source_id}` failed at sampling cell ({row}, {column}): {cause}"
            ),
            Self::NonFiniteAccumulation {
                source_id,
                row,
                column,
                component,
                value,
            } => write!(
                formatter,
                "{component} accumulation became non-finite after source \
                 `{source_id}` at sampling cell ({row}, {column}): {value:?}"
            ),
            Self::AllocationFailed { component, cells } => write!(
                formatter,
                "could not reserve {cells} f64 cells for the {component} component plane"
            ),
        }
    }
}

impl std::error::Error for ReferenceSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request { cause }
            | Self::CellGeometry { cause, .. }
            | Self::SourceAtCell { cause, .. } => Some(cause.as_ref()),
            Self::NonFiniteAccumulation { .. } | Self::AllocationFailed { .. } => None,
        }
    }
}
