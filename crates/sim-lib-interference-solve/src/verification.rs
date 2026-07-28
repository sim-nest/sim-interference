//! Public measurements and diagnostics for reference verification.

use std::fmt;

use crate::ReferenceSolveError;

/// Largest relative error admitted for the closed-form analytic checks.
pub const MAX_ANALYTIC_RELATIVE_ERROR: f64 = 1.0e-12;

/// Largest relative error admitted for a metamorphic field law.
pub const MAX_METAMORPHIC_RELATIVE_ERROR: f64 = 1.0e-12;

/// Complete measurements from the deterministic reference verification suite.
///
/// The Helmholtz value is the matrix member farthest from ideal second-order
/// convergence. Every matrix member is checked before a report is returned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerificationReport {
    /// Largest relative error across the analytic identities and time-sign check.
    pub analytic_max_error: f64,
    /// Relative point Green-function reciprocity error.
    pub reciprocity_relative: f64,
    /// Largest relative superposition error.
    pub linearity_relative: f64,
    /// Largest relative error after a shared rotation and translation.
    pub rigid_motion_relative: f64,
    /// Largest relative error after rotating every phasor by one phase.
    pub global_phase_relative: f64,
    /// Whether every source input permutation produced identical component bits.
    pub source_permutation_identical: bool,
    /// Observed residual order farthest from ideal order two in the fixture matrix.
    pub helmholtz_observed_order: f64,
}

/// A deterministic verification suite failed before producing a passing report.
#[derive(Clone, Debug, PartialEq)]
pub enum VerificationError {
    /// A checked fixture or reference solve was refused.
    Reference {
        /// Exact reference-solver diagnostic.
        cause: Box<ReferenceSolveError>,
    },
    /// A measured analytic or metamorphic error exceeded its declared limit.
    RelativeErrorExceeded {
        /// Stable verification check name.
        check: &'static str,
        /// Measured relative error.
        measured: f64,
        /// Largest accepted relative error.
        limit: f64,
    },
    /// Canonical source permutations changed the field's component bits.
    SourcePermutationChanged,
    /// One named Helmholtz fixture did not exhibit second-order convergence.
    HelmholtzOrderOutOfRange {
        /// Stable fixture name.
        fixture: &'static str,
        /// Measured convergence order.
        observed: f64,
        /// Inclusive lower bound.
        minimum: f64,
        /// Inclusive upper bound.
        maximum: f64,
    },
}

impl From<ReferenceSolveError> for VerificationError {
    fn from(cause: ReferenceSolveError) -> Self {
        Self::Reference {
            cause: Box::new(cause),
        }
    }
}

impl fmt::Display for VerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reference { cause } => write!(formatter, "verification fixture failed: {cause}"),
            Self::RelativeErrorExceeded {
                check,
                measured,
                limit,
            } => write!(
                formatter,
                "verification check `{check}` measured relative error {measured:e}, \
                 exceeding {limit:e}"
            ),
            Self::SourcePermutationChanged => {
                formatter.write_str("canonical source permutations changed field component bits")
            }
            Self::HelmholtzOrderOutOfRange {
                fixture,
                observed,
                minimum,
                maximum,
            } => write!(
                formatter,
                "Helmholtz fixture `{fixture}` observed order {observed}, outside \
                 [{minimum}, {maximum}]"
            ),
        }
    }
}

impl std::error::Error for VerificationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reference { cause } => Some(cause.as_ref()),
            Self::RelativeErrorExceeded { .. }
            | Self::SourcePermutationChanged
            | Self::HelmholtzOrderOutOfRange { .. } => None,
        }
    }
}
