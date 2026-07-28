//! Stable diagnostics for bounded named scenario construction.

use std::fmt;

use sim_lib_interference_core::InterferenceError;

/// A named scenario was refused before returning a partial problem.
#[derive(Clone, Debug, PartialEq)]
pub enum ScenarioError {
    /// A configured limit was zero or exceeded its absolute safety ceiling.
    InvalidLimit {
        /// Stable limit name.
        name: &'static str,
        /// Rejected value.
        value: usize,
        /// Absolute maximum.
        absolute_maximum: usize,
    },
    /// An array or aperture dimension was zero.
    ZeroElementDimension {
        /// Stable dimension name.
        name: &'static str,
    },
    /// Rectangular source count overflowed `usize`.
    SourceCountOverflow {
        /// Requested aperture rows.
        rows: usize,
        /// Requested aperture columns.
        columns: usize,
    },
    /// Source count exceeded the configured limit.
    SourceLimitExceeded {
        /// Requested source count.
        requested: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// One generated source id would exceed its configured byte limit.
    GeneratedIdLimitExceeded {
        /// Required bytes in the longest identity.
        requested: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Aggregate generated id storage would exceed its configured byte limit.
    TotalIdLimitExceeded {
        /// Required aggregate bytes.
        requested: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Strict policy rejected an active element spacing above `lambda / 2`.
    SparseAperture {
        /// Stable aperture axis (`u` or `v`).
        axis: &'static str,
        /// Rejected spacing in wavelengths.
        spacing_wavelengths: f64,
        /// Maximum strict spacing.
        maximum_wavelengths: f64,
    },
    /// A derived geometric or phase value was invalid.
    Core {
        /// Exact checked-domain diagnostic.
        cause: Box<InterferenceError>,
    },
    /// Source-vector storage could not be reserved after admission.
    AllocationFailed {
        /// Admitted source count.
        sources: usize,
    },
}

impl From<InterferenceError> for ScenarioError {
    fn from(cause: InterferenceError) -> Self {
        Self::Core {
            cause: Box::new(cause),
        }
    }
}

impl fmt::Display for ScenarioError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimit {
                name,
                value,
                absolute_maximum,
            } => write!(
                formatter,
                "scenario limit `{name}` must be in 1..={absolute_maximum}: {value}"
            ),
            Self::ZeroElementDimension { name } => {
                write!(
                    formatter,
                    "scenario element dimension `{name}` must be non-zero"
                )
            }
            Self::SourceCountOverflow { rows, columns } => write!(
                formatter,
                "scenario source count overflows usize: {rows} rows by {columns} columns"
            ),
            Self::SourceLimitExceeded { requested, limit } => write!(
                formatter,
                "scenario requests {requested} sources but the limit is {limit}"
            ),
            Self::GeneratedIdLimitExceeded { requested, limit } => write!(
                formatter,
                "scenario generated id requires {requested} bytes but the limit is {limit}"
            ),
            Self::TotalIdLimitExceeded { requested, limit } => write!(
                formatter,
                "scenario generated ids require {requested} bytes but the limit is {limit}"
            ),
            Self::SparseAperture {
                axis,
                spacing_wavelengths,
                maximum_wavelengths,
            } => write!(
                formatter,
                "strict aperture spacing on `{axis}` is {spacing_wavelengths:?} wavelengths, \
                 above {maximum_wavelengths:?}"
            ),
            Self::Core { cause } => write!(formatter, "scenario input is invalid: {cause}"),
            Self::AllocationFailed { sources } => {
                write!(
                    formatter,
                    "could not reserve storage for {sources} scenario sources"
                )
            }
        }
    }
}

impl std::error::Error for ScenarioError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core { cause } => Some(cause.as_ref()),
            _ => None,
        }
    }
}
