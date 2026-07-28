//! Stable diagnostics for certified multi-tone composition.

use std::fmt;

use sim_lib_interference_core::SamplingPlane;

use crate::ReferenceSolveError;

/// A multi-tone study or observation was refused without a partial result.
#[derive(Clone, Debug, PartialEq)]
pub enum MultiToneError {
    /// A tone's scale was zero, negative, or non-finite.
    InvalidWeight {
        /// Frequency of the rejected component.
        frequency_hz: f64,
        /// Rejected scale.
        weight: f64,
    },
    /// An independently certified component solve failed.
    ComponentSolve {
        /// Frequency of the failed component.
        frequency_hz: f64,
        /// Exact reference-solver diagnostic.
        cause: Box<ReferenceSolveError>,
    },
    /// A study must contain at least one tone.
    EmptyStudy,
    /// Equal frequencies must be modeled as one coherent problem.
    DuplicateFrequency {
        /// Rejected repeated frequency.
        frequency_hz: f64,
    },
    /// One component was solved on different physical sample geometry.
    MismatchedPlane {
        /// Frequency of the mismatched component.
        frequency_hz: f64,
        /// Plane established by the first component.
        expected: Box<SamplingPlane>,
        /// Rejected component plane.
        actual: Box<SamplingPlane>,
    },
    /// A shared observation time was non-finite.
    InvalidSeconds {
        /// Rejected time.
        seconds: f64,
    },
    /// Frequency-to-angular-time conversion overflowed.
    NonFiniteAngularTime {
        /// Component frequency.
        frequency_hz: f64,
        /// Requested shared time.
        seconds: f64,
        /// Rejected derived angular time.
        angular_time: f64,
    },
    /// Weighting one component scalar produced a non-finite value.
    NonFiniteContribution {
        /// Component frequency.
        frequency_hz: f64,
        /// Zero-based row.
        row: usize,
        /// Zero-based column.
        column: usize,
        /// Rejected weighted scalar.
        value: f64,
    },
    /// Compensated cross-tone scalar accumulation became non-finite.
    NonFiniteAccumulation {
        /// Zero-based row.
        row: usize,
        /// Zero-based column.
        column: usize,
        /// Rejected accumulated scalar.
        value: f64,
    },
    /// Scalar result storage could not be reserved.
    AllocationFailed {
        /// Requested scalar cells.
        cells: usize,
    },
    /// The set's temporal Nyquist floor could not be represented.
    NonFiniteTemporalSamplingRequirement {
        /// Highest component frequency.
        highest_frequency_hz: f64,
        /// Rejected derived Nyquist sample rate.
        samples_per_second: f64,
    },
    /// Component-certificate storage could not be reserved.
    CertificateAllocationFailed {
        /// Number of component certificates requested.
        tones: usize,
    },
}

impl fmt::Display for MultiToneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWeight {
                frequency_hz,
                weight,
            } => write!(
                formatter,
                "tone at {frequency_hz} Hz requires a finite positive weight: {weight:?}"
            ),
            Self::ComponentSolve {
                frequency_hz,
                cause,
            } => write!(
                formatter,
                "tone at {frequency_hz} Hz could not be certified: {cause}"
            ),
            Self::EmptyStudy => formatter.write_str("multi-tone study requires at least one tone"),
            Self::DuplicateFrequency { frequency_hz } => write!(
                formatter,
                "frequency {frequency_hz} Hz occurs more than once; equal-frequency sources belong in one coherent problem"
            ),
            Self::MismatchedPlane {
                frequency_hz,
                expected,
                actual,
            } => write!(
                formatter,
                "tone at {frequency_hz} Hz uses plane {actual:?}, not the shared plane {expected:?}"
            ),
            Self::InvalidSeconds { seconds } => write!(
                formatter,
                "multi-tone observation time must be finite: {seconds:?}"
            ),
            Self::NonFiniteAngularTime {
                frequency_hz,
                seconds,
                angular_time,
            } => write!(
                formatter,
                "tone at {frequency_hz} Hz and time {seconds} s produced non-finite angular time {angular_time:?}"
            ),
            Self::NonFiniteContribution {
                frequency_hz,
                row,
                column,
                value,
            } => write!(
                formatter,
                "tone at {frequency_hz} Hz produced non-finite weighted scalar {value:?} at ({row}, {column})"
            ),
            Self::NonFiniteAccumulation { row, column, value } => write!(
                formatter,
                "multi-tone scalar accumulation became non-finite at ({row}, {column}): {value:?}"
            ),
            Self::AllocationFailed { cells } => {
                write!(
                    formatter,
                    "could not reserve {cells} multi-tone scalar cells"
                )
            }
            Self::NonFiniteTemporalSamplingRequirement {
                highest_frequency_hz,
                samples_per_second,
            } => write!(
                formatter,
                "highest frequency {highest_frequency_hz} Hz produced non-finite Nyquist rate {samples_per_second:?}"
            ),
            Self::CertificateAllocationFailed { tones } => write!(
                formatter,
                "could not reserve provenance for {tones} multi-tone components"
            ),
        }
    }
}

impl std::error::Error for MultiToneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ComponentSolve { cause, .. } => Some(cause.as_ref()),
            Self::InvalidWeight { .. }
            | Self::EmptyStudy
            | Self::DuplicateFrequency { .. }
            | Self::MismatchedPlane { .. }
            | Self::InvalidSeconds { .. }
            | Self::NonFiniteAngularTime { .. }
            | Self::NonFiniteContribution { .. }
            | Self::NonFiniteAccumulation { .. }
            | Self::AllocationFailed { .. }
            | Self::NonFiniteTemporalSamplingRequirement { .. }
            | Self::CertificateAllocationFailed { .. } => None,
        }
    }
}
