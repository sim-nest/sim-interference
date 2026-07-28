//! Stable public diagnostics for interference-domain validation.

use std::fmt;

/// An invalid value at an interference-domain boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum InterferenceError {
    /// A quantity violated its finite, sign, or zero contract.
    InvalidQuantity {
        /// Stable public quantity name.
        name: &'static str,
        /// Rejected caller-supplied value.
        value: f64,
    },
    /// A direction was zero-length or had a non-finite component.
    InvalidDirection {
        /// Rejected x component.
        x: f64,
        /// Rejected y component.
        y: f64,
        /// Rejected z component.
        z: f64,
    },
    /// A coherent problem was given no sources.
    EmptySourceSet,
    /// A source identity was empty.
    EmptySourceId,
    /// More than one source used the same stable identity.
    DuplicateSourceId {
        /// Repeated source identity.
        id: String,
    },
    /// A finite input combination produced a non-finite propagation value.
    NonFinitePropagation {
        /// Source being evaluated.
        source_id: String,
        /// Stable name for the derived value.
        name: &'static str,
        /// Rejected derived value.
        value: f64,
    },
    /// A point-source sample entered its excluded singular region.
    SingularPointSample {
        /// Point-source identity.
        source_id: String,
        /// Sample distance from the point source.
        distance_metres: f64,
        /// Inclusive exclusion radius.
        singularity_radius_metres: f64,
    },
    /// A sample was behind a forward-plane emitter.
    BehindForwardPlane {
        /// Forward-plane source identity.
        source_id: String,
        /// Negative signed distance from the source plane.
        signed_distance_metres: f64,
    },
}

impl fmt::Display for InterferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidQuantity { name, value } => {
                write!(formatter, "invalid quantity `{name}`: {value:?}")
            }
            Self::InvalidDirection { x, y, z } => {
                write!(formatter, "invalid unit direction: [{x:?}, {y:?}, {z:?}]")
            }
            Self::EmptySourceSet => formatter.write_str("a source set cannot be empty"),
            Self::EmptySourceId => formatter.write_str("a source id cannot be empty"),
            Self::DuplicateSourceId { id } => {
                write!(formatter, "duplicate source id `{id}`")
            }
            Self::NonFinitePropagation {
                source_id,
                name,
                value,
            } => write!(
                formatter,
                "source `{source_id}` produced non-finite `{name}`: {value:?}"
            ),
            Self::SingularPointSample {
                source_id,
                distance_metres,
                singularity_radius_metres,
            } => write!(
                formatter,
                "sample is singular for point source `{source_id}`: distance \
                 {distance_metres:?} m is at or inside radius \
                 {singularity_radius_metres:?} m"
            ),
            Self::BehindForwardPlane {
                source_id,
                signed_distance_metres,
            } => write!(
                formatter,
                "sample is behind forward-plane source `{source_id}`: signed \
                 distance {signed_distance_metres:?} m"
            ),
        }
    }
}

impl std::error::Error for InterferenceError {}
