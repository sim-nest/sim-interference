//! Stable public diagnostics for interference-domain validation.

use std::fmt;

use crate::{SamplingCertificate, WorkMetric};

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
    /// Sampling-plane axes were not orthogonal within the public tolerance.
    NonOrthogonalSamplingAxes {
        /// Dot product of the normalized axes.
        dot_product: f64,
        /// Largest accepted absolute dot product.
        max_abs_dot_product: f64,
    },
    /// A sampling-plane dimension was zero.
    ZeroSamplingDimension {
        /// Stable dimension name (`rows` or `columns`).
        name: &'static str,
    },
    /// The sampling-plane cell count could not be represented.
    SamplingCellCountOverflow {
        /// Requested row count.
        rows: usize,
        /// Requested column count.
        columns: usize,
    },
    /// A derived sampling-plane cell size was not positive and finite.
    InvalidSamplingCellSize {
        /// Stable axis name (`u` or `v`).
        axis: &'static str,
        /// Rejected derived cell size in metres.
        value: f64,
    },
    /// A requested sampling cell lies outside the plane.
    SamplingCellOutOfBounds {
        /// Requested row.
        row: usize,
        /// Requested column.
        column: usize,
        /// Plane row count.
        rows: usize,
        /// Plane column count.
        columns: usize,
    },
    /// A finite sampling input combination produced a non-finite metric.
    NonFiniteSamplingMetric {
        /// Stable metric name.
        name: &'static str,
        /// Rejected derived value.
        value: f64,
    },
    /// A sampling classification threshold was not finite and in-range.
    InvalidSamplingThreshold {
        /// Stable threshold field name.
        name: &'static str,
        /// Rejected caller-supplied value.
        value: f64,
    },
    /// Sampling thresholds did not order resolved before marginal.
    InconsistentSamplingThresholds {
        /// Comfortable carrier samples per wavelength.
        resolved_min_samples_per_wavelength: f64,
        /// Minimum carrier samples per wavelength.
        marginal_min_samples_per_wavelength: f64,
        /// Comfortable envelope fraction.
        resolved_max_envelope_fraction_per_cell: f64,
        /// Maximum marginal envelope fraction.
        marginal_max_envelope_fraction_per_cell: f64,
    },
    /// Strict sampling policy rejected a non-resolved certificate.
    SamplingRefused {
        /// Complete measurements and thresholds that caused refusal.
        certificate: SamplingCertificate,
    },
    /// A point source touched the plane or exceeded finite envelope range.
    UnboundedSamplingEnvelope {
        /// Nearest distance from a point source to the finite plane.
        nearest_point_source_distance_m: f64,
        /// Full diagonal of one sampling cell.
        cell_diagonal_m: f64,
    },
    /// Checked arithmetic could not represent a work-estimate metric.
    WorkEstimateOverflow {
        /// Metric whose derivation overflowed.
        metric: WorkMetric,
    },
    /// A work estimate exceeded one explicit budget field.
    WorkBudgetExceeded {
        /// Metric whose limit was exceeded.
        metric: WorkMetric,
        /// Requested work.
        estimate: u64,
        /// Configured maximum.
        limit: u64,
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
            Self::NonOrthogonalSamplingAxes {
                dot_product,
                max_abs_dot_product,
            } => write!(
                formatter,
                "sampling axes are not orthogonal: dot product \
                 {dot_product:?} exceeds {max_abs_dot_product:?}"
            ),
            Self::ZeroSamplingDimension { name } => {
                write!(formatter, "sampling dimension `{name}` must be non-zero")
            }
            Self::SamplingCellCountOverflow { rows, columns } => write!(
                formatter,
                "sampling cell count overflows usize: {rows} rows by {columns} columns"
            ),
            Self::InvalidSamplingCellSize { axis, value } => write!(
                formatter,
                "sampling cell size on `{axis}` must be positive and finite: {value:?} m"
            ),
            Self::SamplingCellOutOfBounds {
                row,
                column,
                rows,
                columns,
            } => write!(
                formatter,
                "sampling cell ({row}, {column}) is outside {rows} rows by {columns} columns"
            ),
            Self::NonFiniteSamplingMetric { name, value } => {
                write!(
                    formatter,
                    "sampling metric `{name}` is non-finite: {value:?}"
                )
            }
            Self::InvalidSamplingThreshold { name, value } => {
                write!(formatter, "invalid sampling threshold `{name}`: {value:?}")
            }
            Self::InconsistentSamplingThresholds {
                resolved_min_samples_per_wavelength,
                marginal_min_samples_per_wavelength,
                resolved_max_envelope_fraction_per_cell,
                marginal_max_envelope_fraction_per_cell,
            } => write!(
                formatter,
                "sampling thresholds are inconsistent: resolved carrier minimum \
                 {resolved_min_samples_per_wavelength:?} must be at least marginal \
                 minimum {marginal_min_samples_per_wavelength:?}, and resolved \
                 envelope maximum {resolved_max_envelope_fraction_per_cell:?} must \
                 not exceed marginal maximum {marginal_max_envelope_fraction_per_cell:?}"
            ),
            Self::SamplingRefused { certificate } => write!(
                formatter,
                "strict sampling refused {:?}: carrier samples/wavelength \
                 u={:?}, v={:?}; power-fringe samples/period u={:?}, v={:?}; \
                 envelope fraction/cell={:?}",
                certificate.verdict,
                certificate.samples_per_wavelength_u,
                certificate.samples_per_wavelength_v,
                certificate.samples_per_power_fringe_u,
                certificate.samples_per_power_fringe_v,
                certificate.max_envelope_fraction_per_cell,
            ),
            Self::UnboundedSamplingEnvelope {
                nearest_point_source_distance_m,
                cell_diagonal_m,
            } => write!(
                formatter,
                "point-source envelope bound is not finite: nearest plane distance \
                 {nearest_point_source_distance_m:?} m, cell diagonal \
                 {cell_diagonal_m:?} m"
            ),
            Self::WorkEstimateOverflow { metric } => {
                write!(formatter, "work estimate for {metric} overflowed u64")
            }
            Self::WorkBudgetExceeded {
                metric,
                estimate,
                limit,
            } => write!(
                formatter,
                "work budget exceeded for {metric}: estimate {estimate}, limit {limit}"
            ),
        }
    }
}

impl std::error::Error for InterferenceError {}
