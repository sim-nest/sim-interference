//! Honest scalar projection with explicit phase masks and provenance.

use std::{f64::consts::PI, fmt};

use sim_lib_interference_core::SamplingCertificate;

use crate::{HostPhasorField, Observable, ReductionRule};

/// A non-zero two-dimensional field shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridDimensions {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}

impl GridDimensions {
    pub(crate) fn from_field(field: &HostPhasorField) -> Self {
        Self {
            rows: field.rows(),
            columns: field.columns(),
        }
    }

    /// Returns the number of rows.
    pub fn rows(self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn columns(self) -> usize {
        self.columns
    }
}

/// Inclusive bounds on the source-cell rectangle represented by one target cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetectorFootprint {
    pub(crate) min_rows: usize,
    pub(crate) max_rows: usize,
    pub(crate) min_columns: usize,
    pub(crate) max_columns: usize,
}

impl DetectorFootprint {
    pub(crate) fn detail() -> Self {
        Self {
            min_rows: 1,
            max_rows: 1,
            min_columns: 1,
            max_columns: 1,
        }
    }

    /// Returns the smallest number of source rows in a target cell.
    pub fn min_rows(self) -> usize {
        self.min_rows
    }

    /// Returns the largest number of source rows in a target cell.
    pub fn max_rows(self) -> usize {
        self.max_rows
    }

    /// Returns the smallest number of source columns in a target cell.
    pub fn min_columns(self) -> usize {
        self.min_columns
    }

    /// Returns the largest number of source columns in a target cell.
    pub fn max_columns(self) -> usize {
        self.max_columns
    }
}

/// Whether the projection preserves every source sample or integrates detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossClass {
    /// Source and target cells correspond one-to-one.
    Lossless,
    /// Multiple source samples may contribute to one detector cell.
    DetectorIntegration,
}

/// Immutable provenance for one complete scalar projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionCertificate {
    pub(crate) source_dimensions: GridDimensions,
    pub(crate) target_dimensions: GridDimensions,
    pub(crate) footprint: DetectorFootprint,
    pub(crate) observable: Observable,
    pub(crate) phase_floor: f64,
    pub(crate) rule: ReductionRule,
    pub(crate) loss_class: LossClass,
    pub(crate) source_sampling_certificate: SamplingCertificate,
    pub(crate) mask_count: usize,
}

impl ProjectionCertificate {
    /// Returns the source phasor shape.
    pub fn source_dimensions(self) -> GridDimensions {
        self.source_dimensions
    }

    /// Returns the scalar result shape.
    pub fn target_dimensions(self) -> GridDimensions {
        self.target_dimensions
    }

    /// Returns the inclusive source-cell footprint bounds.
    pub fn footprint(self) -> DetectorFootprint {
        self.footprint
    }

    /// Returns the projected observable.
    pub fn observable(self) -> Observable {
        self.observable
    }

    /// Returns the declared phase amplitude floor.
    pub fn phase_floor(self) -> f64 {
        self.phase_floor
    }

    /// Returns the applied reduction rule.
    pub fn rule(self) -> ReductionRule {
        self.rule
    }

    /// Returns the declared information-loss class.
    pub fn loss_class(self) -> LossClass {
        self.loss_class
    }

    /// Returns the source solve's sampling evidence unchanged.
    pub fn source_sampling_certificate(self) -> SamplingCertificate {
        self.source_sampling_certificate
    }

    /// Returns the number of target cells whose phase is undefined.
    pub fn mask_count(self) -> usize {
        self.mask_count
    }
}

/// One projected cell, with undefined values represented outside the numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarSample {
    /// A finite scalar value.
    Value(f64),
    /// Phase is undefined because amplitude is at or below the declared floor.
    Masked,
}

/// A complete row-major scalar projection and its inseparable provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarProjection {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) samples: Vec<ScalarSample>,
    pub(crate) certificate: ProjectionCertificate,
}

impl ScalarProjection {
    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the row-major scalar or masked samples.
    pub fn samples(&self) -> &[ScalarSample] {
        &self.samples
    }

    /// Returns one target cell.
    pub fn cell(&self, row: usize, column: usize) -> Option<ScalarSample> {
        let index = row.checked_mul(self.columns)?.checked_add(column)?;
        (row < self.rows && column < self.columns).then(|| self.samples[index])
    }

    /// Returns the immutable projection evidence.
    pub fn certificate(&self) -> ProjectionCertificate {
        self.certificate
    }
}

/// A scalar projection was refused without returning a partial result.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectionError {
    /// The phase masking threshold was negative or non-finite.
    InvalidPhaseFloor {
        /// Rejected amplitude floor.
        value: f64,
    },
    /// The instantaneous angular time was non-finite.
    InvalidAngularTime {
        /// Rejected angular time.
        wt: f64,
    },
    /// A target grid dimension was zero.
    ZeroTargetDimension {
        /// Rejected dimension name.
        axis: &'static str,
    },
    /// Reduction cannot invent target samples beyond the source resolution.
    TargetExceedsSource {
        /// Source field shape.
        source: GridDimensions,
        /// Rejected target shape.
        target: GridDimensions,
    },
    /// Detail mode was asked to discard source cells.
    DetailReductionRefused {
        /// Source field shape.
        source: GridDimensions,
        /// Rejected smaller target shape.
        target: GridDimensions,
    },
    /// A detector rule was paired with a scalar observable it does not measure.
    IncompatibleReduction {
        /// Requested observable.
        observable: Observable,
        /// Requested detector rule.
        rule: ReductionRule,
    },
    /// A projected scalar was not finite.
    NonFiniteSample {
        /// Zero-based source or target row.
        row: usize,
        /// Zero-based source or target column.
        column: usize,
        /// Observable being projected.
        observable: Observable,
        /// Rejected value.
        value: f64,
    },
    /// Result storage could not be reserved.
    AllocationFailed {
        /// Requested scalar cells.
        cells: usize,
    },
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPhaseFloor { value } => {
                write!(
                    formatter,
                    "phase amplitude floor must be finite and non-negative: {value:?}"
                )
            }
            Self::InvalidAngularTime { wt } => {
                write!(
                    formatter,
                    "instantaneous angular time must be finite: {wt:?}"
                )
            }
            Self::ZeroTargetDimension { axis } => {
                write!(formatter, "target {axis} must be greater than zero")
            }
            Self::TargetExceedsSource { source, target } => write!(
                formatter,
                "target {}x{} exceeds source {}x{}; projection does not upsample",
                target.rows, target.columns, source.rows, source.columns
            ),
            Self::DetailReductionRefused { source, target } => write!(
                formatter,
                "detail projection refuses reduction from {}x{} to {}x{}",
                source.rows, source.columns, target.rows, target.columns
            ),
            Self::IncompatibleReduction { observable, rule } => write!(
                formatter,
                "observable {observable:?} is incompatible with reduction rule {rule:?}"
            ),
            Self::NonFiniteSample {
                row,
                column,
                observable,
                value,
            } => write!(
                formatter,
                "{observable:?} projection produced non-finite sample {value:?} \
                 at ({row}, {column})"
            ),
            Self::AllocationFailed { cells } => {
                write!(
                    formatter,
                    "could not reserve {cells} scalar projection cells"
                )
            }
        }
    }
}

impl std::error::Error for ProjectionError {}

/// Projects one field at full source detail.
///
/// Phase is the only masked observable. A phase cell whose amplitude is equal
/// to the floor is masked, as is one below it. No numeric placeholder is
/// stored for those cells.
pub fn project(
    field: &HostPhasorField,
    source_sampling_certificate: SamplingCertificate,
    observable: Observable,
    phase_floor: f64,
) -> Result<ScalarProjection, ProjectionError> {
    validate_request(observable, phase_floor)?;
    let mut samples = allocate_samples(field.len())?;
    let mut mask_count = 0;
    for index in 0..field.len() {
        let row = index / field.columns();
        let column = index % field.columns();
        let sample = project_complex(
            field.real()[index],
            field.imaginary()[index],
            observable,
            phase_floor,
            row,
            column,
        )?;
        mask_count += usize::from(sample == ScalarSample::Masked);
        samples.push(sample);
    }

    let dimensions = GridDimensions::from_field(field);
    Ok(ScalarProjection {
        rows: field.rows(),
        columns: field.columns(),
        samples,
        certificate: ProjectionCertificate {
            source_dimensions: dimensions,
            target_dimensions: dimensions,
            footprint: DetectorFootprint::detail(),
            observable,
            phase_floor,
            rule: ReductionRule::Detail,
            loss_class: LossClass::Lossless,
            source_sampling_certificate,
            mask_count,
        },
    })
}

pub(crate) fn validate_request(
    observable: Observable,
    phase_floor: f64,
) -> Result<(), ProjectionError> {
    if !phase_floor.is_finite() || phase_floor < 0.0 {
        return Err(ProjectionError::InvalidPhaseFloor { value: phase_floor });
    }
    if let Observable::Instant { wt } = observable
        && !wt.is_finite()
    {
        return Err(ProjectionError::InvalidAngularTime { wt });
    }
    Ok(())
}

pub(crate) fn project_complex(
    real: f64,
    imaginary: f64,
    observable: Observable,
    phase_floor: f64,
    row: usize,
    column: usize,
) -> Result<ScalarSample, ProjectionError> {
    let amplitude = real.hypot(imaginary);
    let sample = match observable {
        Observable::Real => ScalarSample::Value(real),
        Observable::Imaginary => ScalarSample::Value(imaginary),
        Observable::Amplitude => ScalarSample::Value(amplitude),
        Observable::Phase if amplitude <= phase_floor => ScalarSample::Masked,
        Observable::Phase => {
            let phase = imaginary.atan2(real);
            ScalarSample::Value(if phase == PI { -PI } else { phase })
        }
        Observable::MagnitudeSquared => {
            ScalarSample::Value(real.mul_add(real, imaginary * imaginary))
        }
        Observable::Instant { wt } => ScalarSample::Value(if wt == 0.0 {
            real
        } else {
            real * wt.cos() + imaginary * wt.sin()
        }),
    };
    if let ScalarSample::Value(value) = sample
        && !value.is_finite()
    {
        return Err(ProjectionError::NonFiniteSample {
            row,
            column,
            observable,
            value,
        });
    }
    Ok(sample)
}

pub(crate) fn allocate_samples(cells: usize) -> Result<Vec<ScalarSample>, ProjectionError> {
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(cells)
        .map_err(|_| ProjectionError::AllocationFailed { cells })?;
    Ok(samples)
}

#[cfg(test)]
#[path = "projection_tests.rs"]
mod tests;
