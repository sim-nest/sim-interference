//! Detector rules for spatial field reduction.

use std::ops::Range;

use sim_lib_interference_core::SamplingCertificate;

use crate::{
    DetectorFootprint, GridDimensions, HostPhasorField, LossClass, Observable,
    ProjectionCertificate, ProjectionError, ScalarProjection, ScalarSample, project,
    projection::{allocate_samples, project_complex, validate_request},
};

/// The declared rule used to map source cells to target detector cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionRule {
    /// Preserve source detail; any smaller target is refused.
    Detail,
    /// Average the complex field over each footprint, then project it.
    DetectorComplexMean,
    /// Project amplitude at every source cell, then take its area mean.
    DetectorScalarAreaMean,
    /// Project squared magnitude at every source cell, then take its area mean.
    DetectorMagnitudeSquaredAreaMean,
}

/// Reduces a phasor field onto a no-larger target grid using one named detector.
///
/// Each target axis is an integer partition of its source axis. The Cartesian
/// product of those partitions covers every source cell exactly once, including
/// odd dimensions and non-divisible target shapes. Detail mode accepts only the
/// identity shape. Detector rules are deliberately paired with the observables
/// whose semantics they preserve; wrapped phase angles are never averaged.
#[allow(clippy::too_many_arguments)]
pub fn reduce_for_view(
    field: &HostPhasorField,
    source_sampling_certificate: SamplingCertificate,
    observable: Observable,
    phase_floor: f64,
    target_rows: usize,
    target_columns: usize,
    rule: ReductionRule,
) -> Result<ScalarProjection, ProjectionError> {
    validate_request(observable, phase_floor)?;
    let source = GridDimensions::from_field(field);
    let target = target_dimensions(target_rows, target_columns)?;
    if target.rows > source.rows || target.columns > source.columns {
        return Err(ProjectionError::TargetExceedsSource { source, target });
    }
    if rule == ReductionRule::Detail {
        if target != source {
            return Err(ProjectionError::DetailReductionRefused { source, target });
        }
        return project(field, source_sampling_certificate, observable, phase_floor);
    }
    require_compatible(observable, rule)?;

    let cells = target
        .rows
        .checked_mul(target.columns)
        .expect("a target no larger than an existing field must fit");
    let mut samples = allocate_samples(cells)?;
    let mut mask_count = 0;
    for target_row in 0..target.rows {
        let source_rows = axis_partition(target_row, source.rows, target.rows);
        for target_column in 0..target.columns {
            let source_columns = axis_partition(target_column, source.columns, target.columns);
            let sample = reduce_cell(
                field,
                observable,
                phase_floor,
                source_rows.clone(),
                source_columns,
                target_row,
                target_column,
                rule,
            )?;
            mask_count += usize::from(sample == ScalarSample::Masked);
            samples.push(sample);
        }
    }

    Ok(ScalarProjection {
        rows: target.rows,
        columns: target.columns,
        samples,
        certificate: ProjectionCertificate {
            source_dimensions: source,
            target_dimensions: target,
            footprint: DetectorFootprint {
                min_rows: source.rows / target.rows,
                max_rows: source.rows.div_ceil(target.rows),
                min_columns: source.columns / target.columns,
                max_columns: source.columns.div_ceil(target.columns),
            },
            observable,
            phase_floor,
            rule,
            loss_class: if source == target {
                LossClass::Lossless
            } else {
                LossClass::DetectorIntegration
            },
            source_sampling_certificate,
            mask_count,
        },
    })
}

fn target_dimensions(rows: usize, columns: usize) -> Result<GridDimensions, ProjectionError> {
    if rows == 0 {
        return Err(ProjectionError::ZeroTargetDimension { axis: "rows" });
    }
    if columns == 0 {
        return Err(ProjectionError::ZeroTargetDimension { axis: "columns" });
    }
    Ok(GridDimensions { rows, columns })
}

fn require_compatible(observable: Observable, rule: ReductionRule) -> Result<(), ProjectionError> {
    let compatible = matches!(
        (observable, rule),
        (
            Observable::Real
                | Observable::Imaginary
                | Observable::Phase
                | Observable::Instant { .. },
            ReductionRule::DetectorComplexMean
        ) | (Observable::Amplitude, ReductionRule::DetectorScalarAreaMean)
            | (
                Observable::MagnitudeSquared,
                ReductionRule::DetectorMagnitudeSquaredAreaMean
            )
    );
    compatible
        .then_some(())
        .ok_or(ProjectionError::IncompatibleReduction { observable, rule })
}

fn axis_partition(index: usize, source: usize, target: usize) -> Range<usize> {
    let base = source / target;
    let remainder = source % target;
    let start = index * base + index.min(remainder);
    let length = base + usize::from(index < remainder);
    start..start + length
}

#[allow(clippy::too_many_arguments)]
fn reduce_cell(
    field: &HostPhasorField,
    observable: Observable,
    phase_floor: f64,
    source_rows: Range<usize>,
    source_columns: Range<usize>,
    target_row: usize,
    target_column: usize,
    rule: ReductionRule,
) -> Result<ScalarSample, ProjectionError> {
    let count = source_rows.len() * source_columns.len();
    let scale = 1.0 / count as f64;
    match rule {
        ReductionRule::Detail => unreachable!("detail returns before detector traversal"),
        ReductionRule::DetectorComplexMean => {
            let (mut real_sum, mut real_correction) = (0.0, 0.0);
            let (mut imaginary_sum, mut imaginary_correction) = (0.0, 0.0);
            let first_index = source_rows.start * field.columns() + source_columns.start;
            let first_real = field.real()[first_index];
            let first_imaginary = field.imaginary()[first_index];
            let mut constant = true;
            for row in source_rows {
                for column in source_columns.clone() {
                    let index = row * field.columns() + column;
                    constant &= field.real()[index].to_bits() == first_real.to_bits()
                        && field.imaginary()[index].to_bits() == first_imaginary.to_bits();
                    compensated_add(
                        &mut real_sum,
                        &mut real_correction,
                        field.real()[index] * scale,
                    );
                    compensated_add(
                        &mut imaginary_sum,
                        &mut imaginary_correction,
                        field.imaginary()[index] * scale,
                    );
                }
            }
            let (real, imaginary) = if constant {
                (first_real, first_imaginary)
            } else {
                (
                    real_sum + real_correction,
                    imaginary_sum + imaginary_correction,
                )
            };
            project_complex(
                real,
                imaginary,
                observable,
                phase_floor,
                target_row,
                target_column,
            )
        }
        ReductionRule::DetectorScalarAreaMean | ReductionRule::DetectorMagnitudeSquaredAreaMean => {
            let (mut sum, mut correction) = (0.0, 0.0);
            let mut first_value: Option<f64> = None;
            let mut constant = true;
            for row in source_rows {
                for column in source_columns.clone() {
                    let index = row * field.columns() + column;
                    let ScalarSample::Value(value) = project_complex(
                        field.real()[index],
                        field.imaginary()[index],
                        observable,
                        phase_floor,
                        row,
                        column,
                    )?
                    else {
                        unreachable!("amplitude and squared magnitude are never masked");
                    };
                    if let Some(first) = first_value {
                        constant &= value.to_bits() == first.to_bits();
                    } else {
                        first_value = Some(value);
                    }
                    compensated_add(&mut sum, &mut correction, value * scale);
                }
            }
            let value = if constant {
                first_value.expect("detector partitions are non-empty")
            } else {
                sum + correction
            };
            if value.is_finite() {
                Ok(ScalarSample::Value(value))
            } else {
                Err(ProjectionError::NonFiniteSample {
                    row: target_row,
                    column: target_column,
                    observable,
                    value,
                })
            }
        }
    }
}

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let next = *sum + value;
    *correction += if sum.abs() >= value.abs() {
        (*sum - next) + value
    } else {
        (value - next) + *sum
    };
    *sum = next;
}

#[cfg(test)]
#[path = "reduce_tests.rs"]
mod tests;
