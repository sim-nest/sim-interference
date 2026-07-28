use std::f64::consts::PI;

use sim_lib_interference_core::{SamplingCertificate, SamplingThresholds, SamplingVerdict};

use super::{ReductionRule, axis_partition, reduce_for_view};
use crate::{
    GridDimensions, HostPhasorField, LossClass, Observable, ProjectionError, ScalarProjection,
    ScalarSample,
};

fn certificate() -> SamplingCertificate {
    SamplingCertificate {
        thresholds: SamplingThresholds::default(),
        wavelength_m: 1.0,
        samples_per_wavelength_u: 16.0,
        samples_per_wavelength_v: 16.0,
        samples_per_power_fringe_u: 8.0,
        samples_per_power_fringe_v: 8.0,
        nearest_point_source_distance_m: None,
        max_envelope_fraction_per_cell: 0.0,
        verdict: SamplingVerdict::Resolved,
    }
}

fn positive_field() -> HostPhasorField {
    HostPhasorField::from_test_components(3, 5, (1..=15).map(f64::from).collect(), vec![0.0; 15])
}

fn values(projection: &ScalarProjection) -> Vec<f64> {
    projection
        .samples()
        .iter()
        .map(|sample| match sample {
            ScalarSample::Value(value) => *value,
            ScalarSample::Masked => panic!("fixture projection must be numeric"),
        })
        .collect()
}

fn assert_values_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-14,
            "{actual:?} differs from {expected:?}"
        );
    }
}

#[test]
fn detail_refuses_any_smaller_target() {
    let error = reduce_for_view(
        &positive_field(),
        certificate(),
        Observable::Real,
        0.0,
        2,
        5,
        ReductionRule::Detail,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ProjectionError::DetailReductionRefused { .. }
    ));
}

#[test]
fn detector_partitions_cover_odd_non_divisible_source_exactly_once() {
    let mut visits = vec![0_u8; 3 * 5];
    for target_row in 0..2 {
        for target_column in 0..2 {
            for row in axis_partition(target_row, 3, 2) {
                for column in axis_partition(target_column, 5, 2) {
                    visits[row * 5 + column] += 1;
                }
            }
        }
    }
    assert_eq!(visits, vec![1; 15]);
}

#[test]
fn detector_rules_apply_their_declared_means() {
    let field = positive_field();
    let complex = reduce_for_view(
        &field,
        certificate(),
        Observable::Real,
        0.0,
        2,
        2,
        ReductionRule::DetectorComplexMean,
    )
    .unwrap();
    assert_values_close(&values(&complex), &[4.5, 7.0, 12.0, 14.5]);

    let scalar = reduce_for_view(
        &field,
        certificate(),
        Observable::Amplitude,
        0.0,
        2,
        2,
        ReductionRule::DetectorScalarAreaMean,
    )
    .unwrap();
    assert_values_close(&values(&scalar), &[4.5, 7.0, 12.0, 14.5]);

    let squared = reduce_for_view(
        &field,
        certificate(),
        Observable::MagnitudeSquared,
        0.0,
        2,
        2,
        ReductionRule::DetectorMagnitudeSquaredAreaMean,
    )
    .unwrap();
    assert_values_close(
        &values(&squared),
        &[163.0 / 6.0, 222.0 / 4.0, 434.0 / 3.0, 421.0 / 2.0],
    );
}

#[test]
fn complex_mean_resolves_phase_wrap_without_averaging_angles() {
    let delta = 1.0e-6;
    let phases = [PI - delta, -PI + delta];
    let field = HostPhasorField::from_test_components(
        1,
        2,
        phases.iter().map(|phase| phase.cos()).collect(),
        phases.iter().map(|phase| phase.sin()).collect(),
    );
    let projection = reduce_for_view(
        &field,
        certificate(),
        Observable::Phase,
        0.0,
        1,
        1,
        ReductionRule::DetectorComplexMean,
    )
    .unwrap();

    assert_eq!(projection.samples(), [ScalarSample::Value(-PI)]);
}

#[test]
fn complex_cancellation_masks_detector_phase() {
    let field = HostPhasorField::from_test_components(
        1,
        4,
        vec![1.0, -1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, -1.0],
    );
    let projection = reduce_for_view(
        &field,
        certificate(),
        Observable::Phase,
        0.0,
        1,
        1,
        ReductionRule::DetectorComplexMean,
    )
    .unwrap();

    assert_eq!(projection.samples(), [ScalarSample::Masked]);
    assert_eq!(projection.certificate().mask_count(), 1);
}

#[test]
fn non_divisible_detector_footprints_preserve_constant_fields_exactly() {
    let field = HostPhasorField::from_test_components(5, 7, vec![-2.0; 35], vec![3.0; 35]);
    for (observable, rule, expected) in [
        (Observable::Real, ReductionRule::DetectorComplexMean, -2.0),
        (
            Observable::Amplitude,
            ReductionRule::DetectorScalarAreaMean,
            13.0_f64.sqrt(),
        ),
        (
            Observable::MagnitudeSquared,
            ReductionRule::DetectorMagnitudeSquaredAreaMean,
            13.0,
        ),
    ] {
        let projection =
            reduce_for_view(&field, certificate(), observable, 0.0, 2, 3, rule).unwrap();
        assert!(
            values(&projection)
                .iter()
                .all(|value| value.to_bits() == expected.to_bits())
        );
        let footprint = projection.certificate().footprint();
        assert_eq!(
            (
                footprint.min_rows(),
                footprint.max_rows(),
                footprint.min_columns(),
                footprint.max_columns(),
            ),
            (2, 3, 2, 3)
        );
    }
}

#[test]
fn wrapped_phase_cannot_enter_scalar_area_mean() {
    let error = reduce_for_view(
        &positive_field(),
        certificate(),
        Observable::Phase,
        0.0,
        1,
        1,
        ReductionRule::DetectorScalarAreaMean,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ProjectionError::IncompatibleReduction {
            observable: Observable::Phase,
            rule: ReductionRule::DetectorScalarAreaMean,
        }
    );
}

#[test]
fn projection_certificate_retains_dimensions_footprint_rule_loss_sampling_and_masks() {
    let source_certificate = certificate();
    let field = HostPhasorField::from_test_components(1, 2, vec![1.0, -1.0], vec![0.0, 0.0]);
    let projection = reduce_for_view(
        &field,
        source_certificate,
        Observable::Phase,
        0.0,
        1,
        1,
        ReductionRule::DetectorComplexMean,
    )
    .unwrap();
    let certificate = projection.certificate();

    assert_eq!(
        certificate.source_dimensions(),
        GridDimensions {
            rows: 1,
            columns: 2
        }
    );
    assert_eq!(
        certificate.target_dimensions(),
        GridDimensions {
            rows: 1,
            columns: 1
        }
    );
    assert_eq!(certificate.footprint().min_rows(), 1);
    assert_eq!(certificate.footprint().max_rows(), 1);
    assert_eq!(certificate.footprint().min_columns(), 2);
    assert_eq!(certificate.footprint().max_columns(), 2);
    assert_eq!(certificate.rule(), ReductionRule::DetectorComplexMean);
    assert_eq!(certificate.loss_class(), LossClass::DetectorIntegration);
    assert_eq!(
        certificate.source_sampling_certificate(),
        source_certificate
    );
    assert_eq!(certificate.mask_count(), 1);
    assert_eq!(projection.samples(), [ScalarSample::Masked]);
}
