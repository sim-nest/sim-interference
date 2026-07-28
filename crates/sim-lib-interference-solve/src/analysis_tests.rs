use sim_lib_interference_core::{SamplingCertificate, SamplingThresholds, SamplingVerdict};

use super::{AnalysisError, Extremum, ExtremumKind, analyze_fringes};
use crate::{
    HostPhasorField, LossClass, Observable, ReductionRule, ScalarSample, project, reduce_for_view,
};

fn certificate() -> SamplingCertificate {
    SamplingCertificate {
        thresholds: SamplingThresholds::default(),
        wavelength_m: 2.0,
        samples_per_wavelength_u: 16.0,
        samples_per_wavelength_v: 12.0,
        samples_per_power_fringe_u: 8.0,
        samples_per_power_fringe_v: 6.0,
        nearest_point_source_distance_m: None,
        max_envelope_fraction_per_cell: 0.0,
        verdict: SamplingVerdict::Resolved,
    }
}

#[test]
fn reports_stats_strict_moore_extrema_and_contrast_in_row_major_order() {
    let field = HostPhasorField::from_test_components(
        3,
        3,
        vec![2.0, 3.0, 2.0, 3.0, 0.0, 3.0, 2.0, 3.0, 4.0],
        vec![0.0; 9],
    );
    let projection = project(&field, certificate(), Observable::Amplitude, 0.0).unwrap();
    let report = analyze_fringes(&projection, 1.0e-12).unwrap();

    assert_eq!(report.sampling, certificate());
    assert_eq!(report.projection.observable(), Observable::Amplitude);
    assert_eq!(report.projection.rule(), ReductionRule::Detail);
    assert_eq!(report.projection.loss_class(), LossClass::Lossless);
    assert_eq!(report.projection.target_dimensions().rows(), 3);
    assert_eq!(report.projection.target_dimensions().columns(), 3);
    assert_eq!(report.stats.count, 9);
    assert_eq!(report.stats.minimum, 0.0);
    assert_eq!(report.stats.maximum, 4.0);
    assert!((report.stats.mean - 22.0 / 9.0).abs() < 1.0e-15);
    assert!(report.stats.population_variance > 0.0);
    assert_eq!(
        report.extrema,
        vec![
            Extremum {
                row: 1,
                column: 1,
                value: 0.0,
                kind: ExtremumKind::NodeCandidate,
            },
            Extremum {
                row: 2,
                column: 2,
                value: 4.0,
                kind: ExtremumKind::AntinodeCandidate,
            },
        ]
    );
    assert_eq!(report.michelson_contrast, Some(1.0));
}

#[test]
fn plateaus_have_no_arbitrary_extrema_and_dark_fields_have_no_contrast() {
    let field = HostPhasorField::from_test_components(2, 2, vec![0.25; 4], vec![0.0; 4]);
    let projection = project(&field, certificate(), Observable::Amplitude, 0.0).unwrap();
    let report = analyze_fringes(&projection, 0.25).unwrap();

    assert!(report.extrema.is_empty());
    assert_eq!(report.michelson_contrast, None);
    assert_eq!(report.stats.population_variance, 0.0);
}

#[test]
fn magnitude_squared_uses_amplitude_floor_and_preserves_detector_identity() {
    let field = HostPhasorField::from_test_components(2, 2, vec![0.1, 0.2, 0.3, 0.4], vec![0.0; 4]);
    let projection = reduce_for_view(
        &field,
        certificate(),
        Observable::MagnitudeSquared,
        0.0,
        1,
        1,
        ReductionRule::DetectorMagnitudeSquaredAreaMean,
    )
    .unwrap();
    let report = analyze_fringes(&projection, 1.0).unwrap();

    assert_eq!(report.michelson_contrast, None);
    assert_eq!(
        report.projection.rule(),
        ReductionRule::DetectorMagnitudeSquaredAreaMean
    );
    assert_eq!(report.projection.source_dimensions().rows(), 2);
    assert_eq!(report.projection.target_dimensions().rows(), 1);
}

#[test]
fn non_amplitude_projection_and_invalid_floor_fail_closed() {
    let field = HostPhasorField::from_test_components(1, 2, vec![1.0, 2.0], vec![0.0; 2]);
    let projection = project(&field, certificate(), Observable::Real, 0.0).unwrap();

    assert_eq!(
        analyze_fringes(&projection, 0.0),
        Err(AnalysisError::IncompatibleObservable {
            observable: Observable::Real,
        })
    );
    assert!(matches!(
        analyze_fringes(&projection, f64::NAN),
        Err(AnalysisError::InvalidAmplitudeFloor { value }) if value.is_nan()
    ));
    assert!(!projection.samples().contains(&ScalarSample::Masked));
}
