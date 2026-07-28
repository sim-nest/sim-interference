use std::f64::consts::PI;

use sim_lib_interference_core::{SamplingCertificate, SamplingThresholds, SamplingVerdict};

use super::{LossClass, ScalarProjection, ScalarSample, project};
use crate::{HostPhasorField, Observable, ReductionRule};

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

fn field(real: Vec<f64>, imaginary: Vec<f64>) -> HostPhasorField {
    HostPhasorField::from_test_components(1, real.len(), real, imaginary)
}

#[test]
fn every_observable_has_its_declared_scalar_meaning() {
    let field = field(vec![3.0, -1.0], vec![4.0, 0.0]);
    let expected = [
        (
            Observable::Real,
            vec![ScalarSample::Value(3.0), ScalarSample::Value(-1.0)],
        ),
        (
            Observable::Imaginary,
            vec![ScalarSample::Value(4.0), ScalarSample::Value(0.0)],
        ),
        (
            Observable::Amplitude,
            vec![ScalarSample::Value(5.0), ScalarSample::Value(1.0)],
        ),
        (
            Observable::Phase,
            vec![
                ScalarSample::Value(4.0_f64.atan2(3.0)),
                ScalarSample::Value(-PI),
            ],
        ),
        (
            Observable::MagnitudeSquared,
            vec![ScalarSample::Value(25.0), ScalarSample::Value(1.0)],
        ),
        (
            Observable::Instant { wt: PI / 2.0 },
            vec![
                ScalarSample::Value(3.0 * (PI / 2.0).cos() + 4.0),
                ScalarSample::Value(-(PI / 2.0).cos()),
            ],
        ),
    ];

    for (observable, expected_samples) in expected {
        let projection = project(&field, certificate(), observable, 0.0).unwrap();
        assert_eq!(projection.samples(), expected_samples);
        assert_eq!(projection.certificate().observable(), observable);
        assert_eq!(projection.certificate().rule(), ReductionRule::Detail);
        assert_eq!(projection.certificate().loss_class(), LossClass::Lossless);
    }
}

#[test]
fn phase_at_or_below_floor_is_structurally_masked() {
    let field = field(vec![0.0, 0.3, 0.0], vec![0.0, 0.4, 0.6]);
    let projection = project(&field, certificate(), Observable::Phase, 0.5).unwrap();

    assert_eq!(
        projection.samples(),
        [
            ScalarSample::Masked,
            ScalarSample::Masked,
            ScalarSample::Value(PI / 2.0),
        ]
    );
    assert_eq!(projection.certificate().mask_count(), 2);
    assert!(
        !projection
            .samples()
            .iter()
            .any(|sample| matches!(sample, ScalarSample::Value(value) if *value == 0.0))
    );
}

#[test]
fn instant_at_zero_is_bit_identical_to_real() {
    let field = field(vec![-0.0, -3.5, 7.25], vec![9.0, -2.0, 4.0]);
    let real = project(&field, certificate(), Observable::Real, 0.0).unwrap();
    let instant = project(&field, certificate(), Observable::Instant { wt: 0.0 }, 0.0).unwrap();

    let bits = |projection: &ScalarProjection| {
        projection
            .samples()
            .iter()
            .map(|sample| match sample {
                ScalarSample::Value(value) => value.to_bits(),
                ScalarSample::Masked => panic!("real-valued projection cannot be masked"),
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&instant), bits(&real));
}
