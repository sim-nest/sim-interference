use std::f64::consts::{PI, TAU};

use sim_lib_interference_core::{
    FieldAmplitude, Hertz, InterferenceError, Metres, MetresPerSecond, NepersPerMetre,
    PositiveMetres, Radians,
};

fn assert_invalid<T>(
    result: Result<T, InterferenceError>,
    expected_name: &'static str,
    expected_value: f64,
) {
    match result {
        Err(InterferenceError::InvalidQuantity { name, value }) => {
            assert_eq!(name, expected_name);
            if expected_value.is_nan() {
                assert!(value.is_nan());
            } else {
                assert_eq!(value, expected_value);
            }
        }
        Err(other) => panic!("expected invalid quantity, got {other:?}"),
        Ok(_) => panic!("invalid quantity was admitted"),
    }
}

#[test]
fn every_quantity_rejects_non_finite_values() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_invalid(Metres::new(value), Metres::NAME, value);
        assert_invalid(PositiveMetres::new(value), PositiveMetres::NAME, value);
        assert_invalid(Hertz::new(value), Hertz::NAME, value);
        assert_invalid(MetresPerSecond::new(value), MetresPerSecond::NAME, value);
        assert_invalid(NepersPerMetre::new(value), NepersPerMetre::NAME, value);
        assert_invalid(Radians::new(value), Radians::NAME, value);
        assert_invalid(FieldAmplitude::new(value), FieldAmplitude::NAME, value);
    }
}

#[test]
fn signed_metres_admit_both_signs_and_canonicalize_zero() {
    assert_eq!(Metres::new(-12.5).unwrap().get(), -12.5);
    assert_eq!(Metres::new(12.5).unwrap().get(), 12.5);
    assert_eq!(Metres::new(-0.0).unwrap().get().to_bits(), 0.0f64.to_bits());
}

#[test]
fn positive_quantities_reject_zero_and_negative_values() {
    for value in [-1.0, -0.0, 0.0] {
        assert_invalid(PositiveMetres::new(value), PositiveMetres::NAME, value);
        assert_invalid(Hertz::new(value), Hertz::NAME, value);
        assert_invalid(MetresPerSecond::new(value), MetresPerSecond::NAME, value);
    }

    assert_eq!(PositiveMetres::new(0.25).unwrap().get(), 0.25);
    assert_eq!(Hertz::new(440.0).unwrap().get(), 440.0);
    assert_eq!(MetresPerSecond::new(343.0).unwrap().get(), 343.0);
}

#[test]
fn attenuation_and_amplitude_admit_zero_but_reject_negative_values() {
    assert_eq!(
        NepersPerMetre::new(-0.0).unwrap().get().to_bits(),
        0.0f64.to_bits()
    );
    assert_eq!(
        FieldAmplitude::new(-0.0).unwrap().get().to_bits(),
        0.0f64.to_bits()
    );
    assert_eq!(NepersPerMetre::new(0.0).unwrap().get(), 0.0);
    assert_eq!(FieldAmplitude::new(0.0).unwrap().get(), 0.0);
    assert_invalid(
        NepersPerMetre::new(-f64::EPSILON),
        NepersPerMetre::NAME,
        -f64::EPSILON,
    );
    assert_invalid(
        FieldAmplitude::new(-f64::EPSILON),
        FieldAmplitude::NAME,
        -f64::EPSILON,
    );
}

#[test]
fn phase_is_normalized_to_the_half_open_principal_interval() {
    let cases = [
        (0.0, 0.0),
        (-0.0, 0.0),
        (PI, -PI),
        (-PI, -PI),
        (TAU, 0.0),
        (-TAU, 0.0),
        (3.0 * PI / 2.0, -PI / 2.0),
        (-3.0 * PI / 2.0, PI / 2.0),
    ];

    for (input, expected) in cases {
        let actual = Radians::new(input).unwrap().get();
        assert!(
            (actual - expected).abs() <= f64::EPSILON,
            "{input} normalized to {actual}, expected {expected}"
        );
        assert!((-PI..PI).contains(&actual));
    }
}

#[test]
fn diagnostic_text_and_quantity_names_are_stable() {
    assert_eq!(
        Hertz::new(f64::NAN).unwrap_err().to_string(),
        "invalid quantity `frequency-hz`: NaN"
    );
    assert_eq!(
        PositiveMetres::new(0.0).unwrap_err().to_string(),
        "invalid quantity `positive-distance-m`: 0.0"
    );
    assert_eq!(
        NepersPerMetre::new(-1.0).unwrap_err().to_string(),
        "invalid quantity `attenuation-np-m`: -1.0"
    );
    assert_eq!(
        Radians::new(f64::INFINITY).unwrap_err().to_string(),
        "invalid quantity `phase-rad`: inf"
    );
}
