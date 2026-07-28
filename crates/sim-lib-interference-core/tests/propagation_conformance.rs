use std::f64::consts::{FRAC_PI_2, LN_2};

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceError, InterferenceProblem, MetresPerSecond,
    NepersPerMetre, Point3M, PositiveMetres, Radians, ScalarMedium, SourceSet, UnitVector3,
    contribution_at, forward_plane_contribution_at, point_contribution_at,
};

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn problem(attenuation_nepers_per_metre: f64) -> InterferenceProblem {
    let source = Emitter::Point {
        id: "origin".to_owned(),
        position: point(0.0, 0.0, 0.0),
        amplitude_at_reference: FieldAmplitude::new(4.0).unwrap(),
        phase: Radians::new(0.0).unwrap(),
    };
    InterferenceProblem::new(
        Hertz::new(1.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(4.0).unwrap(),
            NepersPerMetre::new(attenuation_nepers_per_metre).unwrap(),
        ),
        SourceSet::new(vec![source]).unwrap(),
        PositiveMetres::new(0.01).unwrap(),
    )
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual:?}, expected {expected:?}, tolerance {tolerance:?}"
    );
}

#[test]
fn point_contribution_has_outgoing_phase_and_inverse_distance_spreading() {
    let problem = problem(0.0);
    let (real, imaginary) = point_contribution_at(
        &problem,
        "point",
        point(0.0, 0.0, 0.0),
        FieldAmplitude::new(4.0).unwrap(),
        Radians::new(0.0).unwrap(),
        point(1.0, 0.0, 0.0),
    )
    .unwrap();

    assert_close(real, 0.0, 8.0 * f64::EPSILON);
    assert_close(imaginary, 4.0, 8.0 * f64::EPSILON);

    let (real, imaginary) = point_contribution_at(
        &problem,
        "point",
        point(0.0, 0.0, 0.0),
        FieldAmplitude::new(4.0).unwrap(),
        Radians::new(0.0).unwrap(),
        point(2.0, 0.0, 0.0),
    )
    .unwrap();
    assert_close(real, -2.0, 8.0 * f64::EPSILON);
    assert_close(imaginary, 0.0, 8.0 * f64::EPSILON);
}

#[test]
fn attenuation_is_the_positive_imaginary_wavenumber_component() {
    let problem = problem(LN_2);
    let (real, imaginary) = forward_plane_contribution_at(
        &problem,
        "plane",
        point(0.0, 0.0, 0.0),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        FieldAmplitude::new(4.0).unwrap(),
        Radians::new(0.0).unwrap(),
        point(1.0, 0.0, 0.0),
    )
    .unwrap();

    assert_close(real, 0.0, 8.0 * f64::EPSILON);
    assert_close(imaginary, 2.0, 8.0 * f64::EPSILON);
}

#[test]
fn singular_and_behind_plane_samples_fail_closed() {
    let problem = problem(0.0);

    assert!(matches!(
        point_contribution_at(
            &problem,
            "point",
            point(0.0, 0.0, 0.0),
            FieldAmplitude::new(1.0).unwrap(),
            Radians::new(0.0).unwrap(),
            point(0.01, 0.0, 0.0),
        ),
        Err(InterferenceError::SingularPointSample { .. })
    ));

    assert_eq!(
        forward_plane_contribution_at(
            &problem,
            "plane",
            point(0.0, 0.0, 0.0),
            UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
            FieldAmplitude::new(1.0).unwrap(),
            Radians::new(0.0).unwrap(),
            point(-0.25, 0.0, 0.0),
        )
        .unwrap_err(),
        InterferenceError::BehindForwardPlane {
            source_id: "plane".to_owned(),
            signed_distance_metres: -0.25,
        }
    );
}

#[test]
fn enum_dispatch_matches_the_focused_plane_function() {
    let problem = problem(0.0);
    let plane = Emitter::ForwardPlane {
        id: "plane".to_owned(),
        through: point(0.0, 0.0, 0.0),
        direction: UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        amplitude: FieldAmplitude::new(3.0).unwrap(),
        phase: Radians::new(FRAC_PI_2).unwrap(),
    };
    let at = point(0.0, 0.0, 0.0);

    assert_eq!(
        contribution_at(&problem, &plane, at).unwrap(),
        forward_plane_contribution_at(
            &problem,
            "plane",
            point(0.0, 0.0, 0.0),
            UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
            FieldAmplitude::new(3.0).unwrap(),
            Radians::new(FRAC_PI_2).unwrap(),
            at,
        )
        .unwrap()
    );
}

#[test]
fn non_finite_derived_geometry_is_rejected() {
    let problem = problem(0.0);
    let result = point_contribution_at(
        &problem,
        "far",
        point(-f64::MAX, 0.0, 0.0),
        FieldAmplitude::new(1.0).unwrap(),
        Radians::new(0.0).unwrap(),
        point(f64::MAX, 0.0, 0.0),
    );

    assert!(matches!(
        result,
        Err(InterferenceError::NonFinitePropagation {
            name: "point-distance-metres",
            ..
        })
    ));
}
