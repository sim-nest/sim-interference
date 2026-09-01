use std::f64::consts::PI;

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceError, InterferenceProblem, MetresPerSecond,
    NepersPerMetre, POINT_SOURCE_REFERENCE_DISTANCE_METRES, Point3M, PositiveMetres, Radians,
    ScalarMedium, SourceSet, UnitVector3,
};

fn point(id: &str) -> Emitter {
    Emitter::Point {
        id: id.to_owned(),
        position: Point3M::from_metres(1.0, 2.0, 3.0).unwrap(),
        amplitude_at_reference: FieldAmplitude::new(2.5).unwrap(),
        phase: Radians::new(0.25).unwrap(),
    }
}

fn forward_plane(id: &str) -> Emitter {
    Emitter::ForwardPlane {
        id: id.to_owned(),
        through: Point3M::from_metres(-1.0, 0.0, 1.0).unwrap(),
        direction: UnitVector3::new(0.0, 0.0, 10.0).unwrap(),
        amplitude: FieldAmplitude::new(1.5).unwrap(),
        phase: Radians::new(-0.5).unwrap(),
    }
}

fn medium() -> ScalarMedium {
    ScalarMedium::new(
        MetresPerSecond::new(4.0).unwrap(),
        NepersPerMetre::new(0.5).unwrap(),
    )
}

#[test]
fn points_admit_only_finite_coordinates() {
    let point = Point3M::from_metres(-1.0, 2.0, -0.0).unwrap();
    assert_eq!(point.coordinates_metres(), [-1.0, 2.0, 0.0]);

    for coordinates in [
        [f64::NAN, 0.0, 0.0],
        [0.0, f64::INFINITY, 0.0],
        [0.0, 0.0, f64::NEG_INFINITY],
    ] {
        assert!(matches!(
            Point3M::from_metres(coordinates[0], coordinates[1], coordinates[2]),
            Err(InterferenceError::InvalidQuantity { .. })
        ));
    }
}

#[test]
fn directions_are_finite_non_zero_and_normalized_without_overflow() {
    let direction = UnitVector3::new(3.0, 4.0, -0.0).unwrap();
    assert_eq!(direction.components(), [0.6, 0.8, 0.0]);

    let large = UnitVector3::new(f64::MAX, f64::MAX, 0.0).unwrap();
    let [x, y, z] = large.components();
    assert!((x.hypot(y).hypot(z) - 1.0).abs() <= f64::EPSILON);

    for components in [
        [0.0, -0.0, 0.0],
        [f64::NAN, 1.0, 0.0],
        [1.0, f64::INFINITY, 0.0],
    ] {
        assert!(matches!(
            UnitVector3::new(components[0], components[1], components[2]),
            Err(InterferenceError::InvalidDirection { .. })
        ));
    }
}

#[test]
fn medium_derives_the_complex_wavenumber_with_the_outgoing_sign() {
    let medium = medium();
    let wave_number = medium.wavenumber(Hertz::new(2.0).unwrap());

    assert_eq!(wave_number.real_radians_per_metre(), PI);
    assert_eq!(wave_number.imaginary_nepers_per_metre(), 0.5);
    assert_eq!(medium.speed().get(), 4.0);
    assert_eq!(medium.attenuation().get(), 0.5);
}

#[test]
fn source_sets_hold_both_canonical_emitter_kinds() {
    let sources = SourceSet::new(vec![forward_plane("plane"), point("point")]).unwrap();

    assert_eq!(sources.len(), 2);
    assert!(!sources.is_empty());
    assert!(matches!(
        &sources.as_slice()[0],
        Emitter::ForwardPlane { .. }
    ));
    assert!(matches!(&sources.as_slice()[1], Emitter::Point { .. }));
}

#[test]
fn source_sets_sort_by_stable_id_and_reject_ambiguous_identity() {
    let sources = SourceSet::new(vec![
        point("z-source"),
        point("a-source"),
        point("m-source"),
    ])
    .unwrap();
    let ids: Vec<_> = sources.iter().map(Emitter::id).collect();
    assert_eq!(ids, ["a-source", "m-source", "z-source"]);

    let permuted = SourceSet::new(vec![
        point("m-source"),
        point("z-source"),
        point("a-source"),
    ])
    .unwrap();
    assert_eq!(sources, permuted);

    assert_eq!(
        SourceSet::new(Vec::new()).unwrap_err(),
        InterferenceError::EmptySourceSet
    );
    assert_eq!(
        SourceSet::new(vec![point("")]).unwrap_err(),
        InterferenceError::EmptySourceId
    );
    assert_eq!(
        SourceSet::new(vec![point("same"), forward_plane("same")]).unwrap_err(),
        InterferenceError::DuplicateSourceId {
            id: "same".to_owned()
        }
    );
}

#[test]
fn one_frequency_is_owned_by_the_complete_problem() {
    let sources = SourceSet::new(vec![point("point"), forward_plane("plane")]).unwrap();
    let problem = InterferenceProblem::new(
        Hertz::new(2.0).unwrap(),
        medium(),
        sources,
        PositiveMetres::new(0.01).unwrap(),
    );

    assert_eq!(problem.frequency.get(), 2.0);
    assert_eq!(problem.sources.len(), 2);
}

#[test]
fn problem_exposes_the_exact_wave_and_reference_distance_convention() {
    let problem = InterferenceProblem::new(
        Hertz::new(2.0).unwrap(),
        medium(),
        SourceSet::new(vec![point("point")]).unwrap(),
        PositiveMetres::new(0.125).unwrap(),
    );

    assert_eq!(problem.angular_frequency_radians_per_second(), 4.0 * PI);
    assert_eq!(problem.wavenumber().real_radians_per_metre(), PI);
    assert_eq!(problem.wavenumber().imaginary_nepers_per_metre(), 0.5);
    assert_eq!(problem.wavelength_metres(), 2.0);
    assert_eq!(problem.singularity_radius.get(), 0.125);
    assert_eq!(POINT_SOURCE_REFERENCE_DISTANCE_METRES, 1.0);

    assert!(PositiveMetres::new(0.0).is_err());
    assert!(PositiveMetres::new(-0.125).is_err());
}
// conformance: model tests prove admitted wave-model invariants and refusal behavior.
