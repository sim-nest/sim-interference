use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, MetresPerSecond, NepersPerMetre, Point3M, PositiveMetres,
    Radians, ScalarMedium, UnitVector3,
};

use super::{
    AperturePolicy, STRICT_MAX_SPACING_WAVELENGTHS, ScenarioBuilder, ScenarioError, ScenarioKind,
    ScenarioLimits,
};

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn builder(limits: ScenarioLimits) -> ScenarioBuilder {
    ScenarioBuilder::new(
        Hertz::new(2.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(4.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        PositiveMetres::new(0.01).unwrap(),
        limits,
    )
}

fn x_axis() -> UnitVector3 {
    UnitVector3::new(1.0, 0.0, 0.0).unwrap()
}

fn y_axis() -> UnitVector3 {
    UnitVector3::new(0.0, 1.0, 0.0).unwrap()
}

#[test]
fn two_point_and_counter_plane_builders_are_canonical_and_named() {
    let two_point = builder(ScenarioLimits::default())
        .two_point(
            point(-1.0, 0.0, 0.0),
            point(1.0, 0.0, 0.0),
            FieldAmplitude::new(2.0).unwrap(),
            Radians::new(std::f64::consts::PI).unwrap(),
        )
        .unwrap();
    assert_eq!(two_point.certificate().kind, ScenarioKind::TwoPoint);
    assert_eq!(two_point.certificate().source_count, 2);
    assert_eq!(
        two_point
            .problem()
            .sources
            .iter()
            .map(Emitter::id)
            .collect::<Vec<_>>(),
        ["scenario/two-point/0", "scenario/two-point/1"]
    );

    let planes = builder(ScenarioLimits::default())
        .counter_propagating_planes(
            point(0.0, 0.0, 0.0),
            x_axis(),
            PositiveMetres::new(4.0).unwrap(),
            FieldAmplitude::new(1.0).unwrap(),
            Radians::new(0.5).unwrap(),
        )
        .unwrap();
    assert_eq!(
        planes.certificate().kind,
        ScenarioKind::CounterPropagatingPlanes
    );
    let sources = planes.problem().sources.as_slice();
    let Emitter::ForwardPlane {
        through: first,
        direction: first_direction,
        ..
    } = &sources[0]
    else {
        panic!("first source must be a plane");
    };
    let Emitter::ForwardPlane {
        through: second,
        direction: second_direction,
        ..
    } = &sources[1]
    else {
        panic!("second source must be a plane");
    };
    assert_eq!(first.coordinates_metres(), [-2.0, 0.0, 0.0]);
    assert_eq!(second.coordinates_metres(), [2.0, 0.0, 0.0]);
    assert_eq!(first_direction.components(), [1.0, 0.0, 0.0]);
    assert_eq!(second_direction.components(), [-1.0, 0.0, 0.0]);
}

#[test]
fn array_and_aperture_normalize_total_amplitude_and_report_spacing() {
    let array = builder(ScenarioLimits::default())
        .phased_array(
            point(0.0, 0.0, 0.0),
            x_axis(),
            4,
            PositiveMetres::new(1.0).unwrap(),
            FieldAmplitude::new(8.0).unwrap(),
            Radians::new(0.0).unwrap(),
            Radians::new(0.25).unwrap(),
            AperturePolicy::Strict,
        )
        .unwrap();
    let certificate = array.certificate();
    assert_eq!(certificate.kind, ScenarioKind::PhasedArray);
    assert_eq!(certificate.total_source_amplitude, 8.0);
    assert_eq!(certificate.amplitude_per_source, 2.0);
    assert_eq!(certificate.element_spacing_wavelengths.u, Some(0.5));
    assert_eq!(certificate.element_spacing_wavelengths.v, None);
    assert!(array.problem().sources.iter().all(|source| {
        matches!(
            source,
            Emitter::Point {
                amplitude_at_reference,
                ..
            } if amplitude_at_reference.get() == 2.0
        )
    }));

    let aperture = builder(ScenarioLimits::default())
        .discrete_aperture(
            point(0.0, 0.0, 0.0),
            x_axis(),
            y_axis(),
            2,
            3,
            PositiveMetres::new(0.5).unwrap(),
            PositiveMetres::new(1.0).unwrap(),
            FieldAmplitude::new(12.0).unwrap(),
            Radians::new(0.0).unwrap(),
            AperturePolicy::Strict,
        )
        .unwrap();
    let certificate = aperture.certificate();
    assert_eq!(certificate.kind, ScenarioKind::DiscreteAperture);
    assert_eq!(certificate.source_count, 6);
    assert_eq!(certificate.amplitude_per_source, 2.0);
    assert_eq!(certificate.element_spacing_wavelengths.u, Some(0.25));
    assert_eq!(certificate.element_spacing_wavelengths.v, Some(0.5));
    assert_eq!(
        aperture
            .problem()
            .sources
            .iter()
            .map(Emitter::id)
            .collect::<Vec<_>>(),
        [
            "scenario/aperture/0",
            "scenario/aperture/1",
            "scenario/aperture/2",
            "scenario/aperture/3",
            "scenario/aperture/4",
            "scenario/aperture/5",
        ]
    );
}

#[test]
fn strict_spacing_refuses_sparse_elements_while_annotate_preserves_truth() {
    let strict = builder(ScenarioLimits::default()).phased_array(
        point(0.0, 0.0, 0.0),
        x_axis(),
        2,
        PositiveMetres::new(1.1).unwrap(),
        FieldAmplitude::new(1.0).unwrap(),
        Radians::new(0.0).unwrap(),
        Radians::new(0.0).unwrap(),
        AperturePolicy::Strict,
    );
    assert_eq!(
        strict,
        Err(ScenarioError::SparseAperture {
            axis: "u",
            spacing_wavelengths: 0.55,
            maximum_wavelengths: STRICT_MAX_SPACING_WAVELENGTHS,
        })
    );

    let annotated = builder(ScenarioLimits::default())
        .phased_array(
            point(0.0, 0.0, 0.0),
            x_axis(),
            2,
            PositiveMetres::new(1.1).unwrap(),
            FieldAmplitude::new(1.0).unwrap(),
            Radians::new(0.0).unwrap(),
            Radians::new(0.0).unwrap(),
            AperturePolicy::Annotate,
        )
        .unwrap();
    assert_eq!(
        annotated.certificate().element_spacing_wavelengths.u,
        Some(0.55)
    );
    assert_eq!(
        annotated.certificate().aperture_policy,
        Some(AperturePolicy::Annotate)
    );
}

#[test]
fn count_and_generated_identity_limits_fail_before_construction() {
    let count_limits = ScenarioLimits::new(4, 96, 1_024).unwrap();
    assert_eq!(
        builder(count_limits).discrete_aperture(
            point(0.0, 0.0, 0.0),
            x_axis(),
            y_axis(),
            2,
            3,
            PositiveMetres::new(0.1).unwrap(),
            PositiveMetres::new(0.1).unwrap(),
            FieldAmplitude::new(1.0).unwrap(),
            Radians::new(0.0).unwrap(),
            AperturePolicy::Strict,
        ),
        Err(ScenarioError::SourceLimitExceeded {
            requested: 6,
            limit: 4,
        })
    );

    let id_limits = ScenarioLimits::new(8, 19, 1_024).unwrap();
    assert_eq!(
        builder(id_limits).two_point(
            point(0.0, 0.0, 0.0),
            point(1.0, 0.0, 0.0),
            FieldAmplitude::new(1.0).unwrap(),
            Radians::new(0.0).unwrap(),
        ),
        Err(ScenarioError::GeneratedIdLimitExceeded {
            requested: 20,
            limit: 19,
        })
    );

    let total_limits = ScenarioLimits::new(8, 96, 39).unwrap();
    assert_eq!(
        builder(total_limits).two_point(
            point(0.0, 0.0, 0.0),
            point(1.0, 0.0, 0.0),
            FieldAmplitude::new(1.0).unwrap(),
            Radians::new(0.0).unwrap(),
        ),
        Err(ScenarioError::TotalIdLimitExceeded {
            requested: 40,
            limit: 39,
        })
    );

    assert_eq!(
        builder(ScenarioLimits::default()).discrete_aperture(
            point(0.0, 0.0, 0.0),
            x_axis(),
            y_axis(),
            usize::MAX,
            2,
            PositiveMetres::new(0.1).unwrap(),
            PositiveMetres::new(0.1).unwrap(),
            FieldAmplitude::new(1.0).unwrap(),
            Radians::new(0.0).unwrap(),
            AperturePolicy::Strict,
        ),
        Err(ScenarioError::SourceCountOverflow {
            rows: usize::MAX,
            columns: 2,
        })
    );
}
