use sim_lib_interference_core::{
    FieldAmplitude, Hertz, MetresPerSecond, NepersPerMetre, Point3M, PositiveMetres, Radians,
    SamplingPlane, ScalarMedium, UnitVector3,
};
use sim_lib_interference_solve::{
    AperturePolicy, Observable, ReductionRule, ReferencePhasorSolver, ScenarioBuilder,
    ScenarioKind, ScenarioLimits, analyze_fringes, project,
};

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn x_axis() -> UnitVector3 {
    UnitVector3::new(1.0, 0.0, 0.0).unwrap()
}

fn y_axis() -> UnitVector3 {
    UnitVector3::new(0.0, 1.0, 0.0).unwrap()
}

fn builder() -> ScenarioBuilder {
    ScenarioBuilder::new(
        Hertz::new(1.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(8.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        PositiveMetres::new(0.01).unwrap(),
        ScenarioLimits::default(),
    )
}

#[test]
fn public_standing_wave_analysis_keeps_sampling_and_projection_identity() {
    let scenario = builder()
        .counter_propagating_planes(
            point(0.0, 0.0, 0.0),
            x_axis(),
            PositiveMetres::new(20.0).unwrap(),
            FieldAmplitude::new(1.0).unwrap(),
            Radians::new(0.0).unwrap(),
        )
        .unwrap();
    let plane = SamplingPlane::new(
        point(-4.0, -0.05, 0.0),
        x_axis(),
        y_axis(),
        PositiveMetres::new(8.0).unwrap(),
        PositiveMetres::new(0.1).unwrap(),
        1,
        64,
    )
    .unwrap();
    let (field, evidence) = ReferencePhasorSolver::default()
        .solve(scenario.problem(), &plane)
        .unwrap();
    let sampling = evidence.preflight().sampling_certificate;
    let projection = project(&field, sampling, Observable::Amplitude, 0.0).unwrap();
    let report = analyze_fringes(&projection, 1.0e-12).unwrap();

    assert_eq!(report.sampling, sampling);
    assert_eq!(report.projection.observable(), Observable::Amplitude);
    assert_eq!(report.projection.rule(), ReductionRule::Detail);
    assert_eq!(report.projection.target_dimensions().rows(), 1);
    assert_eq!(report.projection.target_dimensions().columns(), 64);
    assert_eq!(report.stats.count, 64);
    assert_eq!(report.extrema.len(), 4);
    assert!(report.michelson_contrast.is_some());
}

#[test]
fn public_aperture_certificate_quantifies_normalization_and_spacing() {
    let aperture = builder()
        .discrete_aperture(
            point(0.0, 0.0, 0.0),
            x_axis(),
            y_axis(),
            2,
            3,
            PositiveMetres::new(2.0).unwrap(),
            PositiveMetres::new(4.0).unwrap(),
            FieldAmplitude::new(12.0).unwrap(),
            Radians::new(0.0).unwrap(),
            AperturePolicy::Strict,
        )
        .unwrap();
    let certificate = aperture.certificate();

    assert_eq!(certificate.kind, ScenarioKind::DiscreteAperture);
    assert_eq!(certificate.source_count, 6);
    assert_eq!(certificate.total_source_amplitude, 12.0);
    assert_eq!(certificate.amplitude_per_source, 2.0);
    assert_eq!(certificate.element_spacing_wavelengths.u, Some(0.25));
    assert_eq!(certificate.element_spacing_wavelengths.v, Some(0.5));
    assert_eq!(certificate.element_spacing_wavelengths.maximum(), Some(0.5));
}
// conformance: analysis scenarios prove solver composition and retained evidence.
