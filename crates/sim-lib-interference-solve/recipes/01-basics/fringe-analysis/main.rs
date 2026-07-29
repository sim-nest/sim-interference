use sim_lib_interference_core::{
    FieldAmplitude, Hertz, MetresPerSecond, NepersPerMetre, Point3M, PositiveMetres, Radians,
    SamplingPlane, ScalarMedium, UnitVector3,
};
use sim_lib_interference_solve::{
    AperturePolicy, Observable, ReferencePhasorSolver, ScenarioBuilder, ScenarioLimits,
    analyze_fringes, project,
};

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn main() {
    let x = UnitVector3::new(1.0, 0.0, 0.0).unwrap();
    let y = UnitVector3::new(0.0, 1.0, 0.0).unwrap();
    let builder = ScenarioBuilder::new(
        Hertz::new(1.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(8.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        PositiveMetres::new(0.01).unwrap(),
        ScenarioLimits::default(),
    );
    let standing_wave = builder
        .counter_propagating_planes(
            point(0.0, 0.0, 0.0),
            x,
            PositiveMetres::new(20.0).unwrap(),
            FieldAmplitude::new(1.0).unwrap(),
            Radians::new(0.0).unwrap(),
        )
        .unwrap();
    let plane = SamplingPlane::new(
        point(-4.0, -0.05, 0.0),
        x,
        y,
        PositiveMetres::new(8.0).unwrap(),
        PositiveMetres::new(0.1).unwrap(),
        1,
        64,
    )
    .unwrap();
    let (field, evidence) = ReferencePhasorSolver::default()
        .solve(standing_wave.problem(), &plane)
        .unwrap();
    let sampling = evidence.preflight().sampling_certificate;
    let projection = project(&field, sampling, Observable::Amplitude, 0.0).unwrap();
    let report = analyze_fringes(&projection, 1.0e-12).unwrap();

    println!(
        "standing-wave sources={} sampling={:?}",
        standing_wave.certificate().source_count,
        report.sampling.verdict
    );
    println!(
        "projection={:?}/{:?} shape={}x{}",
        report.projection.observable(),
        report.projection.rule(),
        report.projection.target_dimensions().rows(),
        report.projection.target_dimensions().columns()
    );
    println!(
        "analysis cells={} extrema={} contrast-defined={}",
        report.stats.count,
        report.extrema.len(),
        report.michelson_contrast.is_some()
    );

    let aperture = builder
        .discrete_aperture(
            point(0.0, 0.0, 0.0),
            x,
            y,
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
    println!(
        "aperture elements={} amplitude-each={} spacing={:?}/{:?} wavelengths",
        certificate.source_count,
        certificate.amplitude_per_source,
        certificate.element_spacing_wavelengths.u,
        certificate.element_spacing_wavelengths.v
    );
}
