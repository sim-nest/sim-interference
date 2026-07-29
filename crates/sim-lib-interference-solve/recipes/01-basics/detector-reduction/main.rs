use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, ScalarMedium, SourceSet, UnitVector3,
};
use sim_lib_interference_solve::{
    Observable, ReductionRule, ReferencePhasorSolver, ScalarSample, reduce_for_view,
};

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn main() {
    let problem = InterferenceProblem::new(
        Hertz::new(1.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(100.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(vec![Emitter::ForwardPlane {
            id: "constant".to_owned(),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
            amplitude: FieldAmplitude::new(2.0).unwrap(),
            phase: Radians::new(0.0).unwrap(),
        }])
        .unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    );
    let plane = SamplingPlane::new(
        point(0.0, 0.0, 0.0),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        3,
        5,
    )
    .unwrap();
    let (field, evidence) = ReferencePhasorSolver::default()
        .solve(&problem, &plane)
        .unwrap();
    let projection = reduce_for_view(
        &field,
        evidence.preflight().sampling_certificate,
        Observable::Amplitude,
        0.0,
        2,
        2,
        ReductionRule::DetectorScalarAreaMean,
    )
    .unwrap();
    let certificate = projection.certificate();
    println!(
        "source={}x{} target={}x{} footprint-rows={}..{} footprint-cols={}..{} loss={:?}",
        certificate.source_dimensions().rows(),
        certificate.source_dimensions().columns(),
        certificate.target_dimensions().rows(),
        certificate.target_dimensions().columns(),
        certificate.footprint().min_rows(),
        certificate.footprint().max_rows(),
        certificate.footprint().min_columns(),
        certificate.footprint().max_columns(),
        certificate.loss_class()
    );
    let samples = projection
        .samples()
        .iter()
        .map(|sample| match sample {
            ScalarSample::Value(value) => format!("{value:.6}"),
            ScalarSample::Masked => "masked".to_owned(),
        })
        .collect::<Vec<_>>();
    println!("samples={}", samples.join(","));
}
