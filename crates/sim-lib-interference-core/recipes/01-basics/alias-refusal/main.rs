use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceError, InterferenceProblem, MetresPerSecond,
    NepersPerMetre, Point3M, PositiveMetres, Radians, RequestPreflight, SamplingPlane,
    SamplingPolicy, SamplingThresholds, ScalarMedium, SourceSet, UnitVector3, WorkBudget,
};

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn main() {
    let problem = InterferenceProblem::new(
        Hertz::new(40_000.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(343.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(vec![Emitter::ForwardPlane {
            id: "carrier".to_owned(),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
            amplitude: FieldAmplitude::new(1.0).unwrap(),
            phase: Radians::new(0.0).unwrap(),
        }])
        .unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    );
    let plane = SamplingPlane::new(
        point(0.0, 0.0, 0.0),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        PositiveMetres::new(2.0).unwrap(),
        PositiveMetres::new(2.0).unwrap(),
        200,
        200,
    )
    .unwrap();

    let InterferenceError::SamplingRefused { certificate } = RequestPreflight::admit(
        &problem,
        &plane,
        SamplingPolicy::Strict,
        SamplingThresholds::default(),
        WorkBudget::default(),
    )
    .unwrap_err()
    else {
        unreachable!("strict undersampling must return its certificate");
    };
    println!(
        "strict=refused verdict={:?} carrier-samples={:.6} power-fringe-samples={:.6}",
        certificate.verdict,
        certificate.samples_per_wavelength_u,
        certificate.samples_per_power_fringe_u
    );
}
