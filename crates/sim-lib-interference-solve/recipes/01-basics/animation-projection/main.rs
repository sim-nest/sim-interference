use std::f64::consts::{FRAC_PI_2, PI};

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, ScalarMedium, SourceSet, UnitVector3,
};
use sim_lib_interference_solve::{
    Observable, ReferencePhasorSolver, ScalarSample, project,
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
            id: "frame-source".to_owned(),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
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
        PositiveMetres::new(0.1).unwrap(),
        PositiveMetres::new(0.1).unwrap(),
        1,
        1,
    )
    .unwrap();
    let (field, evidence) = ReferencePhasorSolver::default()
        .solve(&problem, &plane)
        .unwrap();
    let sampling = evidence.preflight().sampling_certificate;

    for wt in [0.0, FRAC_PI_2, PI] {
        let frame = project(&field, sampling, Observable::Instant { wt }, 0.0).unwrap();
        let ScalarSample::Value(sample) = frame.samples()[0] else {
            unreachable!("instantaneous projection is never masked");
        };
        println!("wt={wt:.6} sample={sample:.6}");
    }
}
