use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, ScalarMedium, SourceSet, UnitVector3,
};
use sim_lib_interference_solve::{
    MultiToneStudy, ReferencePhasorSolver, ToneCombination, ToneStudy,
};

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn plane() -> SamplingPlane {
    SamplingPlane::new(
        point(0.0, 0.0, 0.0),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        PositiveMetres::new(0.01).unwrap(),
        PositiveMetres::new(0.01).unwrap(),
        1,
        1,
    )
    .unwrap()
}

fn tone(frequency_hz: f64, plane: SamplingPlane) -> ToneStudy {
    let problem = InterferenceProblem::new(
        Hertz::new(frequency_hz).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(343.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(vec![Emitter::ForwardPlane {
            id: format!("tone-{frequency_hz:.0}"),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
            amplitude: FieldAmplitude::new(1.0).unwrap(),
            phase: Radians::new(0.0).unwrap(),
        }])
        .unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    );
    ToneStudy::solve(problem, plane, 1.0, ReferencePhasorSolver::default()).unwrap()
}

fn main() {
    let plane = plane();
    let study = MultiToneStudy::new(vec![tone(440.0, plane), tone(444.0, plane)]).unwrap();
    for seconds in [0.0, 0.125, 0.25] {
        let frame = study
            .combine(ToneCombination::Instant { seconds })
            .unwrap();
        println!("time={seconds:.3}s sample={:.6}", frame.samples()[0]);
    }
    println!(
        "beat-period={:.3}s components={}",
        1.0 / (444.0 - 440.0),
        study.tones().len()
    );
}
