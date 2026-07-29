use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, ScalarMedium, SourceSet, UnitVector3, contribution_at,
};

fn point(x: f64) -> Point3M {
    Point3M::from_metres(x, 0.0, 0.0).unwrap()
}

fn main() {
    let emitter = Emitter::ForwardPlane {
        id: "travelling".to_owned(),
        through: point(0.0),
        direction: UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        amplitude: FieldAmplitude::new(2.0).unwrap(),
        phase: Radians::new(0.0).unwrap(),
    };
    let problem = InterferenceProblem::new(
        Hertz::new(1.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(4.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(vec![emitter]).unwrap(),
        PositiveMetres::new(0.01).unwrap(),
    );
    let emitter = problem.sources.iter().next().unwrap();

    for distance in [0.0, 1.0, 2.0] {
        let (real, imaginary) = contribution_at(&problem, emitter, point(distance)).unwrap();
        println!(
            "distance={distance:.3}m real={real:.6} imaginary={imaginary:.6}"
        );
    }
}
