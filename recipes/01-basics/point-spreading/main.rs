use sim_lib_interference_core::{
    FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, ScalarMedium, SourceSet, point_contribution_at, Emitter,
};

fn point(x: f64) -> Point3M {
    Point3M::from_metres(x, 0.0, 0.0).unwrap()
}

fn main() {
    let problem = InterferenceProblem::new(
        Hertz::new(1.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(4.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(vec![Emitter::Point {
            id: "point".to_owned(),
            position: point(0.0),
            amplitude_at_reference: FieldAmplitude::new(4.0).unwrap(),
            phase: Radians::new(0.0).unwrap(),
        }])
        .unwrap(),
        PositiveMetres::new(0.01).unwrap(),
    );

    for distance in [1.0, 2.0, 4.0] {
        let (real, imaginary) = point_contribution_at(
            &problem,
            "point",
            point(0.0),
            FieldAmplitude::new(4.0).unwrap(),
            Radians::new(0.0).unwrap(),
            point(distance),
        )
        .unwrap();
        println!(
            "distance={distance:.3}m amplitude={:.6}",
            real.hypot(imaginary)
        );
    }
}
