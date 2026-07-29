use std::f64::consts::LN_2;

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, ScalarMedium, SourceSet, UnitVector3,
    forward_plane_contribution_at,
};

fn point(x: f64) -> Point3M {
    Point3M::from_metres(x, 0.0, 0.0).unwrap()
}

fn main() {
    let problem = InterferenceProblem::new(
        Hertz::new(1.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(4.0).unwrap(),
            NepersPerMetre::new(LN_2).unwrap(),
        ),
        SourceSet::new(vec![Emitter::ForwardPlane {
            id: "attenuated".to_owned(),
            through: point(0.0),
            direction: UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
            amplitude: FieldAmplitude::new(4.0).unwrap(),
            phase: Radians::new(0.0).unwrap(),
        }])
        .unwrap(),
        PositiveMetres::new(0.01).unwrap(),
    );

    for distance in [0.0, 1.0, 2.0] {
        let (real, imaginary) = forward_plane_contribution_at(
            &problem,
            "attenuated",
            point(0.0),
            UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
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
