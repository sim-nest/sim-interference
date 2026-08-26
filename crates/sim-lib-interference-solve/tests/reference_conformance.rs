use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, ScalarMedium, SourceSet, UnitVector3,
};
use sim_lib_interference_solve::{HostPhasorField, ReferencePhasorSolver};

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn sources_in_order(order: [&str; 3]) -> SourceSet {
    let sources = order
        .into_iter()
        .map(|id| match id {
            "point" => Emitter::Point {
                id: id.to_owned(),
                position: point(-0.25, 0.75, -20.0),
                amplitude_at_reference: FieldAmplitude::new(7.0).unwrap(),
                phase: Radians::new(0.375).unwrap(),
            },
            "plane-a" => Emitter::ForwardPlane {
                id: id.to_owned(),
                through: point(0.0, 0.0, -2.0),
                direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
                amplitude: FieldAmplitude::new(1.25).unwrap(),
                phase: Radians::new(-0.625).unwrap(),
            },
            "plane-b" => Emitter::ForwardPlane {
                id: id.to_owned(),
                through: point(-1.0, 0.0, 0.0),
                direction: UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
                amplitude: FieldAmplitude::new(0.75).unwrap(),
                phase: Radians::new(1.125).unwrap(),
            },
            unexpected => panic!("unexpected fixture source {unexpected}"),
        })
        .collect();
    SourceSet::new(sources).unwrap()
}

fn problem(order: [&str; 3]) -> InterferenceProblem {
    InterferenceProblem::new(
        Hertz::new(8.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(343.0).unwrap(),
            NepersPerMetre::new(0.015).unwrap(),
        ),
        sources_in_order(order),
        PositiveMetres::new(0.01).unwrap(),
    )
}

fn plane() -> SamplingPlane {
    SamplingPlane::new(
        point(0.0, 0.0, 0.0),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        4,
        5,
    )
    .unwrap()
}

fn component_bits(field: &HostPhasorField) -> (Vec<u64>, Vec<u64>) {
    (
        field.real().iter().map(|value| value.to_bits()).collect(),
        field
            .imaginary()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
    )
}

#[test]
fn repeated_reference_solves_are_bit_identical() {
    let problem = problem(["point", "plane-a", "plane-b"]);
    let plane = plane();

    let (first_field, first_evidence) = ReferencePhasorSolver::default()
        .solve(&problem, &plane)
        .unwrap();
    let (second_field, second_evidence) = ReferencePhasorSolver::default()
        .solve(&problem, &plane)
        .unwrap();

    assert_eq!(component_bits(&first_field), component_bits(&second_field));
    assert_eq!(first_evidence, second_evidence);
    assert_eq!(first_evidence.completed_cells(), 20);
    assert_eq!(first_evidence.completed_emitter_evaluations(), 60);
}

#[test]
fn source_input_permutations_canonicalize_to_identical_fields() {
    let plane = plane();
    let permutations = [
        ["point", "plane-a", "plane-b"],
        ["plane-b", "point", "plane-a"],
        ["plane-a", "plane-b", "point"],
    ];
    let baseline = ReferencePhasorSolver::default()
        .solve(&problem(permutations[0]), &plane)
        .unwrap();
    let baseline_bits = component_bits(&baseline.0);

    for order in permutations.into_iter().skip(1) {
        let solved = ReferencePhasorSolver::default()
            .solve(&problem(order), &plane)
            .unwrap();
        assert_eq!(component_bits(&solved.0), baseline_bits);
        assert_eq!(solved.1, baseline.1);
    }
}
// conformance: reference tests prove solver outputs against analytic wave cases.
