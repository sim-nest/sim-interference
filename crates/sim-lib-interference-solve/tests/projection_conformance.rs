use std::f64::consts::PI;

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, ScalarMedium, SourceSet, UnitVector3,
};
use sim_lib_interference_solve::{
    LossClass, Observable, ReductionRule, ReferencePhasorSolver, ScalarSample, project,
    reduce_for_view,
};

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn plane(rows: usize, columns: usize) -> SamplingPlane {
    SamplingPlane::new(
        point(0.0, 0.0, 0.0),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        rows,
        columns,
    )
    .unwrap()
}

fn problem(phases: &[f64]) -> InterferenceProblem {
    let sources = phases
        .iter()
        .enumerate()
        .map(|(index, phase)| Emitter::ForwardPlane {
            id: format!("plane-{index}"),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
            amplitude: FieldAmplitude::new(2.0).unwrap(),
            phase: Radians::new(*phase).unwrap(),
        })
        .collect();
    InterferenceProblem::new(
        Hertz::new(1.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(100.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(sources).unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    )
}

#[test]
fn public_projection_preserves_source_evidence_and_constant_fields() {
    let (field, solve_evidence) = ReferencePhasorSolver::default()
        .solve(&problem(&[0.0]), &plane(3, 5))
        .unwrap();
    let sampling = solve_evidence.preflight().sampling_certificate;
    let projection = reduce_for_view(
        &field,
        sampling,
        Observable::Amplitude,
        0.0,
        2,
        2,
        ReductionRule::DetectorScalarAreaMean,
    )
    .unwrap();

    assert_eq!(projection.rows(), 2);
    assert_eq!(projection.columns(), 2);
    assert_eq!(projection.samples(), [ScalarSample::Value(2.0); 4]);
    let certificate = projection.certificate();
    assert_eq!(certificate.source_dimensions().rows(), 3);
    assert_eq!(certificate.source_dimensions().columns(), 5);
    assert_eq!(certificate.target_dimensions().rows(), 2);
    assert_eq!(certificate.target_dimensions().columns(), 2);
    assert_eq!(certificate.footprint().min_rows(), 1);
    assert_eq!(certificate.footprint().max_rows(), 2);
    assert_eq!(certificate.footprint().min_columns(), 2);
    assert_eq!(certificate.footprint().max_columns(), 3);
    assert_eq!(certificate.loss_class(), LossClass::DetectorIntegration);
    assert_eq!(certificate.source_sampling_certificate(), sampling);

    let real = project(&field, sampling, Observable::Real, 0.0).unwrap();
    let instant = project(&field, sampling, Observable::Instant { wt: 0.0 }, 0.0).unwrap();
    assert_eq!(instant.samples(), real.samples());
}

#[test]
fn public_cancellation_projection_is_masked() {
    let (field, evidence) = ReferencePhasorSolver::default()
        .solve(&problem(&[0.0, PI]), &plane(1, 1))
        .unwrap();
    let projection = project(
        &field,
        evidence.preflight().sampling_certificate,
        Observable::Phase,
        1.0e-12,
    )
    .unwrap();

    assert_eq!(projection.samples(), [ScalarSample::Masked]);
    assert_eq!(projection.certificate().mask_count(), 1);
}
// conformance: projection tests prove field lowering against independent references.
