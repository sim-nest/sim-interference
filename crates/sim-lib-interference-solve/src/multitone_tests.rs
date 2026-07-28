use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, SamplingVerdict, ScalarMedium, SourceSet, UnitVector3,
};

use super::{MultiToneError, MultiToneStudy, ToneCombination, ToneStudy};
use crate::ReferencePhasorSolver;

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn plane(origin_x: f64) -> SamplingPlane {
    SamplingPlane::new(
        point(origin_x, 0.0, 0.0),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        16,
        16,
    )
    .unwrap()
}

fn problem(frequency_hz: f64, phase: f64) -> InterferenceProblem {
    InterferenceProblem::new(
        Hertz::new(frequency_hz).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(100.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(vec![Emitter::ForwardPlane {
            id: "plane".to_owned(),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
            amplitude: FieldAmplitude::new(2.0).unwrap(),
            phase: Radians::new(phase).unwrap(),
        }])
        .unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    )
}

fn tone(frequency_hz: f64, phase: f64, weight: f64, plane: SamplingPlane) -> ToneStudy {
    ToneStudy::solve(
        problem(frequency_hz, phase),
        plane,
        weight,
        ReferencePhasorSolver::default(),
    )
    .unwrap()
}

#[test]
fn certified_tones_require_positive_weights_and_identical_physical_planes() {
    let shared_plane = plane(0.0);
    let first = tone(100.0, 0.0, 0.25, shared_plane);
    let second = tone(200.0, 0.5, 2.0, shared_plane);
    let study = MultiToneStudy::new(vec![first, second]).unwrap();

    assert_eq!(study.plane(), shared_plane);
    assert_eq!(study.tones().len(), 2);
    assert_eq!(study.tones()[0].problem().frequency.get(), 100.0);
    assert_eq!(study.tones()[0].weight(), 0.25);
    assert_eq!(study.tones()[0].field().rows(), shared_plane.rows());
    assert_eq!(
        study.tones()[0].sampling_certificate().verdict,
        SamplingVerdict::Resolved
    );
    assert_eq!(
        study.tones()[0].evidence().completed_cells(),
        shared_plane.cell_count() as u64
    );

    assert!(matches!(
        ToneStudy::solve(
            problem(100.0, 0.0),
            shared_plane,
            0.0,
            ReferencePhasorSolver::default()
        ),
        Err(MultiToneError::InvalidWeight {
            frequency_hz: 100.0,
            weight: 0.0
        })
    ));
    assert!(matches!(
        ToneStudy::solve(
            problem(100.0, 0.0),
            shared_plane,
            f64::NAN,
            ReferencePhasorSolver::default()
        ),
        Err(MultiToneError::InvalidWeight {
            frequency_hz: 100.0,
            weight
        }) if weight.is_nan()
    ));
    assert_eq!(
        MultiToneStudy::new(Vec::new()),
        Err(MultiToneError::EmptyStudy)
    );

    let mismatched = tone(150.0, 0.0, 1.0, plane(0.125));
    assert!(matches!(
        MultiToneStudy::new(vec![
            tone(100.0, 0.0, 1.0, shared_plane),
            mismatched
        ]),
        Err(MultiToneError::MismatchedPlane {
            frequency_hz: 150.0,
            expected,
            actual
        }) if *expected == shared_plane && *actual == plane(0.125)
    ));
}

#[test]
fn duplicate_frequencies_are_refused_as_one_coherent_problem() {
    let shared_plane = plane(0.0);
    let duplicate = MultiToneStudy::new(vec![
        tone(200.0, 0.0, 1.0, shared_plane),
        tone(100.0, 0.0, 1.0, shared_plane),
        tone(200.0, 1.0, 2.0, shared_plane),
    ]);

    assert_eq!(
        duplicate,
        Err(MultiToneError::DuplicateFrequency {
            frequency_hz: 200.0
        })
    );

    let ordered = MultiToneStudy::new(vec![
        tone(200.0, 0.0, 1.0, shared_plane),
        tone(100.0, 0.0, 1.0, shared_plane),
    ])
    .unwrap();
    assert_eq!(ordered.tones()[0].frequency().get(), 100.0);
    assert_eq!(ordered.tones()[1].frequency().get(), 200.0);
}

#[test]
fn combination_adds_only_incoherent_power_or_instantaneous_scalars() {
    let shared_plane = plane(0.0);
    let study = MultiToneStudy::new(vec![
        tone(100.0, 0.0, 1.0, shared_plane),
        tone(200.0, std::f64::consts::PI, 1.0, shared_plane),
    ])
    .unwrap();

    let incoherent = study
        .combine(ToneCombination::IncoherentMagnitudeSquared)
        .unwrap();
    assert_eq!(incoherent.rows(), 16);
    assert_eq!(incoherent.columns(), 16);
    assert_eq!(incoherent.len(), 256);
    assert!(!incoherent.is_empty());
    assert_eq!(
        incoherent.combination(),
        ToneCombination::IncoherentMagnitudeSquared
    );
    assert!(incoherent.samples().iter().all(|value| *value == 8.0));
    assert_eq!(incoherent.cell(0, 0), Some(8.0));
    assert_eq!(incoherent.cell(16, 0), None);

    let instant = study
        .combine(ToneCombination::Instant { seconds: 0.0 })
        .unwrap();
    assert!(instant.samples().iter().all(|value| *value == 0.0));
    assert!(matches!(
        study.combine(ToneCombination::Instant { seconds: f64::NAN }),
        Err(MultiToneError::InvalidSeconds { seconds }) if seconds.is_nan()
    ));
}

#[test]
fn highest_frequency_defines_set_requirements_and_all_certificates_survive() {
    let shared_plane = plane(0.0);
    let study = MultiToneStudy::new(vec![
        tone(200.0, 0.5, 2.0, shared_plane),
        tone(100.0, 0.0, 0.25, shared_plane),
    ])
    .unwrap();

    let requirements = study.sampling_requirements();
    assert_eq!(requirements.highest_frequency().get(), 200.0);
    assert_eq!(requirements.spatial_certificate().wavelength_m, 0.5);
    assert_eq!(requirements.minimum_temporal_samples_per_second(), 400.0);
    assert_eq!(requirements.maximum_temporal_step_seconds(), 0.0025);
    assert_eq!(
        requirements.spatial_certificate(),
        study.tones()[1].sampling_certificate()
    );

    let projection = study
        .combine(ToneCombination::IncoherentMagnitudeSquared)
        .unwrap();
    let certificate = projection.certificate();
    assert_eq!(certificate.plane(), shared_plane);
    assert_eq!(certificate.sampling_requirements(), requirements);
    assert_eq!(certificate.components().len(), 2);
    assert_eq!(certificate.components()[0].frequency().get(), 100.0);
    assert_eq!(certificate.components()[0].weight(), 0.25);
    assert_eq!(
        certificate.components()[0]
            .sampling_certificate()
            .wavelength_m,
        1.0
    );
    assert_eq!(
        certificate.components()[0]
            .solve_evidence()
            .completed_cells(),
        shared_plane.cell_count() as u64
    );
    assert_eq!(certificate.components()[1].frequency().get(), 200.0);
    assert_eq!(certificate.components()[1].weight(), 2.0);
    assert_eq!(
        certificate.components()[1]
            .sampling_certificate()
            .wavelength_m,
        0.5
    );
}
