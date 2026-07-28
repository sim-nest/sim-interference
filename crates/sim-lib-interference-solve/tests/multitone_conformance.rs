use std::f64::consts::TAU;

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, ScalarMedium, SourceSet, UnitVector3,
};
use sim_lib_interference_solve::{
    MultiToneStudy, Observable, ReferencePhasorSolver, ScalarSample, ToneCombination, ToneStudy,
    project,
};

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn plane() -> SamplingPlane {
    SamplingPlane::new(
        point(0.0, 0.0, 0.0),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        PositiveMetres::new(0.1).unwrap(),
        PositiveMetres::new(0.1).unwrap(),
        1,
        1,
    )
    .unwrap()
}

fn tone(frequency_hz: f64, phase: f64) -> ToneStudy {
    let problem = InterferenceProblem::new(
        Hertz::new(frequency_hz).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(1_000.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(vec![Emitter::ForwardPlane {
            id: "plane".to_owned(),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
            amplitude: FieldAmplitude::new(1.0).unwrap(),
            phase: Radians::new(phase).unwrap(),
        }])
        .unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    );
    ToneStudy::solve(problem, plane(), 1.0, ReferencePhasorSolver::default()).unwrap()
}

#[test]
fn one_tone_composition_is_identical_to_coherent_projection() {
    let study = MultiToneStudy::new(vec![tone(100.0, 0.375)]).unwrap();
    let component = &study.tones()[0];
    let sampling = component.sampling_certificate();

    let expected_power = project(
        component.field(),
        sampling,
        Observable::MagnitudeSquared,
        0.0,
    )
    .unwrap();
    let actual_power = study
        .combine(ToneCombination::IncoherentMagnitudeSquared)
        .unwrap();
    assert_eq!(
        expected_power.samples(),
        &[ScalarSample::Value(actual_power.samples()[0])]
    );

    let seconds = 0.00125;
    let wt = TAU * component.frequency().get() * seconds;
    let expected_instant =
        project(component.field(), sampling, Observable::Instant { wt }, 0.0).unwrap();
    let actual_instant = study.combine(ToneCombination::Instant { seconds }).unwrap();
    assert_eq!(
        expected_instant.samples(),
        &[ScalarSample::Value(actual_instant.samples()[0])]
    );
}

#[test]
fn two_tone_beat_envelope_period_is_within_one_temporal_sample() {
    let low_hz = 440.0;
    let high_hz = 444.0;
    let study = MultiToneStudy::new(vec![tone(high_hz, 0.0), tone(low_hz, 0.0)]).unwrap();
    let expected_period = 1.0 / (high_hz - low_hz);
    let sample_rate_hz = 8.0 * study.sampling_requirements().highest_frequency().get();
    let sample_step_seconds = 1.0 / sample_rate_hz;
    let first_peak = envelope_peak_near(
        &study,
        expected_period,
        expected_period / 8.0,
        sample_step_seconds,
    );
    let second_peak = envelope_peak_near(
        &study,
        2.0 * expected_period,
        expected_period / 8.0,
        sample_step_seconds,
    );
    let measured_period = second_peak - first_peak;
    let error = (measured_period - expected_period).abs();

    println!(
        "f1={low_hz:.3}Hz f2={high_hz:.3}Hz expected_beat_period={expected_period:.6}s \
         measured_envelope_period={measured_period:.6}s sample_step={sample_step_seconds:.9}s"
    );
    assert!(
        error <= sample_step_seconds,
        "beat-period error {error} exceeds one temporal sample {sample_step_seconds}"
    );
}

fn envelope_peak_near(
    study: &MultiToneStudy,
    centre_seconds: f64,
    half_window_seconds: f64,
    step_seconds: f64,
) -> f64 {
    let first = ((centre_seconds - half_window_seconds) / step_seconds).floor() as usize;
    let last = ((centre_seconds + half_window_seconds) / step_seconds).ceil() as usize;
    let mut best_index = first;
    let mut best_magnitude = f64::NEG_INFINITY;
    for index in first..=last {
        let seconds = index as f64 * step_seconds;
        let magnitude = study
            .combine(ToneCombination::Instant { seconds })
            .unwrap()
            .cell(0, 0)
            .unwrap()
            .abs();
        if magnitude > best_magnitude {
            best_magnitude = magnitude;
            best_index = index;
        }
    }
    best_index as f64 * step_seconds
}
