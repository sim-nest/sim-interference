//! End-to-end conformance for sampling and work preflight.

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceError, InterferenceProblem, MetresPerSecond,
    NepersPerMetre, Point3M, PositiveMetres, Radians, RequestPreflight, SamplingPlane,
    SamplingPolicy, SamplingThresholds, SamplingVerdict, ScalarMedium, SourceSet, UnitVector3,
    WorkBudget, WorkEstimate, WorkMetric,
};

fn forward_plane() -> Emitter {
    Emitter::ForwardPlane {
        id: "plane".to_owned(),
        through: Point3M::from_metres(0.0, 0.0, 0.0).unwrap(),
        direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
        amplitude: FieldAmplitude::new(1.0).unwrap(),
        phase: Radians::new(0.0).unwrap(),
    }
}

fn point(z_m: f64) -> Emitter {
    Emitter::Point {
        id: "point".to_owned(),
        position: Point3M::from_metres(0.5, 0.5, z_m).unwrap(),
        amplitude_at_reference: FieldAmplitude::new(1.0).unwrap(),
        phase: Radians::new(0.0).unwrap(),
    }
}

fn air_problem(frequency_hz: f64, emitter: Emitter) -> InterferenceProblem {
    InterferenceProblem::new(
        Hertz::new(frequency_hz).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(343.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(vec![emitter]).unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    )
}

fn plane(extent_m: f64, cells_per_axis: usize) -> Result<SamplingPlane, InterferenceError> {
    SamplingPlane::new(
        Point3M::from_metres(0.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        PositiveMetres::new(extent_m).unwrap(),
        PositiveMetres::new(extent_m).unwrap(),
        cells_per_axis,
        cells_per_axis,
    )
}

fn strict_preflight(
    problem: &InterferenceProblem,
    plane: &SamplingPlane,
) -> Result<RequestPreflight, InterferenceError> {
    RequestPreflight::admit(
        problem,
        plane,
        SamplingPolicy::Strict,
        SamplingThresholds::default(),
        WorkBudget::default(),
    )
}

#[test]
fn forty_kilohertz_air_on_two_metres_at_two_hundred_cells_refuses() {
    let error = strict_preflight(
        &air_problem(40_000.0, forward_plane()),
        &plane(2.0, 200).unwrap(),
    )
    .unwrap_err();
    let InterferenceError::SamplingRefused { certificate } = error else {
        panic!("expected strict sampling refusal");
    };

    assert_eq!(certificate.verdict, SamplingVerdict::Aliased);
    let expected_carrier_samples = (343.0 / 40_000.0) / (2.0 / 200.0);
    assert_eq!(
        certificate.samples_per_wavelength_u,
        expected_carrier_samples
    );
    assert_eq!(
        certificate.samples_per_power_fringe_u,
        expected_carrier_samples / 2.0
    );
}

#[test]
fn forty_kilohertz_air_on_a_resolved_grid_passes() {
    let preflight = strict_preflight(
        &air_problem(40_000.0, forward_plane()),
        &plane(2.0, 2_048).unwrap(),
    )
    .unwrap();

    assert_eq!(
        preflight.sampling_certificate.verdict,
        SamplingVerdict::Resolved
    );
    assert!(preflight.sampling_certificate.samples_per_wavelength_u > 8.0);
    assert_eq!(preflight.work_estimate.cells, 2_048 * 2_048);
}

#[test]
fn near_source_envelope_under_resolution_refuses_despite_carrier_resolution() {
    let error = strict_preflight(
        &air_problem(1_000.0, point(0.01)),
        &plane(1.0, 256).unwrap(),
    )
    .unwrap_err();
    let InterferenceError::SamplingRefused { certificate } = error else {
        panic!("expected envelope sampling refusal");
    };

    assert!(certificate.samples_per_wavelength_u > 8.0);
    assert!(certificate.max_envelope_fraction_per_cell > 0.10);
    assert_eq!(certificate.verdict, SamplingVerdict::Aliased);
}

#[test]
fn arithmetic_overflow_is_refused_by_allocation_free_constructors() {
    assert_eq!(
        SamplingPlane::new(
            Point3M::from_metres(0.0, 0.0, 0.0).unwrap(),
            UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
            UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
            PositiveMetres::new(1.0).unwrap(),
            PositiveMetres::new(1.0).unwrap(),
            usize::MAX,
            2,
        ),
        Err(InterferenceError::SamplingCellCountOverflow {
            rows: usize::MAX,
            columns: 2,
        })
    );
    assert_eq!(
        WorkEstimate::new(u64::MAX, 2),
        Err(InterferenceError::WorkEstimateOverflow {
            metric: WorkMetric::EmitterEvaluations,
        })
    );
}
// conformance: sampling acceptance tests prove strict and evidence-preserving policies.
