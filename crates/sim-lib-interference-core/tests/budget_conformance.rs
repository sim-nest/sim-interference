use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceError, InterferenceProblem, MetresPerSecond,
    NepersPerMetre, Point3M, PositiveMetres, Radians, RequestPreflight, SamplingPlane,
    SamplingPolicy, SamplingThresholds, ScalarMedium, SourceSet, UnitVector3, WorkBudget,
    WorkEstimate, WorkMetric,
};

fn problem() -> InterferenceProblem {
    let emitter = Emitter::ForwardPlane {
        id: "plane".to_owned(),
        through: Point3M::from_metres(0.0, 0.0, 0.0).unwrap(),
        direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
        amplitude: FieldAmplitude::new(1.0).unwrap(),
        phase: Radians::new(0.0).unwrap(),
    };
    InterferenceProblem::new(
        Hertz::new(1_000.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(343.0).unwrap(),
            NepersPerMetre::new(0.0).unwrap(),
        ),
        SourceSet::new(vec![emitter]).unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    )
}

fn plane(cells_per_axis: usize) -> SamplingPlane {
    SamplingPlane::new(
        Point3M::from_metres(0.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        cells_per_axis,
        cells_per_axis,
    )
    .unwrap()
}

#[test]
fn estimate_accounts_for_all_required_work_dimensions() {
    assert_eq!(
        WorkEstimate::new(4, 2).unwrap(),
        WorkEstimate {
            cells: 4,
            emitters: 2,
            emitter_evaluations: 8,
            host_bytes: 64,
            result_bytes: 64,
            certificate_stencil_work: 28,
        }
    );
}

#[test]
fn every_budget_field_is_enforced_with_estimate_and_limit() {
    let estimate = WorkEstimate::new(4, 2).unwrap();
    let mut budget = WorkBudget {
        max_cells: estimate.cells,
        max_emitter_evaluations: estimate.emitter_evaluations,
        max_host_bytes: estimate.host_bytes,
        max_result_bytes: estimate.result_bytes,
        max_certificate_stencil_work: estimate.certificate_stencil_work,
    };
    assert_eq!(budget.admit(&estimate), Ok(()));

    budget.max_certificate_stencil_work -= 1;
    assert_eq!(
        budget.admit(&estimate),
        Err(InterferenceError::WorkBudgetExceeded {
            metric: WorkMetric::CertificateStencilWork,
            estimate: 28,
            limit: 27,
        })
    );
}

#[test]
fn preflight_returns_sampling_and_work_evidence_before_solver_storage_exists() {
    let preflight = RequestPreflight::admit(
        &problem(),
        &plane(64),
        SamplingPolicy::Strict,
        SamplingThresholds::default(),
        WorkBudget::default(),
    )
    .unwrap();

    assert_eq!(preflight.work_estimate.cells, 4_096);
    assert_eq!(preflight.work_estimate.emitter_evaluations, 4_096);
    assert_eq!(preflight.work_estimate.result_bytes, 65_536);
}

#[test]
fn checked_estimation_rejects_arithmetic_overflow() {
    assert_eq!(
        WorkEstimate::new(u64::MAX, 2),
        Err(InterferenceError::WorkEstimateOverflow {
            metric: WorkMetric::EmitterEvaluations,
        })
    );
}
