//! Differential and hardware evidence over the published compute providers.

use std::sync::Arc;

use sim_kernel::{
    Consistency, Cx, DefaultFactory, EagerPolicy, Error, EvalFabric, EvalMode, EvalRequest, Expr,
    Symbol,
};
use sim_lib_compute_auto::{
    AutoComputeProfile, AutoRouteDecision, AutoTensorExecutor, BenchmarkBounds,
    ComputeDeviceIdentity, ComputeEvidenceKind, ComputeThermalPowerContext,
    measure_bounded_profile,
};
use sim_lib_compute_model::{ModeledComputeProfile, ModeledTensorExecutor};
use sim_lib_compute_wgpu::{
    ComputeWgpuLib, WgpuAdapterProbe, WgpuDiscovery, compute_wgpu_capability,
    compute_wgpu_site_symbol,
};
use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, SamplingPolicy, SamplingThresholds, ScalarMedium,
    SourceSet, UnitVector3, WorkBudget,
};
use sim_lib_interference_runtime::{
    InterferenceLib, InterferenceRecordsLib, PlaneDescriptor, ProblemDescriptor, StudyDescriptor,
    solve_function_symbol,
};
use sim_lib_interference_solve::{HostPhasorField, ReferencePhasorSolver};
use sim_lib_numbers_tensor::{
    TensorExecutor, TensorLocation, TensorNumbersLib, TensorSite, domains,
};

use crate::{
    CpuFallbackReason, DifferentialReport, DifferentialTolerances, HardwareEvidenceMetrics,
    HardwareEvidenceReport, HardwareMeasurementResult, PhaseBudget, ProviderRoute,
    TensorStudyConfig, TensorStudySolver, TileProfile, compare_dense_to_reference,
    compare_materialized_to_reference, solve_dense_f32_cpu,
};

// conformance: dense, modeled, automatic, and physical wgpu providers share one differential and hardware-evidence contract.

const PHYSICAL_ENV: &str = "SIM_INTERFERENCE_WGPU_PHYSICAL";
const TARGET_ENV: &str = "SIM_INTERFERENCE_ACCEPTANCE_TARGET";
#[derive(Clone)]
struct MatrixCase {
    name: &'static str,
    problem: InterferenceProblem,
    plane: SamplingPlane,
    profile: TileProfile,
}

#[test]
fn f64_dense_and_modeled_share_one_differential_matrix() {
    for case in matrix_cases() {
        let oracle = reference(&case);
        let dense = dense(&case);
        let dense_report =
            compare_dense_to_reference(&oracle, &dense, DifferentialTolerances::default()).unwrap();
        assert_report(case.name, "dense", &dense_report);

        let executor = ModeledTensorExecutor::new(modeled_profile(&case));
        let solver = solver(&case);
        let (study, mut cx) = solve_through_executor(
            solver.clone(),
            Arc::new(executor.clone()),
            Symbol::new("site/compute/model"),
            &case,
        )
        .unwrap();
        assert_resident_lifecycle(&case, &study);
        let materialized = study.field.materialize_host(&mut cx).unwrap();
        let modeled_report = compare_materialized_to_reference(
            &oracle,
            &materialized,
            DifferentialTolerances::default(),
        )
        .unwrap();
        assert_report(case.name, "modeled", &modeled_report);
        assert_eq!(
            differential_shape(&dense_report),
            differential_shape(&modeled_report),
            "{} did not use identical comparison reporting",
            case.name
        );

        let repeated = study.field.materialize_host(&mut cx).unwrap();
        assert_eq!(field_bits(&materialized), field_bits(&repeated));
        assert_eq!(executor.snapshot().readbacks, 2);
        assert!(matches!(
            solver.snapshot().last_route,
            ProviderRoute::ProviderCompleted(_)
        ));
    }
}

#[test]
fn matrix_covers_segments_distance_cancellation_attenuation_and_crossover_edges() {
    let cases = matrix_cases();
    let segmented = cases
        .iter()
        .find(|case| case.name == "attenuated-multi-segment")
        .unwrap();
    let segmented_dense = dense(segmented);
    assert_eq!(segmented_dense.evidence().tiles(), 1);
    assert!(segmented_dense.evidence().max_segments_per_tensor() > 1);

    let long_world = cases
        .iter()
        .find(|case| case.name == "long-world-distance")
        .unwrap();
    assert!(long_world.plane.origin().coordinates_metres()[0] >= 1_000.0);
    assert!(dense(long_world).evidence().observed_max_abs_psi_rad() <= std::f64::consts::PI);

    let cancellation = cases
        .iter()
        .find(|case| case.name == "exact-cancellation")
        .unwrap();
    let cancellation_report = compare_dense_to_reference(
        &reference(cancellation),
        &dense(cancellation),
        DifferentialTolerances::default(),
    )
    .unwrap();
    assert!(cancellation_report.passed());
    assert_eq!(cancellation_report.phase_cells, 0);
    assert_eq!(cancellation_report.phase, None);

    measured_crossover_edges();
}

#[test]
fn explicit_wgpu_absence_and_auto_cpu_choice_are_pre_submission() {
    let case = matrix_cases().remove(0);
    let wgpu = ComputeWgpuLib::from_discovery(WgpuDiscovery::default());
    let absent_solver = solver(&case);
    let mut cx = runtime_cx(absent_solver.clone());
    cx.load_lib(&wgpu).unwrap();
    let error = solve_through_registered_site(&mut cx, compute_wgpu_site_symbol(0), &case)
        .unwrap_err()
        .to_string();
    assert!(error.contains("is unavailable"), "{error}");
    assert_eq!(absent_solver.snapshot().last_route, ProviderRoute::Idle);

    let auto = AutoTensorExecutor::default();
    assert_eq!(auto.route_decision(), AutoRouteDecision::Absent);
    assert!(auto.uses_cpu_fallback());
    assert!(auto.routing_events().is_empty());
    let auto_solver = solver(&case);
    let (study, _) = solve_through_executor(
        auto_solver.clone(),
        Arc::new(auto.clone()),
        Symbol::new("site/compute/auto"),
        &case,
    )
    .unwrap();
    assert_eq!(study.evidence.dtype, domains::f64());
    assert_eq!(
        auto_solver.snapshot().last_route,
        ProviderRoute::Cpu(CpuFallbackReason::ProviderCpuChoice)
    );
    assert!(
        auto.routing_events().is_empty(),
        "automatic CPU placement must be selected before the first Tensor submission"
    );
}

#[test]
fn wgpu_matrix_repeats_same_profile_one_hundred_times_when_opted_in() {
    if std::env::var_os(PHYSICAL_ENV).is_none() {
        let evidence = HardwareEvidenceReport::not_measured("unavailable", "unavailable");
        println!("{evidence}");
        assert_eq!(evidence.result(), HardwareMeasurementResult::NotMeasured);
        assert!(!evidence.satisfies_hardware_gate());
        return;
    }

    let wgpu = ComputeWgpuLib::probe().expect("opted-in wgpu discovery must complete");
    let target = std::env::var(TARGET_ENV).ok();
    let probe = wgpu
        .discovery()
        .adapters
        .iter()
        .find(|probe| {
            target
                .as_deref()
                .is_none_or(|target| target_matches(target, &probe.adapter.name))
        })
        .cloned()
        .expect("opted-in wgpu measurement requires the requested probe-backed adapter");
    for case in matrix_cases() {
        run_wgpu(&wgpu, &probe, &case, crate::HARDWARE_DETERMINISM_REPEATS);
    }
    if target.as_deref().is_some_and(large_target) {
        let case = crossover_case("above-5080-crossover", 129, 129);
        run_wgpu(&wgpu, &probe, &case, 3);
    }
}

#[test]
fn wgpu_evidence_recipe_reports_not_measured_without_a_hardware_claim() {
    let evidence = HardwareEvidenceReport::not_measured("unavailable", "unavailable");
    assert_eq!(
        evidence.to_string(),
        include_str!("../recipes/01-basics/wgpu-differential-evidence/expected.txt").trim_end()
    );
    assert!(!evidence.satisfies_hardware_gate());
}

fn run_wgpu(lib: &ComputeWgpuLib, probe: &WgpuAdapterProbe, case: &MatrixCase, repeats: usize) {
    let oracle = reference(case);
    let dense = dense(case);
    let dense_report =
        compare_dense_to_reference(&oracle, &dense, DifferentialTolerances::default()).unwrap();
    assert_report(case.name, "dense", &dense_report);

    let mut cx = runtime_cx(solver(case));
    cx.grant(compute_wgpu_capability());
    cx.load_lib(lib).unwrap();
    let mut baseline = None;
    let mut final_report = None;
    let mut final_study = None;
    let mut deterministic = true;
    for _ in 0..repeats {
        let study = solve_through_registered_site(
            &mut cx,
            compute_wgpu_site_symbol(probe.adapter.ordinal),
            case,
        )
        .unwrap();
        assert_resident_lifecycle(case, &study);
        let materialized = study.field.materialize_host(&mut cx).unwrap();
        let report = compare_materialized_to_reference(
            &oracle,
            &materialized,
            DifferentialTolerances::default(),
        )
        .unwrap();
        assert_report(case.name, "wgpu", &report);
        let bits = field_bits(&materialized);
        if let Some(expected) = &baseline {
            deterministic &= expected == &bits;
        } else {
            baseline = Some(bits);
        }
        final_report = Some(report);
        final_study = Some(study);
    }
    let study = final_study.unwrap();
    let report = final_report.unwrap();
    let evidence = HardwareEvidenceReport::measured(
        adapter_id(probe),
        profile_id(probe),
        HardwareEvidenceMetrics {
            tiles: dense.evidence().tiles(),
            segments: study.evidence.segments,
            max_abs_psi: dense.evidence().observed_max_abs_psi_rad(),
            intermediate_materializations: study.evidence.intermediate_materializations,
            final_materializations: study.evidence.final_materializations,
            repeats,
            deterministic,
        },
        &report,
    );
    println!("case={}\n{evidence}", case.name);
    println!(
        "acceptance-case=(case (id \"{}\") (cells \"{}\") (sources \"{}\") (tiles \"{}\") (segments \"{}\") (repeats \"{}\") (max_psi \"{:.9e}\") (component_tolerance \"{:.9e}\") (phase_tolerance \"{:.9e}\") (max_component_abs \"{:.9e}\") (max_phase_abs \"{:.9e}\") (intermediate_materializations \"{}\") (final_materializations \"{}\") (result \"pass\"))",
        case.name,
        case.plane.cell_count(),
        case.problem.sources.len(),
        dense.evidence().tiles(),
        study.evidence.segments,
        repeats,
        dense.evidence().observed_max_abs_psi_rad(),
        DifferentialTolerances::default().component.absolute,
        DifferentialTolerances::default().phase.absolute,
        report.max_component_absolute_error(),
        report.max_phase_absolute_error(),
        study.evidence.intermediate_materializations,
        study.evidence.final_materializations,
    );
    assert!(
        evidence.satisfies_hardware_gate(),
        "{} failed measured hardware evidence: {evidence}",
        case.name
    );
}

fn measured_crossover_edges() {
    let identity = ComputeDeviceIdentity::new("measured-adapter", "driver-v1", "wgpu");
    let mut measured = measure_bounded_profile(
        identity.clone(),
        modeled_profile(&matrix_cases()[0]),
        ComputeThermalPowerContext {
            thermal: "steady".to_owned(),
            power: "bounded".to_owned(),
        },
        "interference-crossover-v1",
        7,
        BenchmarkBounds::default(),
    );
    measured.provenance.evidence_kind = ComputeEvidenceKind::PhysicalDevice;

    let below_case = crossover_case("crossover-below", 7, 9);
    let below_auto = AutoTensorExecutor::new(AutoComputeProfile {
        modeled: None,
        measured: Some(measured.clone()),
        expected: Some(identity.clone()),
        now_tick: 8,
    });
    assert_eq!(below_auto.route_decision(), AutoRouteDecision::Device);
    let below_solver = crossover_solver(&below_case);
    let (below, _) = solve_through_executor(
        below_solver.clone(),
        Arc::new(below_auto.clone()),
        Symbol::new("site/compute/auto"),
        &below_case,
    )
    .unwrap();
    assert_eq!(below.evidence.dtype, domains::f64());
    assert_eq!(
        below_solver.snapshot().last_route,
        ProviderRoute::Cpu(CpuFallbackReason::BelowCrossover)
    );
    assert!(below_auto.routing_events().is_empty());

    let edge_case = crossover_case("crossover-at", 8, 8);
    let edge_auto = AutoTensorExecutor::new(AutoComputeProfile {
        modeled: None,
        measured: Some(measured),
        expected: Some(identity),
        now_tick: 8,
    });
    let edge_solver = crossover_solver(&edge_case);
    let (edge, mut edge_cx) = solve_through_executor(
        edge_solver.clone(),
        Arc::new(edge_auto.clone()),
        Symbol::new("site/compute/auto"),
        &edge_case,
    )
    .unwrap();
    let materialized = edge.field.materialize_host(&mut edge_cx).unwrap();
    let report = compare_materialized_to_reference(
        &reference(&edge_case),
        &materialized,
        DifferentialTolerances::default(),
    )
    .unwrap();
    assert_report(edge_case.name, "auto-at-crossover", &report);
    assert!(matches!(
        edge_solver.snapshot().last_route,
        ProviderRoute::ProviderCompleted(_)
    ));
    let events = edge_auto.routing_events();
    assert!(!events.is_empty());
    assert!(
        events
            .iter()
            .all(|event| event.decision == AutoRouteDecision::Device)
    );
}

fn solve_through_executor(
    solver: TensorStudySolver,
    executor: Arc<dyn TensorExecutor>,
    site_symbol: Symbol,
    case: &MatrixCase,
) -> sim_kernel::Result<(StudyDescriptor, Cx)> {
    let mut cx = runtime_cx(solver);
    let site = TensorSite::new(site_symbol, executor, Vec::new());
    let study = realize_study(&mut cx, &site, case)?;
    Ok((study, cx))
}

fn solve_through_registered_site(
    cx: &mut Cx,
    site_symbol: Symbol,
    case: &MatrixCase,
) -> sim_kernel::Result<StudyDescriptor> {
    let site_value = cx
        .registry()
        .site_by_symbol(&site_symbol)
        .cloned()
        .ok_or_else(|| Error::Eval(format!("explicit wgpu site {site_symbol} is unavailable")))?;
    let site = site_value
        .object()
        .as_eval_fabric()
        .ok_or_else(|| Error::Eval(format!("{site_symbol} is not an EvalFabric site")))?;
    realize_study(cx, site, case)
}

fn realize_study(
    cx: &mut Cx,
    site: &dyn EvalFabric,
    case: &MatrixCase,
) -> sim_kernel::Result<StudyDescriptor> {
    let problem_name = Symbol::qualified("hardware-test", "problem");
    let plane_name = Symbol::qualified("hardware-test", "plane");
    let problem_value = cx
        .factory()
        .opaque(Arc::new(ProblemDescriptor::from_problem(&case.problem)))?;
    let plane_value = cx
        .factory()
        .opaque(Arc::new(PlaneDescriptor::from_plane(case.plane)))?;
    cx.env_mut().define(problem_name.clone(), problem_value);
    cx.env_mut().define(plane_name.clone(), plane_value);
    let reply = site.realize(
        cx,
        EvalRequest {
            expr: Expr::Call {
                operator: Box::new(Expr::Symbol(solve_function_symbol())),
                args: vec![
                    Expr::Symbol(problem_name),
                    Expr::Symbol(plane_name),
                    Expr::Map(vec![
                        (
                            Expr::Symbol(Symbol::new("sampling")),
                            Expr::Symbol(Symbol::new("annotate")),
                        ),
                        (
                            Expr::Symbol(Symbol::new("work-budget")),
                            Expr::Symbol(Symbol::new("default")),
                        ),
                    ]),
                ],
            },
            result_shape: None,
            required_capabilities: Vec::new(),
            deadline: None,
            consistency: Consistency::LocalFirst,
            mode: EvalMode::Eval,
            answer_limit: None,
            stream_buffer: None,
            stream: false,
            trace: false,
        },
    )?;
    reply
        .value
        .object()
        .downcast_ref::<StudyDescriptor>()
        .cloned()
        .ok_or_else(|| Error::Eval("compute site did not return an interference Study".to_owned()))
}

fn runtime_cx(solver: TensorStudySolver) -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_float::F32NumbersLib::new())
        .unwrap();
    cx.load_lib(&TensorNumbersLib::new()).unwrap();
    cx.load_lib(&InterferenceRecordsLib).unwrap();
    cx.load_lib(&InterferenceLib).unwrap();
    cx.load_lib(&crate::InterferenceComputeLib::new(solver))
        .unwrap();
    cx
}

fn solver(case: &MatrixCase) -> TensorStudySolver {
    TensorStudySolver::new(TensorStudyConfig {
        min_accelerated_cells: 1,
        tile_profile: case.profile,
        ..TensorStudyConfig::default()
    })
}

fn crossover_solver(case: &MatrixCase) -> TensorStudySolver {
    TensorStudySolver::new(TensorStudyConfig {
        min_accelerated_cells: 64,
        tile_profile: case.profile,
        ..TensorStudyConfig::default()
    })
}

fn assert_resident_lifecycle(case: &MatrixCase, study: &StudyDescriptor) {
    assert_eq!(study.evidence.dtype, domains::f32(), "{}", case.name);
    assert_eq!(
        study.evidence.intermediate_materializations, 0,
        "{}",
        case.name
    );
    assert_eq!(study.evidence.final_materializations, 2, "{}", case.name);
    assert!(study.evidence.segments >= 2, "{}", case.name);
    assert!(matches!(
        study.field.real.location(),
        TensorLocation::Resident { .. }
    ));
    assert!(matches!(
        study.field.imag.location(),
        TensorLocation::Resident { .. }
    ));
}

fn differential_shape(report: &DifferentialReport) -> (usize, bool) {
    (report.phase_cells, report.phase.is_some())
}

fn assert_report(case: &str, provider: &str, report: &DifferentialReport) {
    assert!(report.real.passed(), "{case}/{provider}: {report:?}");
    assert!(report.imaginary.passed(), "{case}/{provider}: {report:?}");
    assert!(report.amplitude.passed(), "{case}/{provider}: {report:?}");
    assert!(
        report.phase.is_none_or(crate::DifferentialMaximum::passed),
        "{case}/{provider}: {report:?}"
    );
    assert!(
        report.magnitude_squared.passed(),
        "{case}/{provider}: {report:?}"
    );
    assert!(report.passed(), "{case}/{provider}: {report:?}");
}

fn reference(case: &MatrixCase) -> HostPhasorField {
    ReferencePhasorSolver::new(
        SamplingPolicy::Annotate,
        SamplingThresholds::default(),
        WorkBudget::default(),
    )
    .solve(&case.problem, &case.plane)
    .unwrap()
    .0
}

fn dense(case: &MatrixCase) -> crate::DenseF32Field {
    let mut cx = dense_cx();
    solve_dense_f32_cpu(
        &mut cx,
        &case.problem,
        case.plane,
        PhaseBudget::default(),
        case.profile,
    )
    .unwrap()
}

fn dense_cx() -> Cx {
    let mut cx = sim_kernel::testing::eager_cx();
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_float::F32NumbersLib::new())
        .unwrap();
    cx
}

fn matrix_cases() -> Vec<MatrixCase> {
    vec![
        MatrixCase {
            name: "attenuated-multi-segment",
            problem: point_problem(0.012),
            plane: plane(12.0, 8, 8),
            profile: segmented_profile(64),
        },
        MatrixCase {
            name: "long-world-distance",
            problem: forward_problem(vec![(0.37, 1.0)]),
            plane: plane(1_000.0, 4, 5),
            profile: TileProfile::default(),
        },
        MatrixCase {
            name: "exact-cancellation",
            problem: forward_problem(vec![(0.0, 1.0), (std::f64::consts::PI, 1.0)]),
            plane: plane(10.0, 4, 4),
            profile: TileProfile::default(),
        },
    ]
}

fn target_matches(target: &str, adapter: &str) -> bool {
    match target {
        "gpu:nvidia/rtx-5080-laptop" => adapter.contains("RTX 5080"),
        "gpu:nvidia/rtx-5090" => adapter.contains("RTX 5090"),
        "gpu:amd/gfx1151" => ["RADV STRIX_HALO", "Radeon 8060S", "AMD Radeon Graphics"]
            .iter()
            .any(|needle| adapter.contains(needle)),
        _ => false,
    }
}
fn large_target(target: &str) -> bool {
    matches!(target, "gpu:nvidia/rtx-5090" | "gpu:amd/gfx1151")
}
fn crossover_case(name: &'static str, rows: usize, columns: usize) -> MatrixCase {
    MatrixCase {
        name,
        problem: point_problem(0.004),
        plane: plane(8.0, rows, columns),
        profile: TileProfile::default(),
    }
}
fn point_problem(attenuation: f64) -> InterferenceProblem {
    problem(
        attenuation,
        vec![
            Emitter::Point {
                id: "left".to_owned(),
                position: point(0.0, -0.04, 0.0),
                amplitude_at_reference: FieldAmplitude::new(1.0).unwrap(),
                phase: Radians::new(0.2).unwrap(),
            },
            Emitter::Point {
                id: "right".to_owned(),
                position: point(0.0, 0.05, 0.01),
                amplitude_at_reference: FieldAmplitude::new(0.7).unwrap(),
                phase: Radians::new(-0.35).unwrap(),
            },
        ],
    )
}
fn forward_problem(phases: Vec<(f64, f64)>) -> InterferenceProblem {
    let sources = phases
        .into_iter()
        .enumerate()
        .map(|(index, (phase, amplitude))| Emitter::ForwardPlane {
            id: format!("plane-{index}"),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(1.0, 0.125, 0.0).unwrap(),
            amplitude: FieldAmplitude::new(amplitude).unwrap(),
            phase: Radians::new(phase).unwrap(),
        })
        .collect();
    problem(0.0, sources)
}
fn problem(attenuation: f64, sources: Vec<Emitter>) -> InterferenceProblem {
    InterferenceProblem::new(
        Hertz::new(343.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(343.0).unwrap(),
            NepersPerMetre::new(attenuation).unwrap(),
        ),
        SourceSet::new(sources).unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    )
}
fn plane(distance: f64, rows: usize, columns: usize) -> SamplingPlane {
    SamplingPlane::new(
        point(distance, -0.125, -0.125),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
        PositiveMetres::new(0.25).unwrap(),
        PositiveMetres::new(0.25).unwrap(),
        rows,
        columns,
    )
    .unwrap()
}
fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}
fn segmented_profile(cells: usize) -> TileProfile {
    TileProfile {
        max_elements_per_tile: cells,
        max_segment_bytes: 64,
        max_segments_per_tensor: cells.div_ceil(16),
        max_tensor_bytes: (cells * 4) as u64,
        max_result_bytes: (cells * 8) as u64,
        ..TileProfile::default()
    }
}
fn modeled_profile(case: &MatrixCase) -> ModeledComputeProfile {
    ModeledComputeProfile {
        provider: format!("modeled/{}", case.name),
        max_queue_depth: 64,
        max_queue_bytes: 1 << 20,
        max_resident_bytes: 16 << 20,
        segment_tile_bytes: case.profile.max_segment_bytes,
        max_storage_binding_bytes: case.profile.max_segment_bytes,
        submission_deadline_ticks: 1_000,
        fault: None,
        auto_flush_batches: true,
    }
}
fn field_bits(field: &HostPhasorField) -> (Vec<u64>, Vec<u64>) {
    (
        field.real().iter().map(|value| value.to_bits()).collect(),
        field
            .imaginary()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
    )
}

fn adapter_id(probe: &WgpuAdapterProbe) -> String {
    format!(
        "{}/{:04x}:{:04x}/{}",
        probe.adapter.backend, probe.adapter.vendor, probe.adapter.device, probe.adapter.ordinal
    )
}

fn profile_id(probe: &WgpuAdapterProbe) -> String {
    format!(
        "wgpu-v1/buffer-{}/binding-{}/wg-{}",
        probe.adapter.granted_limits.max_buffer_size,
        probe.adapter.granted_limits.max_storage_buffer_binding_size,
        probe
            .adapter
            .granted_limits
            .max_compute_invocations_per_workgroup
    )
}
