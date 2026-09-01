//! Headless provider routing, residency, fallback, and fault conformance.

use std::{
    any::Any,
    sync::{Arc, Mutex, OnceLock},
};

use sim_kernel::{
    Consistency, Cx, DefaultFactory, EagerPolicy, Error, EvalFabric, EvalMode, EvalRequest, Expr,
    Symbol, Value,
};
use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, SamplingPolicy, SamplingThresholds, ScalarMedium,
    SourceSet, UnitVector3, WorkBudget,
};
use sim_lib_interference_runtime::{
    InterferenceLib, InterferenceRecordsLib, PlaneDescriptor, ProblemDescriptor, SolveRequest,
    StudyDescriptor, StudySolver, solve_function_symbol,
};
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, SubmissionEvidence, Tensor, TensorExecError, TensorExecution,
    TensorExecutor, TensorExecutorCard, TensorLocation, TensorNumbersLib, TensorRequest,
    TensorSite, TensorStorage, cos_op_symbol, domains,
};

use crate::{
    CpuFallbackReason, InterferenceComputeLib, ProviderRoute, RECIPES, TensorStudyConfig,
    TensorStudySolver, TileProfile, preflight::required_operation_symbols,
};

// conformance: Env-bound TensorStudySolver selects modeled resident execution.

#[test]
fn tensor_site_environment_selects_the_registered_compute_solver() {
    let solver = TensorStudySolver::new(TensorStudyConfig {
        min_accelerated_cells: 1,
        ..TensorStudyConfig::default()
    });
    let mut cx = runtime_cx(solver.clone());
    let problem = modeled_problem();
    let plane = modeled_plane(8, 8);
    let problem_name = Symbol::qualified("test", "problem");
    let plane_name = Symbol::qualified("test", "plane");
    let problem_value = cx
        .factory()
        .opaque(Arc::new(ProblemDescriptor::from_problem(&problem)))
        .unwrap();
    let plane_value = cx
        .factory()
        .opaque(Arc::new(PlaneDescriptor::from_plane(plane)))
        .unwrap();
    cx.env_mut().define(problem_name.clone(), problem_value);
    cx.env_mut().define(plane_name.clone(), plane_value);

    let executor = ModeledExecutor::default();
    let site = TensorSite::new(
        modeled_site_symbol(),
        Arc::new(executor.clone()),
        Vec::new(),
    );
    let reply = site
        .realize(
            &mut cx,
            eval_request(Expr::Call {
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
            }),
        )
        .unwrap();
    let study = reply
        .value
        .object()
        .downcast_ref::<StudyDescriptor>()
        .expect("TensorSite solve returns an interference Study");
    assert!(matches!(
        study.field.real.location(),
        TensorLocation::Resident { site, .. } if site == modeled_site_symbol()
    ));
    assert!(matches!(
        study.field.imag.location(),
        TensorLocation::Resident { site, .. } if site == modeled_site_symbol()
    ));
    assert_eq!(
        study.evidence.provider,
        Symbol::qualified("compute", "executor/model")
    );
    assert_eq!(study.evidence.uploads, 3);
    assert!(study.evidence.submissions > study.evidence.uploads);
    assert_eq!(study.evidence.intermediate_materializations, 0);
    assert_eq!(study.evidence.final_materializations, 2);
    assert_eq!(executor.snapshot().readbacks, 0);
    assert_eq!(
        solver.snapshot().last_route,
        ProviderRoute::ProviderCompleted(Symbol::qualified("compute", "executor/model"))
    );
}

#[test]
fn cpu_is_selected_before_submission_for_every_decline_boundary() {
    let problem = modeled_problem();
    let plane = modeled_plane(8, 8);

    let absent = TensorStudySolver::new(TensorStudyConfig {
        min_accelerated_cells: 1,
        ..TensorStudyConfig::default()
    });
    let mut absent_cx = runtime_cx(absent.clone());
    let absent_request = solve_request(&problem, &plane);
    let absent_study = absent.solve(&mut absent_cx, &absent_request).unwrap();
    assert_eq!(absent_study.evidence.dtype, domains::f64());
    assert_eq!(
        absent.snapshot().last_route,
        ProviderRoute::Cpu(CpuFallbackReason::NoExecutor)
    );

    let unsupported = ModeledExecutor {
        missing_operation: Some(cos_op_symbol()),
        ..ModeledExecutor::default()
    };
    let unsupported_solver = TensorStudySolver::new(TensorStudyConfig {
        min_accelerated_cells: 1,
        ..TensorStudyConfig::default()
    });
    let (unsupported_study, _) =
        solve_through_site(unsupported_solver.clone(), Arc::new(unsupported.clone())).unwrap();
    assert_eq!(unsupported_study.evidence.dtype, domains::f64());
    assert_eq!(unsupported.snapshot().accepted, 0);
    assert_eq!(
        unsupported_solver.snapshot().last_route,
        ProviderRoute::Cpu(CpuFallbackReason::UnsupportedOperations)
    );

    let dtype = ModeledExecutor::default();
    let dtype_solver = TensorStudySolver::new(TensorStudyConfig {
        min_accelerated_cells: 1,
        tile_profile: TileProfile {
            supports_f32: false,
            ..TileProfile::default()
        },
        ..TensorStudyConfig::default()
    });
    let (dtype_study, _) =
        solve_through_site(dtype_solver.clone(), Arc::new(dtype.clone())).unwrap();
    assert_eq!(dtype_study.evidence.dtype, domains::f64());
    assert_eq!(dtype.snapshot().accepted, 0);
    assert_eq!(
        dtype_solver.snapshot().last_route,
        ProviderRoute::Cpu(CpuFallbackReason::UnsupportedDtype)
    );

    let below = ModeledExecutor::default();
    let below_solver = TensorStudySolver::new(TensorStudyConfig {
        min_accelerated_cells: 65,
        ..TensorStudyConfig::default()
    });
    let (below_study, _) =
        solve_through_site(below_solver.clone(), Arc::new(below.clone())).unwrap();
    assert_eq!(below_study.evidence.dtype, domains::f64());
    assert_eq!(below.snapshot().accepted, 0);
    assert_eq!(
        below_solver.snapshot().last_route,
        ProviderRoute::Cpu(CpuFallbackReason::BelowCrossover)
    );
}

#[test]
fn immutable_inputs_upload_once_and_only_final_components_read_back() {
    let solver = TensorStudySolver::new(TensorStudyConfig {
        min_accelerated_cells: 1,
        ..TensorStudyConfig::default()
    });
    let executor = ModeledExecutor::default();
    let (study, mut cx) =
        solve_through_site(solver, Arc::new(executor.clone())).expect("modeled resident solve");

    assert_eq!(study.evidence.uploads, 3);
    assert_eq!(
        u64::try_from(executor.snapshot().accepted).unwrap(),
        study.evidence.submissions
    );
    assert_eq!(study.evidence.intermediate_materializations, 0);
    assert_eq!(executor.snapshot().readbacks, 0);
    assert!(matches!(
        study.field.real.location(),
        TensorLocation::Resident { .. }
    ));
    assert!(matches!(
        study.field.imag.location(),
        TensorLocation::Resident { .. }
    ));

    let first = study.field.materialize_host(&mut cx).unwrap();
    assert_eq!(first.len(), 64);
    assert_eq!(executor.snapshot().readbacks, 2);
    let repeated = study.field.materialize_host(&mut cx).unwrap();
    assert_eq!(first, repeated);
    assert_eq!(executor.snapshot().readbacks, 2);
    assert_eq!(study.evidence.final_materializations, 2);
}

#[test]
fn selected_provider_faults_never_restart_on_cpu() {
    for (fault, diagnostic, accepted) in [
        (ModeledFault::Oom, "out of memory", 0),
        (ModeledFault::DeviceLost, "device lost", 1),
        (ModeledFault::Deadline, "deadline expired", 0),
        (ModeledFault::Execution, "execution failed", 1),
    ] {
        let solver = TensorStudySolver::new(TensorStudyConfig {
            min_accelerated_cells: 1,
            ..TensorStudyConfig::default()
        });
        let executor = ModeledExecutor::with_fault(fault);
        let error = match solve_through_site(solver.clone(), Arc::new(executor.clone())) {
            Ok(_) => panic!("selected provider fault must fail the solve"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains(diagnostic), "{fault:?}: {error}");
        assert!(
            error.contains("CPU restart is forbidden"),
            "{fault:?}: {error}"
        );
        assert_eq!(executor.snapshot().accepted, accepted, "{fault:?}");
        let snapshot = solver.snapshot();
        assert_eq!(snapshot.cpu_fallbacks, 0, "{fault:?}");
        assert_eq!(snapshot.provider_completions, 0, "{fault:?}");
        assert_eq!(snapshot.provider_failures, 1, "{fault:?}");
        assert_eq!(
            snapshot.last_route,
            ProviderRoute::ProviderFailed(Symbol::qualified("compute", "executor/model")),
            "{fault:?}"
        );
    }

    let solver = TensorStudySolver::new(TensorStudyConfig {
        min_accelerated_cells: 1,
        ..TensorStudyConfig::default()
    });
    let executor = ModeledExecutor::with_fault(ModeledFault::Readback);
    let (study, mut cx) = solve_through_site(solver.clone(), Arc::new(executor.clone()))
        .expect("readback failure happens only on observation");
    let first = study
        .field
        .materialize_host(&mut cx)
        .unwrap_err()
        .to_string();
    let repeated = study
        .field
        .materialize_host(&mut cx)
        .unwrap_err()
        .to_string();
    assert!(first.contains("readback failed"));
    assert_eq!(first, repeated);
    assert_eq!(executor.snapshot().readbacks, 1);
    assert_eq!(executor.snapshot().materialization_failures, 1);
    let snapshot = solver.snapshot();
    assert_eq!(snapshot.cpu_fallbacks, 0);
    assert_eq!(snapshot.provider_completions, 1);
    assert_eq!(snapshot.provider_failures, 0);
    assert_eq!(
        snapshot.last_route,
        ProviderRoute::ProviderCompleted(Symbol::qualified("compute", "executor/model"))
    );
}

#[test]
fn modeled_resident_recipe_reports_checked_evidence() {
    const SETUP: &str = include_str!("../recipes/01-basics/modeled-resident-study/setup.siml");
    let solver = TensorStudySolver::new(TensorStudyConfig {
        min_accelerated_cells: 1,
        ..TensorStudyConfig::default()
    });
    let executor = ModeledExecutor::default();
    let (study, mut cx) = solve_through_site(solver.clone(), Arc::new(executor.clone())).unwrap();
    study.field.materialize_host(&mut cx).unwrap();
    study.field.materialize_host(&mut cx).unwrap();
    let fallback = match solver.snapshot().last_route {
        ProviderRoute::ProviderCompleted(_) => "none",
        _ => "unexpected",
    };
    let report = format!(
        "provider={} tiles=1 uploads={} submits={}\n\
         intermediate_materializations={} final_materializations={}\n\
         fallback={fallback} result=pass",
        study.evidence.profile.as_deref().unwrap(),
        study.evidence.uploads,
        study.evidence.submissions,
        study.evidence.intermediate_materializations,
        executor.snapshot().readbacks,
    );
    assert_eq!(
        report,
        include_str!("../recipes/01-basics/modeled-resident-study/expected.txt").trim_end()
    );
    let cards = sim_cookbook::recipes_from_embedded(RECIPES).unwrap();
    let card = cards
        .iter()
        .find(|card| card.id.ends_with("modeled-resident-study"))
        .expect("embedded modeled resident recipe");
    assert_eq!(card.setup, SETUP.as_bytes());
}

fn modeled_site_symbol() -> Symbol {
    Symbol::new("site/compute/model")
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ModeledFault {
    #[default]
    None,
    Oom,
    DeviceLost,
    Deadline,
    Execution,
    Readback,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ModeledSnapshot {
    accepted: usize,
    queued: usize,
    readbacks: usize,
    materialization_failures: usize,
}

#[derive(Clone, Default)]
struct ModeledExecutor {
    fault: ModeledFault,
    missing_operation: Option<Symbol>,
    state: Arc<Mutex<ModeledSnapshot>>,
}

impl ModeledExecutor {
    fn with_fault(fault: ModeledFault) -> Self {
        Self {
            fault,
            missing_operation: None,
            state: Arc::new(Mutex::new(ModeledSnapshot::default())),
        }
    }

    fn snapshot(&self) -> ModeledSnapshot {
        *self.state.lock().expect("modeled state poisoned")
    }

    fn prepare_tensor(tensor: &Tensor) -> Tensor {
        tensor
            .storage()
            .as_any()
            .downcast_ref::<ModeledStorage>()
            .and_then(ModeledStorage::resident_tensor)
            .unwrap_or_else(|| tensor.clone())
    }
}

impl TensorExecutor for ModeledExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            Symbol::qualified("compute", "executor/model"),
            "model",
            Symbol::qualified("compute", "modeled-resident"),
            required_operation_symbols()
                .into_iter()
                .filter(|operation| self.missing_operation.as_ref() != Some(operation))
                .collect(),
            None,
        )
    }

    fn execute(
        &self,
        cx: &mut Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        match self.fault {
            ModeledFault::Oom => {
                return Err(TensorExecError::InvalidRequest {
                    message: Arc::from("modeled compute out of memory before submission"),
                });
            }
            ModeledFault::Deadline => {
                return Err(TensorExecError::InvalidRequest {
                    message: Arc::from("modeled compute submission deadline expired"),
                });
            }
            _ => {}
        }
        let allocation = {
            let mut state = self.state.lock().expect("modeled state poisoned");
            state.accepted += 1;
            state.queued += 1;
            state.accepted
        };
        if self.fault == ModeledFault::DeviceLost {
            return Err(TensorExecError::Eval {
                message: Arc::from("modeled compute device lost during execution"),
            });
        }
        if self.fault == ModeledFault::Execution {
            return Err(TensorExecError::Eval {
                message: Arc::from("modeled compute execution failed"),
            });
        }
        let request = TensorRequest::new(
            request.operation,
            request.inputs.iter().map(Self::prepare_tensor).collect(),
            request.output,
        );
        match CpuTensorExecutor::new().execute(cx, request)? {
            TensorExecution::Complete(tensor) => {
                let cells = tensor.cells().map_err(TensorExecError::from)?;
                let storage = ModeledStorage {
                    shape: tensor.shape().to_vec(),
                    dtype: tensor.dtype().clone(),
                    cells,
                    allocation,
                    executor: self.clone(),
                    materialized: OnceLock::new(),
                };
                Ok(TensorExecution::Complete(Tensor::from_storage(
                    tensor.shape().to_vec(),
                    tensor.dtype().clone(),
                    Arc::new(storage),
                )?))
            }
            unsupported => Ok(unsupported),
        }
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        let mut state = self.state.lock().expect("modeled state poisoned");
        let accepted = state.queued;
        state.queued = 0;
        Ok(SubmissionEvidence::new(self.card().symbol, accepted))
    }
}

struct ModeledStorage {
    shape: Vec<usize>,
    dtype: Symbol,
    cells: Arc<[Value]>,
    allocation: usize,
    executor: ModeledExecutor,
    materialized: OnceLock<sim_kernel::Result<Arc<dyn TensorStorage>>>,
}

impl ModeledStorage {
    fn resident_tensor(&self) -> Option<Tensor> {
        Tensor::from_storage(
            self.shape.clone(),
            self.dtype.clone(),
            Arc::new(HostStorage {
                dtype: self.dtype.clone(),
                cells: self.cells.clone(),
            }),
        )
        .ok()
    }
}

impl TensorStorage for ModeledStorage {
    fn dtype(&self) -> &Symbol {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.cells.len()
    }

    fn location(&self) -> TensorLocation {
        TensorLocation::Resident {
            site: modeled_site_symbol(),
            allocation: Symbol::qualified("compute.alloc", self.allocation.to_string()),
        }
    }

    fn cell(&self, index: usize) -> sim_kernel::Result<Value> {
        self.materialize()?.cell(index)
    }

    fn materialize(&self) -> sim_kernel::Result<Arc<dyn TensorStorage>> {
        self.materialized
            .get_or_init(|| {
                let mut state = self.executor.state.lock().expect("modeled state poisoned");
                state.readbacks += 1;
                if self.executor.fault == ModeledFault::Readback {
                    state.materialization_failures += 1;
                    return Err(Error::Eval("modeled compute readback failed".to_owned()));
                }
                Ok(Arc::new(HostStorage {
                    dtype: self.dtype.clone(),
                    cells: self.cells.clone(),
                }) as Arc<dyn TensorStorage>)
            })
            .clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct HostStorage {
    dtype: Symbol,
    cells: Arc<[Value]>,
}

impl TensorStorage for HostStorage {
    fn dtype(&self) -> &Symbol {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.cells.len()
    }

    fn location(&self) -> TensorLocation {
        TensorLocation::Host
    }

    fn cell(&self, index: usize) -> sim_kernel::Result<Value> {
        self.cells
            .get(index)
            .cloned()
            .ok_or_else(|| Error::Eval("modeled Tensor index out of bounds".to_owned()))
    }

    fn materialize(&self) -> sim_kernel::Result<Arc<dyn TensorStorage>> {
        Ok(Arc::new(Self {
            dtype: self.dtype.clone(),
            cells: self.cells.clone(),
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn runtime_cx(solver: TensorStudySolver) -> Cx {
    let mut cx = Cx::new(
        Arc::new(EagerPolicy),
        Arc::new(DefaultFactory),
        sim_kernel::HandleSeed::new(0x7d10_1806_df9a_bcba),
    );
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_float::F32NumbersLib::new())
        .unwrap();
    cx.load_lib(&TensorNumbersLib::new()).unwrap();
    cx.load_lib(&InterferenceRecordsLib).unwrap();
    cx.load_lib(&InterferenceLib).unwrap();
    cx.load_lib(&InterferenceComputeLib::new(solver)).unwrap();
    cx
}

fn solve_through_site(
    solver: TensorStudySolver,
    executor: Arc<dyn TensorExecutor>,
) -> sim_kernel::Result<(StudyDescriptor, Cx)> {
    let mut cx = runtime_cx(solver);
    let problem = modeled_problem();
    let plane = modeled_plane(8, 8);
    let problem_name = Symbol::qualified("test", "problem");
    let plane_name = Symbol::qualified("test", "plane");
    let problem_value = cx
        .factory()
        .opaque(Arc::new(ProblemDescriptor::from_problem(&problem)))?;
    let plane_value = cx
        .factory()
        .opaque(Arc::new(PlaneDescriptor::from_plane(plane)))?;
    cx.env_mut().define(problem_name.clone(), problem_value);
    cx.env_mut().define(plane_name.clone(), plane_value);

    let site = TensorSite::new(modeled_site_symbol(), executor, Vec::new());
    let reply = site.realize(
        &mut cx,
        eval_request(Expr::Call {
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
        }),
    )?;
    let study = reply
        .value
        .object()
        .downcast_ref::<StudyDescriptor>()
        .cloned()
        .ok_or_else(|| Error::Eval("TensorSite did not return an interference Study".to_owned()))?;
    Ok((study, cx))
}

fn solve_request<'a>(
    problem: &'a InterferenceProblem,
    plane: &'a SamplingPlane,
) -> SolveRequest<'a> {
    SolveRequest::new(
        problem,
        plane,
        SamplingPolicy::Annotate,
        SamplingThresholds::default(),
        WorkBudget::default(),
    )
}

fn modeled_problem() -> InterferenceProblem {
    InterferenceProblem::new(
        Hertz::new(343.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(343.0).unwrap(),
            NepersPerMetre::new(0.01).unwrap(),
        ),
        SourceSet::new(vec![Emitter::Point {
            id: "source".to_owned(),
            position: Point3M::from_metres(0.0, 0.0, 0.0).unwrap(),
            amplitude_at_reference: FieldAmplitude::new(1.0).unwrap(),
            phase: Radians::new(0.25).unwrap(),
        }])
        .unwrap(),
        PositiveMetres::new(0.01).unwrap(),
    )
}

fn modeled_plane(rows: usize, columns: usize) -> SamplingPlane {
    SamplingPlane::new(
        Point3M::from_metres(1.0, -0.125, -0.125).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
        PositiveMetres::new(0.25).unwrap(),
        PositiveMetres::new(0.25).unwrap(),
        rows,
        columns,
    )
    .unwrap()
}

fn eval_request(expr: Expr) -> EvalRequest {
    EvalRequest {
        expr,
        result_shape: None,
        required_capabilities: Vec::new(),
        deadline: None,
        consistency: Consistency::LocalFirst,
        mode: EvalMode::Eval,
        answer_limit: None,
        stream_buffer: None,
        stream: false,
        trace: false,
    }
}
