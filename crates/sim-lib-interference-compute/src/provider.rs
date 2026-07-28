//! Provider routing from the runtime study seam to canonical Tensor execution.

use std::sync::{Arc, Mutex};

use sim_kernel::{
    AbiVersion, Dependency, Error, Export, Lib, LibManifest, LibTarget, Linker, Result, Symbol,
    Version,
};
use sim_lib_interference_core::RequestPreflight;
use sim_lib_interference_runtime::{
    InterferenceStudy, PhasorFieldDescriptor, PlaneDescriptor, ProblemDescriptor,
    ReferenceStudySolver, SamplingCertificateDescriptor, SolveRequest, SolverProvider,
    StudyDescriptor, StudyEvidenceDescriptor, StudySolver, WorkEstimateDescriptor,
    tensor_study_solver_symbol,
};
use sim_lib_numbers_tensor::{TensorExecutorCard, active_tensor_executor, domains};

use crate::{
    DifferentialTolerances, LoweringPlan, PhaseBudget, TileProfile,
    preflight::required_operation_symbols, resident::execute_resident,
};

/// Stable library symbol for the accelerated interference provider.
pub fn interference_compute_lib_symbol() -> Symbol {
    Symbol::qualified("sim", "interference-compute")
}

/// A reason the Tensor provider was declined before any submission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuFallbackReason {
    /// No Tensor executor is active in the current environment.
    NoExecutor,
    /// An automatic executor selected its CPU route before submission.
    ProviderCpuChoice,
    /// The configured provider profile does not admit canonical f32.
    UnsupportedDtype,
    /// The executor card omits at least one operation required by the lowering.
    UnsupportedOperations,
    /// The admitted work is below the configured provider crossover.
    BelowCrossover,
    /// The provider limits would require host assembly of multiple output tiles.
    MultipleOutputTiles,
}

/// Last routing outcome observed by a [`TensorStudySolver`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderRoute {
    /// No request has reached this solver.
    Idle,
    /// CPU was selected before provider submission.
    Cpu(CpuFallbackReason),
    /// The named executor completed a resident Study.
    ProviderCompleted(Symbol),
    /// The named executor was selected and the request failed without restart.
    ProviderFailed(Symbol),
}

/// Stable solver counters and last route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorStudySnapshot {
    /// Requests routed to the deterministic CPU reference before submission.
    pub cpu_fallbacks: u64,
    /// Requests completed through an active Tensor executor.
    pub provider_completions: u64,
    /// Selected provider requests that failed without CPU restart.
    pub provider_failures: u64,
    /// Most recent routing outcome.
    pub last_route: ProviderRoute,
}

impl Default for TensorStudySnapshot {
    fn default() -> Self {
        Self {
            cpu_fallbacks: 0,
            provider_completions: 0,
            provider_failures: 0,
            last_route: ProviderRoute::Idle,
        }
    }
}

/// Admission, crossover, accuracy, and evidence settings for Tensor studies.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorStudyConfig {
    /// Smallest admitted cell count sent to an active provider.
    pub min_accelerated_cells: u64,
    /// Residual phase and predicted arithmetic limits.
    pub phase_budget: PhaseBudget,
    /// Provider allocation, segmentation, and dtype limits.
    pub tile_profile: TileProfile,
    /// Published fixed f32 comparison tolerances.
    pub tolerances: DifferentialTolerances,
    /// Stable adapter identity retained in Study evidence.
    pub adapter: String,
}

impl Default for TensorStudyConfig {
    fn default() -> Self {
        Self {
            min_accelerated_cells: 4_096,
            phase_budget: PhaseBudget::default(),
            tile_profile: TileProfile::default(),
            tolerances: DifferentialTolerances::default(),
            adapter: "interference-tensor-v1".to_owned(),
        }
    }
}

/// Study solver that preselects CPU or executes one resident Tensor plan.
#[derive(Clone)]
pub struct TensorStudySolver {
    config: TensorStudyConfig,
    snapshot: Arc<Mutex<TensorStudySnapshot>>,
}

impl TensorStudySolver {
    /// Builds a Tensor study solver from explicit routing settings.
    pub fn new(config: TensorStudyConfig) -> Self {
        Self {
            config,
            snapshot: Arc::new(Mutex::new(TensorStudySnapshot::default())),
        }
    }

    /// Returns a stable snapshot of routing outcomes.
    pub fn snapshot(&self) -> TensorStudySnapshot {
        self.snapshot
            .lock()
            .expect("Tensor study snapshot poisoned")
            .clone()
    }

    fn fallback(
        &self,
        cx: &mut sim_kernel::Cx,
        request: &SolveRequest<'_>,
        reason: CpuFallbackReason,
    ) -> Result<InterferenceStudy> {
        {
            let mut snapshot = self
                .snapshot
                .lock()
                .expect("Tensor study snapshot poisoned");
            snapshot.cpu_fallbacks = snapshot.cpu_fallbacks.saturating_add(1);
            snapshot.last_route = ProviderRoute::Cpu(reason);
        }
        ReferenceStudySolver.solve(cx, request)
    }

    fn fail(&self, executor: Symbol, error: impl core::fmt::Display) -> Error {
        let mut snapshot = self
            .snapshot
            .lock()
            .expect("Tensor study snapshot poisoned");
        snapshot.provider_failures = snapshot.provider_failures.saturating_add(1);
        snapshot.last_route = ProviderRoute::ProviderFailed(executor.clone());
        Error::Eval(format!(
            "interference Tensor provider {executor} failed after selection; CPU restart is forbidden: {error}"
        ))
    }

    fn card_is_eligible(&self, card: &TensorExecutorCard) -> bool {
        required_operation_symbols()
            .iter()
            .all(|required| card.operations.iter().any(|actual| actual == required))
    }
}

impl Default for TensorStudySolver {
    fn default() -> Self {
        Self::new(TensorStudyConfig::default())
    }
}

impl StudySolver for TensorStudySolver {
    fn solve(
        &self,
        cx: &mut sim_kernel::Cx,
        request: &SolveRequest<'_>,
    ) -> Result<InterferenceStudy> {
        let admitted = RequestPreflight::admit(
            request.problem(),
            request.plane(),
            request.sampling_policy(),
            request.sampling_thresholds(),
            request.work_budget(),
        )
        .map_err(|error| Error::Eval(format!("interference request admission failed: {error}")))?;
        let Some(executor) = active_tensor_executor(cx) else {
            return self.fallback(cx, request, CpuFallbackReason::NoExecutor);
        };
        let card = executor.card();
        if card.locality == Symbol::qualified("compute", "auto") && card.provider == "auto/cpu" {
            return self.fallback(cx, request, CpuFallbackReason::ProviderCpuChoice);
        }
        if !self.config.tile_profile.supports_f32 {
            return self.fallback(cx, request, CpuFallbackReason::UnsupportedDtype);
        }
        if !self.card_is_eligible(&card) {
            return self.fallback(cx, request, CpuFallbackReason::UnsupportedOperations);
        }
        if admitted.work_estimate.cells < self.config.min_accelerated_cells {
            return self.fallback(cx, request, CpuFallbackReason::BelowCrossover);
        }
        let plan = LoweringPlan::preflight_with_executor(
            cx,
            request.problem(),
            *request.plane(),
            self.config.phase_budget,
            self.config.tile_profile,
            executor,
        )
        .map_err(|error| self.fail(card.symbol.clone(), error))?;
        if plan.tile_plan().tiles().len() != 1 {
            return self.fallback(cx, request, CpuFallbackReason::MultipleOutputTiles);
        }
        let execution =
            execute_resident(cx, &plan).map_err(|error| self.fail(card.symbol.clone(), error))?;
        let tolerances = self.config.tolerances;
        let field = PhasorFieldDescriptor::new(
            request.plane().rows(),
            request.plane().columns(),
            execution.real,
            execution.imaginary,
        )
        .map_err(|error| self.fail(card.symbol.clone(), error))?;
        let evidence = StudyEvidenceDescriptor {
            sampling_policy: match request.sampling_policy() {
                sim_lib_interference_core::SamplingPolicy::Strict => {
                    Symbol::qualified("interference", "strict")
                }
                sim_lib_interference_core::SamplingPolicy::Annotate => {
                    Symbol::qualified("interference", "annotate")
                }
            },
            sampling: SamplingCertificateDescriptor::from_certificate(
                admitted.sampling_certificate,
            ),
            work: WorkEstimateDescriptor::from_estimate(admitted.work_estimate),
            provider: card.symbol.clone(),
            dtype: domains::f32(),
            component_absolute_tolerance: tolerances.component.absolute,
            squared_magnitude_absolute_tolerance: tolerances.magnitude_squared.absolute,
            completed_cells: admitted.work_estimate.cells,
            completed_emitter_evaluations: admitted.work_estimate.emitter_evaluations,
            uploads: execution.uploads,
            submissions: execution.submissions,
            intermediate_materializations: 0,
            final_materializations: 2,
            segments: execution.segments,
            adapter: self.config.adapter.clone(),
            profile: Some(card.provider),
        };
        let study = StudyDescriptor::new(
            ProblemDescriptor::from_problem(request.problem()),
            PlaneDescriptor::from_plane(*request.plane()),
            field,
            evidence,
        )
        .map_err(|error| self.fail(card.symbol.clone(), error))?;
        let mut snapshot = self
            .snapshot
            .lock()
            .expect("Tensor study snapshot poisoned");
        snapshot.provider_completions = snapshot.provider_completions.saturating_add(1);
        snapshot.last_route = ProviderRoute::ProviderCompleted(card.symbol);
        Ok(study)
    }
}

/// Loadable library that registers the Tensor study provider.
#[derive(Clone, Default)]
pub struct InterferenceComputeLib {
    solver: TensorStudySolver,
}

impl InterferenceComputeLib {
    /// Builds a provider library around a configured, observable solver.
    pub fn new(solver: TensorStudySolver) -> Self {
        Self { solver }
    }
}

impl Lib for InterferenceComputeLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: interference_compute_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![Dependency {
                id: Symbol::qualified("sim", "interference"),
                minimum_version: None,
            }],
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: tensor_study_solver_symbol(),
            }],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.value(
            tensor_study_solver_symbol(),
            SolverProvider::new(Arc::new(self.solver.clone())).into_value()?,
        )
    }
}
