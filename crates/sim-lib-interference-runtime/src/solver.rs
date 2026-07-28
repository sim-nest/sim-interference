//! Late-bound interference-study solver contract and CPU reference provider.

use std::{any::Any, sync::Arc};

use sim_kernel::{Cx, DefaultFactory, Error, Factory, Object, Result, Symbol, Value};
use sim_lib_interference_core::{
    InterferenceProblem, SamplingPlane, SamplingPolicy, SamplingThresholds, WorkBudget,
};
use sim_lib_interference_solve::ReferencePhasorSolver;

use crate::StudyDescriptor;

/// Complete runtime study returned by a [`StudySolver`].
pub type InterferenceStudy = StudyDescriptor;

/// Immutable, fully checked request passed to a study-solver provider.
#[derive(Clone, Copy, Debug)]
pub struct SolveRequest<'a> {
    problem: &'a InterferenceProblem,
    plane: &'a SamplingPlane,
    sampling_policy: SamplingPolicy,
    sampling_thresholds: SamplingThresholds,
    work_budget: WorkBudget,
}

impl<'a> SolveRequest<'a> {
    /// Builds a request from checked domain values and explicit admission policy.
    pub fn new(
        problem: &'a InterferenceProblem,
        plane: &'a SamplingPlane,
        sampling_policy: SamplingPolicy,
        sampling_thresholds: SamplingThresholds,
        work_budget: WorkBudget,
    ) -> Self {
        Self {
            problem,
            plane,
            sampling_policy,
            sampling_thresholds,
            work_budget,
        }
    }

    /// Returns the coherent problem to solve.
    pub fn problem(self) -> &'a InterferenceProblem {
        self.problem
    }

    /// Returns the exact physical sampling plane.
    pub fn plane(self) -> &'a SamplingPlane {
        self.plane
    }

    /// Returns the sampling-admission policy.
    pub fn sampling_policy(self) -> SamplingPolicy {
        self.sampling_policy
    }

    /// Returns the sampling-classification thresholds.
    pub fn sampling_thresholds(self) -> SamplingThresholds {
        self.sampling_thresholds
    }

    /// Returns the complete pre-allocation work budget.
    pub fn work_budget(self) -> WorkBudget {
        self.work_budget
    }
}

/// Narrow provider seam for producing one complete interference study.
pub trait StudySolver: Send + Sync + 'static {
    /// Solves one admitted request or returns no partial study.
    fn solve(&self, cx: &mut Cx, request: &SolveRequest<'_>) -> Result<InterferenceStudy>;
}

/// Runtime value carrying one loadable [`StudySolver`].
#[derive(Clone)]
pub struct SolverProvider {
    solver: Arc<dyn StudySolver>,
}

impl SolverProvider {
    /// Wraps a loadable study solver.
    pub fn new(solver: Arc<dyn StudySolver>) -> Self {
        Self { solver }
    }

    /// Returns the wrapped solver.
    pub fn solver(&self) -> Arc<dyn StudySolver> {
        self.solver.clone()
    }

    /// Boxes this provider as a kernel value.
    pub fn into_value(self) -> Result<Value> {
        DefaultFactory.opaque(Arc::new(self))
    }
}

impl Object for SolverProvider {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok("#<interference-study-solver>".to_owned())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for SolverProvider {}

/// Stable binding and registry symbol for the active study solver.
pub fn study_solver_symbol() -> Symbol {
    Symbol::qualified("interference", "study-solver")
}

/// Resolves the active solver from the child environment, then the registry.
///
/// A present but malformed child binding is rejected rather than silently
/// bypassed. EvalFabric sites can therefore override the registry default
/// without changing the solve expression.
pub fn resolve_study_solver(cx: &Cx) -> Result<Arc<dyn StudySolver>> {
    if let Some(value) = cx.env().get(&study_solver_symbol()) {
        return solver_from_value(&value, "active environment");
    }
    let value = cx
        .registry()
        .value_by_symbol(&study_solver_symbol())
        .ok_or_else(|| Error::Eval(format!("no {} is installed", study_solver_symbol())))?;
    solver_from_value(value, "runtime registry")
}

fn solver_from_value(value: &Value, source: &str) -> Result<Arc<dyn StudySolver>> {
    value
        .object()
        .downcast_ref::<SolverProvider>()
        .map(SolverProvider::solver)
        .ok_or_else(|| {
            Error::Eval(format!(
                "{} in the {source} is not a SolverProvider",
                study_solver_symbol()
            ))
        })
}

/// Deterministic CPU `f64` implementation of [`StudySolver`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceStudySolver;

impl StudySolver for ReferenceStudySolver {
    fn solve(&self, _cx: &mut Cx, request: &SolveRequest<'_>) -> Result<InterferenceStudy> {
        let solver = ReferencePhasorSolver::new(
            request.sampling_policy(),
            request.sampling_thresholds(),
            request.work_budget(),
        );
        let (field, evidence) =
            solver
                .solve(request.problem(), request.plane())
                .map_err(|error| {
                    Error::Eval(format!("interference reference solve failed: {error}"))
                })?;
        StudyDescriptor::from_reference(request.problem(), *request.plane(), field, &evidence)
    }
}
