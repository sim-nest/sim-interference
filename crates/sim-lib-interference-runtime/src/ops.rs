//! Shape-checked runtime operations for coherent interference studies.

use std::sync::Arc;

use sim_citizen::CitizenField;
use sim_kernel::{
    Cx, Demand, Error, Expr, ExprKind, LoadCx, PreparedArgs, Result, Symbol, Value,
    force_list_to_vec,
};
use sim_lib_interference_core::{
    Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, PositiveMetres, SamplingPlane,
    ScalarMedium, SourceSet,
};
use sim_lib_interference_solve::{
    MultiToneStudy, Observable, ReductionRule, ReferencePhasorSolver, ToneCombination, ToneStudy,
    analyze_fringes, reduce_for_view,
};
use sim_shape::{
    Bindings, ExprKindShape, FunctionCase, FunctionObject, ListShape, RepeatShape, Shape,
};

use crate::{
    PlaneDescriptor, ProblemDescriptor, ProjectionRequestDescriptor, ScalarProjectionDescriptor,
    SolveRequest, StudyDescriptor, resolve_study_solver,
    shapes::{plane_shape, problem_shape, projection_shape, study_shape},
};
use crate::{ops_outputs::*, ops_values::*};

/// Returns the runtime constructor symbol for coherent problems.
pub fn problem_function_symbol() -> Symbol {
    Symbol::qualified("interference", "problem")
}

/// Returns the runtime constructor symbol for physical sampling planes.
pub fn sampling_plane_function_symbol() -> Symbol {
    Symbol::qualified("interference", "sampling-plane")
}

/// Returns the provider-routed study solve symbol.
pub fn solve_function_symbol() -> Symbol {
    Symbol::qualified("interference", "solve")
}

/// Returns the propagation-free scalar projection symbol.
pub fn project_function_symbol() -> Symbol {
    Symbol::qualified("interference", "project")
}

/// Returns the certified fringe-analysis symbol.
pub fn analyze_function_symbol() -> Symbol {
    Symbol::qualified("interference", "analyze")
}

/// Returns the bounded named-scenario constructor symbol.
pub fn scenarios_function_symbol() -> Symbol {
    Symbol::qualified("interference", "scenarios")
}

/// Returns the independently certified multi-tone composition symbol.
pub fn multitone_function_symbol() -> Symbol {
    Symbol::qualified("interference", "multitone")
}

pub(crate) fn function_symbols() -> [Symbol; 7] {
    [
        problem_function_symbol(),
        sampling_plane_function_symbol(),
        solve_function_symbol(),
        project_function_symbol(),
        analyze_function_symbol(),
        scenarios_function_symbol(),
        multitone_function_symbol(),
    ]
}

pub(crate) fn runtime_functions(cx: &mut LoadCx) -> Vec<(Symbol, FunctionObject)> {
    vec![
        function(
            cx,
            problem_function_symbol(),
            vec![map_shape()],
            problem_shape(),
            problem_impl,
        ),
        function(
            cx,
            sampling_plane_function_symbol(),
            vec![map_shape()],
            plane_shape(),
            sampling_plane_impl,
        ),
        function(
            cx,
            solve_function_symbol(),
            vec![problem_shape(), plane_shape(), map_shape()],
            study_shape(),
            solve_impl,
        ),
        function(
            cx,
            project_function_symbol(),
            vec![study_shape(), map_shape()],
            projection_shape(),
            project_impl,
        ),
        function(
            cx,
            analyze_function_symbol(),
            vec![study_shape(), map_shape()],
            map_shape(),
            analyze_impl,
        ),
        function(
            cx,
            scenarios_function_symbol(),
            vec![map_shape()],
            map_shape(),
            scenarios_impl,
        ),
        function(
            cx,
            multitone_function_symbol(),
            vec![
                Arc::new(RepeatShape::with_bounds(problem_shape(), 1, None)),
                plane_shape(),
                map_shape(),
            ],
            map_shape(),
            multitone_impl,
        ),
    ]
}

fn function(
    cx: &mut LoadCx,
    symbol: Symbol,
    args: Vec<Arc<dyn Shape>>,
    result: Arc<dyn Shape>,
    implementation: fn(&mut Cx, &PreparedArgs, Bindings) -> Result<Value>,
) -> (Symbol, FunctionObject) {
    let callable = FunctionObject::new(
        cx.fresh_function_id(),
        symbol.clone(),
        vec![FunctionCase {
            id: cx.fresh_case_id(),
            name: Symbol::qualified(symbol.to_string(), "checked"),
            args: Arc::new(ListShape::tuple(args.clone())),
            result: Some(result),
            demand: vec![Demand::Value; args.len()],
            priority: 10,
            implementation,
        }],
    );
    (symbol, callable)
}

fn map_shape() -> Arc<dyn Shape> {
    Arc::new(ExprKindShape::new(ExprKind::Map))
}

fn problem_impl(cx: &mut Cx, args: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let [spec] = args.values() else {
        return arity("interference/problem", 1, args.len());
    };
    let problem = build_problem(cx, spec.clone())?;
    boxed(cx, problem)
}

fn sampling_plane_impl(cx: &mut Cx, args: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let [spec] = args.values() else {
        return arity("interference/sampling-plane", 1, args.len());
    };
    let plane = build_plane(cx, spec.clone())?;
    boxed(cx, plane)
}

fn solve_impl(cx: &mut Cx, args: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let [problem, plane, options] = args.values() else {
        return arity("interference/solve", 3, args.len());
    };
    let problem = descriptor::<ProblemDescriptor>(problem, "interference/solve problem")?;
    let plane = descriptor::<PlaneDescriptor>(plane, "interference/solve plane")?;
    let options = map_from_value(cx, options.clone(), "interference/solve options")?;
    reject_extra(
        &options,
        &["sampling", "sampling-thresholds", "work-budget"],
        "interference/solve options",
    )?;
    let config = solve_config(&options)?;
    let problem = problem.to_problem()?;
    let plane = plane.to_plane()?;
    let request = SolveRequest::new(
        &problem,
        &plane,
        config.sampling_policy,
        config.sampling_thresholds,
        config.work_budget,
    );
    let solver = resolve_study_solver(cx)?;
    let study = solver.solve(cx, &request)?;
    boxed(cx, study)
}

fn project_impl(cx: &mut Cx, args: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let [study, request] = args.values() else {
        return arity("interference/project", 2, args.len());
    };
    let study = descriptor::<StudyDescriptor>(study, "interference/project study")?;
    let request = projection_request(cx, request.clone(), study)?;
    let projection = project_study(cx, study, &request)?;
    boxed(cx, projection)
}

fn analyze_impl(cx: &mut Cx, args: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let [study, request] = args.values() else {
        return arity("interference/analyze", 2, args.len());
    };
    let study = descriptor::<StudyDescriptor>(study, "interference/analyze study")?;
    let request_map = map_from_value(cx, request.clone(), "interference/analyze request")?;
    reject_extra(
        &request_map,
        &[
            "observable",
            "wt",
            "phase-floor",
            "target-rows",
            "target-cols",
            "reduction",
            "amplitude-floor",
        ],
        "interference/analyze request",
    )?;
    let amplitude_floor = optional_decode::<f64>(&request_map, "amplitude-floor")?.unwrap_or(0.0);
    let request = projection_request_from_map(&request_map, study)?;
    let projection = project_domain(cx, study, &request)?;
    let report = analyze_fringes(&projection, amplitude_floor)
        .map_err(|error| Error::Eval(format!("interference analysis failed: {error}")))?;
    fringe_report_value(cx, &report)
}

fn scenarios_impl(cx: &mut Cx, args: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let [request] = args.values() else {
        return arity("interference/scenarios", 1, args.len());
    };
    let map = map_from_value(cx, request.clone(), "interference/scenarios request")?;
    let scenario = build_scenario(&map)?;
    scenario_value(cx, scenario)
}

fn multitone_impl(cx: &mut Cx, args: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let [problems, plane, options] = args.values() else {
        return arity("interference/multitone", 3, args.len());
    };
    let problem_values = problems
        .object()
        .as_list()
        .ok_or_else(|| Error::Eval("interference/multitone problems must be a list".to_owned()))
        .and_then(|list| force_list_to_vec(cx, list, "interference/multitone problems"))?;
    let problems = problem_values
        .iter()
        .map(|value| {
            descriptor::<ProblemDescriptor>(value, "interference/multitone problem")?.to_problem()
        })
        .collect::<Result<Vec<_>>>()?;
    let plane = descriptor::<PlaneDescriptor>(plane, "interference/multitone plane")?.to_plane()?;
    let options = map_from_value(cx, options.clone(), "interference/multitone options")?;
    reject_extra(
        &options,
        &[
            "weights",
            "combination",
            "seconds",
            "sampling",
            "sampling-thresholds",
            "work-budget",
        ],
        "interference/multitone options",
    )?;
    let weights = required_list(&options, "weights", "interference/multitone options")?
        .iter()
        .map(|value| f64::decode_field_expr(value, "weight"))
        .collect::<Result<Vec<_>>>()?;
    if weights.len() != problems.len() {
        return Err(Error::Eval(format!(
            "interference/multitone requires one weight per problem: {} problems, {} weights",
            problems.len(),
            weights.len()
        )));
    }
    let config = solve_config(&options)?;
    let solver = ReferencePhasorSolver::new(
        config.sampling_policy,
        config.sampling_thresholds,
        config.work_budget,
    );
    let tones = problems
        .into_iter()
        .zip(weights)
        .map(|(problem, weight)| ToneStudy::solve(problem, plane, weight, solver))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| Error::Eval(format!("interference multi-tone solve failed: {error}")))?;
    let study = MultiToneStudy::new(tones).map_err(|error| {
        Error::Eval(format!("interference multi-tone admission failed: {error}"))
    })?;
    let combination = match required_symbol(&options, "combination", "multitone combination")?
        .name
        .as_ref()
    {
        "incoherent-magnitude-squared" => ToneCombination::IncoherentMagnitudeSquared,
        "instant" => ToneCombination::Instant {
            seconds: required_decode(&options, "seconds")?,
        },
        other => {
            return Err(Error::Eval(format!(
                "unknown interference/multitone combination {other}"
            )));
        }
    };
    let projection = study
        .combine(combination)
        .map_err(|error| Error::Eval(format!("interference multi-tone combine failed: {error}")))?;
    multitone_value(cx, &projection)
}

fn build_problem(cx: &mut Cx, spec: Value) -> Result<ProblemDescriptor> {
    let map = map_from_value(cx, spec, "interference/problem")?;
    reject_extra(
        &map,
        &[
            "frequency-hz",
            "speed-m-s",
            "attenuation-np-m",
            "singularity-radius-m",
            "sources",
        ],
        "interference/problem",
    )?;
    let medium = ScalarMedium::new(
        MetresPerSecond::new(required_decode(&map, "speed-m-s")?)
            .map_err(domain_error("interference/problem speed-m-s"))?,
        NepersPerMetre::new(required_decode(&map, "attenuation-np-m")?)
            .map_err(domain_error("interference/problem attenuation-np-m"))?,
    );
    let sources = required_list(&map, "sources", "interference/problem")?
        .iter()
        .map(build_emitter)
        .collect::<Result<Vec<_>>>()?;
    let problem = InterferenceProblem::new(
        Hertz::new(required_decode(&map, "frequency-hz")?)
            .map_err(domain_error("interference/problem frequency-hz"))?,
        medium,
        SourceSet::new(sources).map_err(domain_error("interference/problem sources"))?,
        PositiveMetres::new(required_decode(&map, "singularity-radius-m")?)
            .map_err(domain_error("interference/problem singularity-radius-m"))?,
    );
    Ok(ProblemDescriptor::from_problem(&problem))
}

fn build_plane(cx: &mut Cx, spec: Value) -> Result<PlaneDescriptor> {
    let map = map_from_value(cx, spec, "interference/sampling-plane")?;
    reject_extra(
        &map,
        &[
            "origin-m",
            "u-axis",
            "v-axis",
            "extent-u-m",
            "extent-v-m",
            "rows",
            "cols",
        ],
        "interference/sampling-plane",
    )?;
    let plane = SamplingPlane::new(
        point3(
            required_field(&map, "origin-m", "interference/sampling-plane")?,
            "origin-m",
        )?,
        unit_vector(
            required_field(&map, "u-axis", "interference/sampling-plane")?,
            "u-axis",
        )?,
        unit_vector(
            required_field(&map, "v-axis", "interference/sampling-plane")?,
            "v-axis",
        )?,
        PositiveMetres::new(required_decode(&map, "extent-u-m")?)
            .map_err(domain_error("sampling-plane extent-u-m"))?,
        PositiveMetres::new(required_decode(&map, "extent-v-m")?)
            .map_err(domain_error("sampling-plane extent-v-m"))?,
        required_decode(&map, "rows")?,
        required_decode(&map, "cols")?,
    )
    .map_err(domain_error("interference/sampling-plane"))?;
    Ok(PlaneDescriptor::from_plane(plane))
}

fn projection_request(
    cx: &mut Cx,
    value: Value,
    study: &StudyDescriptor,
) -> Result<ProjectionRequestDescriptor> {
    let map = map_from_value(cx, value, "interference/project request")?;
    reject_extra(
        &map,
        &[
            "observable",
            "wt",
            "phase-floor",
            "target-rows",
            "target-cols",
            "reduction",
        ],
        "interference/project request",
    )?;
    projection_request_from_map(&map, study)
}

fn projection_request_from_map(
    map: &[(Expr, Expr)],
    study: &StudyDescriptor,
) -> Result<ProjectionRequestDescriptor> {
    let observable = match required_symbol(map, "observable", "projection observable")?
        .name
        .as_ref()
    {
        "real" => Observable::Real,
        "imaginary" => Observable::Imaginary,
        "amplitude" => Observable::Amplitude,
        "phase" => Observable::Phase,
        "magnitude-squared" => Observable::MagnitudeSquared,
        "instant" => Observable::Instant {
            wt: required_decode(map, "wt")?,
        },
        other => {
            return Err(Error::Eval(format!(
                "unknown projection observable {other}"
            )));
        }
    };
    let reduction = match required_symbol(map, "reduction", "projection reduction")?
        .name
        .as_ref()
    {
        "detail" => ReductionRule::Detail,
        "detector-complex-mean" => ReductionRule::DetectorComplexMean,
        "detector-scalar-area-mean" => ReductionRule::DetectorScalarAreaMean,
        "detector-magnitude-squared-area-mean" => ReductionRule::DetectorMagnitudeSquaredAreaMean,
        other => return Err(Error::Eval(format!("unknown projection reduction {other}"))),
    };
    ProjectionRequestDescriptor::new(
        observable,
        optional_decode::<f64>(map, "phase-floor")?.unwrap_or(0.0),
        optional_decode::<usize>(map, "target-rows")?.unwrap_or(study.plane.rows),
        optional_decode::<usize>(map, "target-cols")?.unwrap_or(study.plane.columns),
        reduction,
    )
}

fn project_study(
    cx: &mut Cx,
    study: &StudyDescriptor,
    request: &ProjectionRequestDescriptor,
) -> Result<ScalarProjectionDescriptor> {
    let projection = project_domain(cx, study, request)?;
    ScalarProjectionDescriptor::from_projection(&projection)
}

fn project_domain(
    cx: &mut Cx,
    study: &StudyDescriptor,
    request: &ProjectionRequestDescriptor,
) -> Result<sim_lib_interference_solve::ScalarProjection> {
    let field = study.field.materialize_host(cx)?;
    reduce_for_view(
        &field,
        study.evidence.sampling.to_certificate()?,
        request.observable()?,
        request.phase_floor,
        request.target_rows,
        request.target_columns,
        request.reduction()?,
    )
    .map_err(|error| Error::Eval(format!("interference projection failed: {error}")))
}
