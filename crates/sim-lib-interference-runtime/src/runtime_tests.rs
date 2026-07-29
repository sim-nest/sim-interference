//! Runtime library, callable Shape, provider-isolation, and exact recipe tests.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use sim_citizen::CitizenField;
use sim_kernel::{
    Args, Cx, DefaultFactory, EagerPolicy, Env, Error, Export, Expr, Lib, Result, Symbol, Value,
};

use crate::{
    InterferenceLib, InterferenceRecordsLib, RECIPES, SolveRequest, SolverProvider,
    StudyDescriptor, StudySolver, analyze_function_symbol, multitone_function_symbol,
    problem_function_symbol, project_function_symbol, sampling_plane_function_symbol,
    scenarios_function_symbol, solve_function_symbol, study_solver_symbol,
};

#[test]
fn interference_lib_exports_and_runs_every_runtime_operation() {
    let mut cx = interference_cx();
    let problem = call_problem(&mut cx, 1_000.0);
    let second_problem = call_problem(&mut cx, 1_200.0);
    let plane = runtime_plane(&mut cx, 4, 4);
    let solve_options = runtime_value(
        &mut cx,
        map([
            ("sampling", Expr::Symbol(Symbol::new("annotate"))),
            ("work-budget", Expr::Symbol(Symbol::new("default"))),
        ]),
    );
    let study = cx
        .call_function(
            &solve_function_symbol(),
            Args::new(vec![problem.clone(), plane.clone(), solve_options]),
        )
        .unwrap();
    assert!(study.object().downcast_ref::<StudyDescriptor>().is_some());

    let projection_request = runtime_value(
        &mut cx,
        map([
            ("observable", Expr::Symbol(Symbol::new("amplitude"))),
            ("reduction", Expr::Symbol(Symbol::new("detail"))),
            ("target-rows", field_expr(&4_usize)),
            ("target-cols", field_expr(&4_usize)),
        ]),
    );
    let projection = cx
        .call_function(
            &project_function_symbol(),
            Args::new(vec![study.clone(), projection_request]),
        )
        .unwrap();
    assert!(
        projection
            .object()
            .downcast_ref::<crate::ScalarProjectionDescriptor>()
            .is_some()
    );

    let analyze_request = runtime_value(
        &mut cx,
        map([
            ("observable", Expr::Symbol(Symbol::new("amplitude"))),
            ("reduction", Expr::Symbol(Symbol::new("detail"))),
            ("target-rows", field_expr(&4_usize)),
            ("target-cols", field_expr(&4_usize)),
            ("amplitude-floor", field_expr(&0.0)),
        ]),
    );
    let analysis = cx
        .call_function(
            &analyze_function_symbol(),
            Args::new(vec![study, analyze_request]),
        )
        .unwrap();
    assert!(analysis.object().as_table_impl().is_some());

    let scenario_spec = runtime_value(
        &mut cx,
        map([
            ("kind", Expr::Symbol(Symbol::new("two-point"))),
            ("frequency-hz", field_expr(&1_000.0)),
            ("speed-m-s", field_expr(&343.0)),
            ("attenuation-np-m", field_expr(&0.0)),
            ("singularity-radius-m", field_expr(&0.01)),
            ("first-m", vector([-0.25, 0.0, 0.0])),
            ("second-m", vector([0.25, 0.0, 0.0])),
            ("amplitude-per-source", field_expr(&1.0)),
            ("relative-phase-rad", field_expr(&std::f64::consts::PI)),
        ]),
    );
    let scenario = cx
        .call_function(&scenarios_function_symbol(), Args::new(vec![scenario_spec]))
        .unwrap();
    assert!(scenario.object().as_table_impl().is_some());

    let problems = cx.factory().list(vec![problem, second_problem]).unwrap();
    let multitone_options = runtime_value(
        &mut cx,
        map([
            (
                "weights",
                Expr::List(vec![field_expr(&1.0), field_expr(&0.5)]),
            ),
            (
                "combination",
                Expr::Symbol(Symbol::new("incoherent-magnitude-squared")),
            ),
            ("sampling", Expr::Symbol(Symbol::new("annotate"))),
            ("work-budget", Expr::Symbol(Symbol::new("default"))),
        ]),
    );
    let multitone = cx
        .call_function(
            &multitone_function_symbol(),
            Args::new(vec![problems, plane, multitone_options]),
        )
        .unwrap();
    assert!(multitone.object().as_table_impl().is_some());

    let manifest = InterferenceLib.manifest();
    for symbol in [
        problem_function_symbol(),
        sampling_plane_function_symbol(),
        solve_function_symbol(),
        project_function_symbol(),
        analyze_function_symbol(),
        scenarios_function_symbol(),
        multitone_function_symbol(),
    ] {
        assert!(
            manifest.exports.iter().any(
                |export| matches!(export, Export::Function { symbol: found, .. } if found == &symbol)
            ),
            "missing runtime export {symbol}"
        );
    }
}

#[test]
fn runtime_shapes_check_arguments_results_and_project_never_propagates() {
    struct UnusableSolver {
        calls: Arc<AtomicUsize>,
        result: Option<StudyDescriptor>,
    }

    impl StudySolver for UnusableSolver {
        fn solve(&self, _cx: &mut Cx, _request: &SolveRequest<'_>) -> Result<StudyDescriptor> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.result
                .clone()
                .ok_or_else(|| Error::Eval("solver must not be called".to_owned()))
        }
    }

    let mut cx = interference_cx();
    let problem = call_problem(&mut cx, 1_000.0);
    let plane = runtime_plane(&mut cx, 4, 4);
    let options = runtime_value(
        &mut cx,
        map([
            ("sampling", Expr::Symbol(Symbol::new("annotate"))),
            ("work-budget", Expr::Symbol(Symbol::new("default"))),
        ]),
    );
    let study = cx
        .call_function(
            &solve_function_symbol(),
            Args::new(vec![problem.clone(), plane.clone(), options.clone()]),
        )
        .unwrap();

    for symbol in [
        problem_function_symbol(),
        sampling_plane_function_symbol(),
        solve_function_symbol(),
        project_function_symbol(),
        analyze_function_symbol(),
        scenarios_function_symbol(),
        multitone_function_symbol(),
    ] {
        let function = cx.registry().function_by_symbol(&symbol).unwrap().clone();
        let callable = function.object().as_callable().unwrap();
        assert!(callable.browse_args_shape(&mut cx).unwrap().is_some());
        assert!(callable.browse_result_shape(&mut cx).unwrap().is_some());
    }

    let wrong_shape = cx
        .call_function(
            &solve_function_symbol(),
            Args::new(vec![plane.clone(), plane, options]),
        )
        .unwrap_err()
        .to_string();
    assert!(
        wrong_shape.contains("shape")
            || wrong_shape.contains("matching function case")
            || wrong_shape.contains("matching overload")
            || wrong_shape.contains("interference/Problem"),
        "unexpected argument-shape error: {wrong_shape}"
    );

    let calls = Arc::new(AtomicUsize::new(0));
    let mut child = Env::child(Arc::new(cx.env().clone()));
    child.define(
        study_solver_symbol(),
        SolverProvider::new(Arc::new(UnusableSolver {
            calls: calls.clone(),
            result: None,
        }))
        .into_value()
        .unwrap(),
    );
    cx.with_env(child, |cx| {
        let request = runtime_value(
            cx,
            map([
                ("observable", Expr::Symbol(Symbol::new("instant"))),
                ("wt", field_expr(&std::f64::consts::FRAC_PI_2)),
                (
                    "reduction",
                    Expr::Symbol(Symbol::new("detector-complex-mean")),
                ),
                ("target-rows", field_expr(&2_usize)),
                ("target-cols", field_expr(&2_usize)),
            ]),
        );
        cx.call_function(
            &project_function_symbol(),
            Args::new(vec![study.clone(), request]),
        )?;
        Ok(())
    })
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut malformed = study
        .object()
        .downcast_ref::<StudyDescriptor>()
        .unwrap()
        .clone();
    malformed.field.rows += 1;
    let mut child = Env::child(Arc::new(cx.env().clone()));
    child.define(
        study_solver_symbol(),
        SolverProvider::new(Arc::new(UnusableSolver {
            calls: Arc::new(AtomicUsize::new(0)),
            result: Some(malformed),
        }))
        .into_value()
        .unwrap(),
    );
    let result_error = cx
        .with_env(child, |cx| {
            let plane = runtime_plane(cx, 4, 4);
            let options = runtime_value(
                cx,
                map([
                    ("sampling", Expr::Symbol(Symbol::new("annotate"))),
                    ("work-budget", Expr::Symbol(Symbol::new("default"))),
                ]),
            );
            cx.call_function(
                &solve_function_symbol(),
                Args::new(vec![problem, plane, options]),
            )
        })
        .unwrap_err();
    assert!(result_error.to_string().contains("shape"));
}

#[test]
fn exact_cpu_lisp_recipe_is_embedded_and_runs_through_the_checked_runtime() {
    let source = include_str!("../recipes/01-basics/two-source-cancellation/setup.siml");
    let cards = sim_cookbook::recipes_from_embedded(RECIPES).unwrap();
    let card = cards
        .iter()
        .find(|card| card.id.ends_with("two-source-cancellation"))
        .expect("embedded CPU recipe");
    assert_eq!(card.setup, source.as_bytes());
    assert_eq!(card.codec, "lisp");

    let mut cx = interference_cx();
    let problem = call_problem(&mut cx, 1_000.0);
    let plane = runtime_plane(&mut cx, 1_024, 1_024);
    let options = runtime_value(
        &mut cx,
        map([
            ("sampling", Expr::Symbol(Symbol::new("strict"))),
            ("work-budget", Expr::Symbol(Symbol::new("default"))),
        ]),
    );
    let study = cx
        .call_function(
            &solve_function_symbol(),
            Args::new(vec![problem, plane, options]),
        )
        .unwrap();
    let study = study
        .object()
        .downcast_ref::<StudyDescriptor>()
        .expect("Shape-checked interference/Study");
    assert_eq!((study.field.rows, study.field.cols), (1_024, 1_024));
    assert_eq!(study.evidence.work.cells, 1_048_576);
    assert_eq!(study.evidence.work.emitter_evaluations, 2_097_152);
}

fn interference_cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&InterferenceRecordsLib).unwrap();
    cx.load_lib(&InterferenceLib).unwrap();
    cx
}

fn call_problem(cx: &mut Cx, frequency_hz: f64) -> Value {
    let spec = runtime_value(
        cx,
        map([
            ("frequency-hz", field_expr(&frequency_hz)),
            ("speed-m-s", field_expr(&343.0)),
            ("attenuation-np-m", field_expr(&0.0)),
            ("singularity-radius-m", field_expr(&0.01)),
            (
                "sources",
                Expr::List(vec![
                    map([
                        ("kind", Expr::Symbol(Symbol::new("point"))),
                        ("id", Expr::Symbol(Symbol::new("left"))),
                        ("position-m", vector([-0.25, 0.0, 0.0])),
                        ("amplitude-at-reference", field_expr(&1.0)),
                        ("phase-rad", field_expr(&0.0)),
                    ]),
                    map([
                        ("kind", Expr::Symbol(Symbol::new("point"))),
                        ("id", Expr::Symbol(Symbol::new("right"))),
                        ("position-m", vector([0.25, 0.0, 0.0])),
                        ("amplitude-at-reference", field_expr(&1.0)),
                        ("phase-rad", field_expr(&std::f64::consts::PI)),
                    ]),
                ]),
            ),
        ]),
    );
    cx.call_function(&problem_function_symbol(), Args::new(vec![spec]))
        .unwrap()
}

fn runtime_plane(cx: &mut Cx, rows: usize, cols: usize) -> Value {
    let spec = runtime_value(
        cx,
        map([
            ("origin-m", vector([-1.0, -1.0, 0.5])),
            ("u-axis", vector([1.0, 0.0, 0.0])),
            ("v-axis", vector([0.0, 1.0, 0.0])),
            ("extent-u-m", field_expr(&2.0)),
            ("extent-v-m", field_expr(&2.0)),
            ("rows", field_expr(&rows)),
            ("cols", field_expr(&cols)),
        ]),
    );
    cx.call_function(&sampling_plane_function_symbol(), Args::new(vec![spec]))
        .unwrap()
}

fn runtime_value(cx: &mut Cx, expr: Expr) -> Value {
    sim_citizen::value_from_expr(cx, &expr).unwrap()
}

fn field_expr<T: CitizenField>(value: &T) -> Expr {
    value.encode_field()
}

fn vector<const N: usize>(values: [f64; N]) -> Expr {
    Expr::List(values.into_iter().map(|value| field_expr(&value)).collect())
}

fn map<const N: usize>(entries: [(&str, Expr); N]) -> Expr {
    Expr::Map(
        entries
            .into_iter()
            .map(|(key, value)| (Expr::Symbol(Symbol::new(key)), value))
            .collect(),
    )
}
