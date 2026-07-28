use std::sync::Arc;

use sim_citizen::CitizenField;
use sim_kernel::{Args, Cx, DefaultFactory, EagerPolicy, Expr, Symbol, Value};
use sim_lib_interference_runtime::{
    InterferenceLib, InterferenceRecordsLib, StudyDescriptor, problem_function_symbol,
    sampling_plane_function_symbol, solve_function_symbol,
};

fn main() -> sim_kernel::Result<()> {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&InterferenceRecordsLib)?;
    cx.load_lib(&InterferenceLib)?;

    let problem_spec = value(
        &mut cx,
        map([
            ("frequency-hz", field(&1_000.0)),
            ("speed-m-s", field(&343.0)),
            ("attenuation-np-m", field(&0.0)),
            ("singularity-radius-m", field(&0.01)),
            (
                "sources",
                Expr::List(vec![
                    map([
                        ("kind", Expr::Symbol(Symbol::new("point"))),
                        ("id", Expr::Symbol(Symbol::new("left"))),
                        ("position-m", vector([-0.25, 0.0, 0.0])),
                        ("amplitude-at-reference", field(&1.0)),
                        ("phase-rad", field(&0.0)),
                    ]),
                    map([
                        ("kind", Expr::Symbol(Symbol::new("point"))),
                        ("id", Expr::Symbol(Symbol::new("right"))),
                        ("position-m", vector([0.25, 0.0, 0.0])),
                        ("amplitude-at-reference", field(&1.0)),
                        ("phase-rad", field(&std::f64::consts::PI)),
                    ]),
                ]),
            ),
        ]),
    )?;
    let problem = cx.call_function(&problem_function_symbol(), Args::new(vec![problem_spec]))?;

    let plane_spec = value(
        &mut cx,
        map([
            ("origin-m", vector([-1.0, -1.0, 0.5])),
            ("u-axis", vector([1.0, 0.0, 0.0])),
            ("v-axis", vector([0.0, 1.0, 0.0])),
            ("extent-u-m", field(&2.0)),
            ("extent-v-m", field(&2.0)),
            ("rows", field(&1_024_usize)),
            ("cols", field(&1_024_usize)),
        ]),
    )?;
    let plane = cx.call_function(
        &sampling_plane_function_symbol(),
        Args::new(vec![plane_spec]),
    )?;
    let options = value(
        &mut cx,
        map([
            ("sampling", Expr::Symbol(Symbol::new("strict"))),
            ("work-budget", Expr::Symbol(Symbol::new("default"))),
        ]),
    )?;
    let study = cx.call_function(
        &solve_function_symbol(),
        Args::new(vec![problem, plane, options]),
    )?;
    let study = study
        .object()
        .downcast_ref::<StudyDescriptor>()
        .expect("interference/solve result Shape");
    println!(
        "interference/Study rows={} cols={} emitters={} cells={} evaluations={} verdict={}",
        study.field.rows,
        study.field.cols,
        study.evidence.work.emitters,
        study.evidence.work.cells,
        study.evidence.work.emitter_evaluations,
        study.evidence.sampling.verdict.name
    );
    Ok(())
}

fn value(cx: &mut Cx, expr: Expr) -> sim_kernel::Result<Value> {
    sim_citizen::value_from_expr(cx, &expr)
}

fn field<T: CitizenField>(value: &T) -> Expr {
    value.encode_field()
}

fn vector<const N: usize>(values: [f64; N]) -> Expr {
    Expr::List(values.into_iter().map(|value| field(&value)).collect())
}

fn map<const N: usize>(entries: [(&str, Expr); N]) -> Expr {
    Expr::Map(
        entries
            .into_iter()
            .map(|(key, value)| (Expr::Symbol(Symbol::new(key)), value))
            .collect(),
    )
}
