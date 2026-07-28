//! Shared decoding and value-projection helpers for runtime operations.

use std::sync::Arc;

use sim_citizen::CitizenField;
use sim_kernel::{Cx, Error, Expr, Result, Symbol, Value};
use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Point3M, Radians, SamplingPolicy, SamplingThresholds, UnitVector3,
    WorkBudget,
};

#[derive(Clone, Copy)]
pub(crate) struct SolveConfig {
    pub(crate) sampling_policy: SamplingPolicy,
    pub(crate) sampling_thresholds: SamplingThresholds,
    pub(crate) work_budget: WorkBudget,
}

pub(crate) fn solve_config(map: &[(Expr, Expr)]) -> Result<SolveConfig> {
    let sampling_policy = match optional_symbol(map, "sampling")?
        .unwrap_or_else(|| Symbol::new("strict"))
        .name
        .as_ref()
    {
        "strict" => SamplingPolicy::Strict,
        "annotate" => SamplingPolicy::Annotate,
        other => return Err(Error::Eval(format!("unknown sampling policy {other}"))),
    };
    let sampling_thresholds = optional_field(map, "sampling-thresholds")
        .map(|value| {
            let value = expect_map(value, "sampling-thresholds")?;
            SamplingThresholds::new(
                required_decode(value, "resolved-min-samples-per-wavelength")?,
                required_decode(value, "marginal-min-samples-per-wavelength")?,
                required_decode(value, "resolved-max-envelope-fraction-per-cell")?,
                required_decode(value, "marginal-max-envelope-fraction-per-cell")?,
            )
            .map_err(domain_error("sampling-thresholds"))
        })
        .transpose()?
        .unwrap_or_default();
    let work_budget = match optional_field(map, "work-budget") {
        None => WorkBudget::default(),
        Some(Expr::Symbol(symbol)) if symbol.name.as_ref() == "default" => WorkBudget::default(),
        Some(value) => {
            let value = expect_map(value, "work-budget")?;
            reject_extra(
                value,
                &[
                    "max-cells",
                    "max-emitter-evaluations",
                    "max-host-bytes",
                    "max-result-bytes",
                    "max-certificate-stencil-work",
                ],
                "work-budget",
            )?;
            WorkBudget {
                max_cells: required_decode(value, "max-cells")?,
                max_emitter_evaluations: required_decode(value, "max-emitter-evaluations")?,
                max_host_bytes: required_decode(value, "max-host-bytes")?,
                max_result_bytes: required_decode(value, "max-result-bytes")?,
                max_certificate_stencil_work: required_decode(
                    value,
                    "max-certificate-stencil-work",
                )?,
            }
        }
    };
    Ok(SolveConfig {
        sampling_policy,
        sampling_thresholds,
        work_budget,
    })
}

pub(crate) fn point3(expr: &Expr, field: &'static str) -> Result<Point3M> {
    let [x, y, z] = vector3(expr, field)?;
    Point3M::from_metres(x, y, z).map_err(domain_error(field))
}

pub(crate) fn build_emitter(expr: &Expr) -> Result<Emitter> {
    let map = expect_map(expr, "interference/problem source")?;
    let kind = required_symbol(map, "kind", "interference/problem source")?;
    let id = symbol_or_string(
        required_field(map, "id", "interference/problem source")?,
        "source id",
    )?;
    let amplitude = FieldAmplitude::new(required_decode(map, "amplitude-at-reference")?)
        .map_err(domain_error("source amplitude-at-reference"))?;
    let phase = Radians::new(required_decode(map, "phase-rad")?)
        .map_err(domain_error("source phase-rad"))?;
    match kind.name.as_ref() {
        "point" => {
            reject_extra(
                map,
                &[
                    "kind",
                    "id",
                    "position-m",
                    "amplitude-at-reference",
                    "phase-rad",
                ],
                "interference point source",
            )?;
            Ok(Emitter::Point {
                id,
                position: point3(
                    required_field(map, "position-m", "point source")?,
                    "position-m",
                )?,
                amplitude_at_reference: amplitude,
                phase,
            })
        }
        "forward-plane" => {
            reject_extra(
                map,
                &[
                    "kind",
                    "id",
                    "through-m",
                    "direction",
                    "amplitude-at-reference",
                    "phase-rad",
                ],
                "interference forward-plane source",
            )?;
            Ok(Emitter::ForwardPlane {
                id,
                through: point3(
                    required_field(map, "through-m", "plane source")?,
                    "through-m",
                )?,
                direction: unit_vector(
                    required_field(map, "direction", "plane source")?,
                    "direction",
                )?,
                amplitude,
                phase,
            })
        }
        other => Err(Error::Eval(format!(
            "unknown interference source kind {other}"
        ))),
    }
}

pub(crate) fn unit_vector(expr: &Expr, field: &'static str) -> Result<UnitVector3> {
    let [x, y, z] = vector3(expr, field)?;
    UnitVector3::new(x, y, z).map_err(domain_error(field))
}

fn vector3(expr: &Expr, field: &'static str) -> Result<[f64; 3]> {
    let values = match expr {
        Expr::List(values) | Expr::Vector(values) => values,
        _ => return Err(Error::Eval(format!("{field} must be a three-number list"))),
    };
    let [x, y, z] = values.as_slice() else {
        return Err(Error::Eval(format!(
            "{field} must contain exactly three numbers"
        )));
    };
    Ok([
        f64::decode_field_expr(x, field)?,
        f64::decode_field_expr(y, field)?,
        f64::decode_field_expr(z, field)?,
    ])
}

pub(crate) fn map_from_value(
    cx: &mut Cx,
    value: Value,
    context: &'static str,
) -> Result<Vec<(Expr, Expr)>> {
    let expr = sim_citizen::value_to_expr(cx, value, context)?;
    let Expr::Map(entries) = expr else {
        return Err(Error::Eval(format!("{context} must be a map")));
    };
    Ok(entries)
}

pub(crate) fn expect_map<'a>(expr: &'a Expr, context: &str) -> Result<&'a [(Expr, Expr)]> {
    let Expr::Map(entries) = expr else {
        return Err(Error::Eval(format!("{context} must be a map")));
    };
    Ok(entries)
}

pub(crate) fn required_field<'a>(
    map: &'a [(Expr, Expr)],
    name: &str,
    context: &str,
) -> Result<&'a Expr> {
    optional_field(map, name)
        .ok_or_else(|| Error::Eval(format!("{context} is missing field {name}")))
}

pub(crate) fn optional_field<'a>(map: &'a [(Expr, Expr)], name: &str) -> Option<&'a Expr> {
    map.iter().find_map(|(key, value)| {
        matches!(key, Expr::Symbol(symbol) if symbol.namespace.is_none() && symbol.name.as_ref() == name)
            .then_some(value)
    })
}

pub(crate) fn required_list<'a>(
    map: &'a [(Expr, Expr)],
    name: &str,
    context: &str,
) -> Result<&'a [Expr]> {
    match required_field(map, name, context)? {
        Expr::List(values) | Expr::Vector(values) => Ok(values),
        _ => Err(Error::Eval(format!(
            "{context} field {name} must be a list"
        ))),
    }
}

pub(crate) fn required_symbol(map: &[(Expr, Expr)], name: &str, context: &str) -> Result<Symbol> {
    match required_field(map, name, context)? {
        Expr::Symbol(symbol) => Ok(symbol.clone()),
        _ => Err(Error::Eval(format!(
            "{context} field {name} must be a symbol"
        ))),
    }
}

fn optional_symbol(map: &[(Expr, Expr)], name: &str) -> Result<Option<Symbol>> {
    optional_field(map, name)
        .map(|value| match value {
            Expr::Symbol(symbol) => Ok(symbol.clone()),
            _ => Err(Error::Eval(format!("field {name} must be a symbol"))),
        })
        .transpose()
}

pub(crate) fn required_decode<T: CitizenField>(
    map: &[(Expr, Expr)],
    name: &'static str,
) -> Result<T> {
    T::decode_field_expr(required_field(map, name, "runtime request")?, name)
}

pub(crate) fn optional_decode<T: CitizenField>(
    map: &[(Expr, Expr)],
    name: &'static str,
) -> Result<Option<T>> {
    optional_field(map, name)
        .map(|value| T::decode_field_expr(value, name))
        .transpose()
}

pub(crate) fn reject_extra(map: &[(Expr, Expr)], allowed: &[&str], context: &str) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for (key, _) in map {
        let Expr::Symbol(key) = key else {
            return Err(Error::Eval(format!("{context} keys must be symbols")));
        };
        if key.namespace.is_some() || !allowed.contains(&key.name.as_ref()) {
            return Err(Error::Eval(format!("{context} has unknown field {key}")));
        }
        if !seen.insert(key.clone()) {
            return Err(Error::Eval(format!("{context} repeats field {key}")));
        }
    }
    Ok(())
}

pub(crate) fn symbol_or_string(expr: &Expr, context: &str) -> Result<String> {
    match expr {
        Expr::Symbol(symbol) if symbol.namespace.is_none() => Ok(symbol.name.to_string()),
        Expr::String(value) => Ok(value.clone()),
        _ => Err(Error::Eval(format!("{context} must be a symbol or string"))),
    }
}

pub(crate) fn descriptor<'a, T: 'static>(value: &'a Value, context: &str) -> Result<&'a T> {
    value
        .object()
        .downcast_ref::<T>()
        .ok_or_else(|| Error::Eval(format!("{context} has the wrong runtime type")))
}

pub(crate) fn boxed<T>(cx: &Cx, value: T) -> Result<Value>
where
    T: sim_kernel::Object + sim_kernel::ObjectCompat + 'static,
{
    cx.factory().opaque(Arc::new(value))
}

pub(crate) fn field_value<T: CitizenField>(cx: &Cx, value: &T) -> Result<Value> {
    data_value(cx, &value.encode_field())
}

pub(crate) fn symbol_value(cx: &Cx, value: Symbol) -> Result<Value> {
    cx.factory().symbol(value)
}

fn data_value(cx: &Cx, expr: &Expr) -> Result<Value> {
    match expr {
        Expr::Nil => cx.factory().nil(),
        Expr::Bool(value) => cx.factory().bool(*value),
        Expr::Number(value) => cx
            .factory()
            .number_literal(value.domain.clone(), value.canonical.clone()),
        Expr::Symbol(value) => cx.factory().symbol(value.clone()),
        Expr::String(value) => cx.factory().string(value.clone()),
        Expr::Bytes(value) => cx.factory().bytes(value.clone()),
        Expr::List(items) => cx.factory().list(
            items
                .iter()
                .map(|item| data_value(cx, item))
                .collect::<Result<Vec<_>>>()?,
        ),
        Expr::Map(entries) => cx.factory().table(
            entries
                .iter()
                .map(|(key, value)| {
                    let Expr::Symbol(key) = key else {
                        return Err(Error::Eval(
                            "runtime output map keys must be symbols".to_owned(),
                        ));
                    };
                    Ok((key.clone(), data_value(cx, value)?))
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        other => cx.factory().expr(other.clone()),
    }
}

pub(crate) fn domain_error<E: std::fmt::Debug>(context: &'static str) -> impl FnOnce(E) -> Error {
    move |error| Error::Eval(format!("{context}: {error:?}"))
}

pub(crate) fn arity<T>(function: &str, expected: usize, actual: usize) -> Result<T> {
    Err(Error::Eval(format!(
        "{function} expects {expected} arguments, found {actual}"
    )))
}
