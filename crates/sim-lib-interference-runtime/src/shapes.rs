//! Runtime Shapes for interference call arguments and results.

use std::{marker::PhantomData, sync::Arc};

use sim_kernel::{Cx, Expr, MatchScore, Result, Shape, ShapeDoc, ShapeMatch, Symbol, Value};
use sim_shape::shape_value;

use crate::{
    PlaneDescriptor, ProblemDescriptor, ProjectionRequestDescriptor, ScalarProjectionDescriptor,
    StudyDescriptor, citizen::RecordCitizenSpec,
};

/// Shape symbol for coherent problem records.
pub fn problem_shape_symbol() -> Symbol {
    Symbol::qualified("interference", "Problem")
}

/// Shape symbol for physical sampling-plane records.
pub fn plane_shape_symbol() -> Symbol {
    Symbol::qualified("interference", "Plane")
}

/// Shape symbol for complete Tensor-backed studies.
pub fn study_shape_symbol() -> Symbol {
    Symbol::qualified("interference", "Study")
}

/// Shape symbol for scalar projection requests.
pub fn projection_request_shape_symbol() -> Symbol {
    Symbol::qualified("interference", "ProjectionRequest")
}

/// Shape symbol for scalar projection results.
pub fn projection_shape_symbol() -> Symbol {
    Symbol::qualified("interference", "Projection")
}

/// Returns the five public Shape symbols registered by the record library.
pub fn interference_shape_symbols() -> Vec<Symbol> {
    vec![
        problem_shape_symbol(),
        plane_shape_symbol(),
        study_shape_symbol(),
        projection_request_shape_symbol(),
        projection_shape_symbol(),
    ]
}

pub(crate) fn register_interference_shapes(linker: &mut sim_kernel::Linker<'_>) -> Result<()> {
    for (symbol, shape) in shape_specs() {
        linker.shape_value(symbol.clone(), shape_value(symbol, shape))?;
    }
    Ok(())
}

fn shape_specs() -> Vec<(Symbol, Arc<dyn Shape>)> {
    vec![
        (problem_shape_symbol(), problem_shape()),
        (plane_shape_symbol(), plane_shape()),
        (study_shape_symbol(), study_shape()),
        (
            projection_request_shape_symbol(),
            projection_request_shape(),
        ),
        (projection_shape_symbol(), projection_shape()),
    ]
}

pub(crate) fn problem_shape() -> Arc<dyn Shape> {
    record_shape::<ProblemDescriptor>(
        problem_shape_symbol(),
        "Problem",
        "checked coherent single-frequency problem",
    )
}

pub(crate) fn plane_shape() -> Arc<dyn Shape> {
    record_shape::<PlaneDescriptor>(
        plane_shape_symbol(),
        "Plane",
        "checked orthonormal finite physical sampling plane",
    )
}

pub(crate) fn study_shape() -> Arc<dyn Shape> {
    record_shape::<StudyDescriptor>(
        study_shape_symbol(),
        "Study",
        "Tensor field with matching problem, plane, sampling, work, and provider evidence",
    )
}

pub(crate) fn projection_request_shape() -> Arc<dyn Shape> {
    record_shape::<ProjectionRequestDescriptor>(
        projection_request_shape_symbol(),
        "ProjectionRequest",
        "checked observable, phase floor, target dimensions, and detector rule",
    )
}

pub(crate) fn projection_shape() -> Arc<dyn Shape> {
    record_shape::<ScalarProjectionDescriptor>(
        projection_shape_symbol(),
        "Projection",
        "one-Tensor scalar projection with exact mask and certificate",
    )
}

fn record_shape<T>(symbol: Symbol, name: &'static str, detail: &'static str) -> Arc<dyn Shape>
where
    T: RecordCitizenSpec,
{
    Arc::new(RecordShape::<T> {
        symbol,
        name,
        detail,
        marker: PhantomData,
    })
}

struct RecordShape<T> {
    symbol: Symbol,
    name: &'static str,
    detail: &'static str,
    marker: PhantomData<T>,
}

impl<T> Shape for RecordShape<T>
where
    T: RecordCitizenSpec,
{
    fn symbol(&self) -> Option<Symbol> {
        Some(self.symbol.clone())
    }

    fn check_value(&self, _cx: &mut Cx, value: Value) -> Result<ShapeMatch> {
        let Some(record) = value.object().downcast_ref::<T>() else {
            return Ok(ShapeMatch::reject(format!("{} record expected", self.name)));
        };
        Ok(match record.validate() {
            Ok(()) => ShapeMatch::accept(MatchScore::exact(100)),
            Err(error) => ShapeMatch::reject(format!("malformed {}: {error}", self.name)),
        })
    }

    fn check_expr(&self, cx: &mut Cx, expr: &Expr) -> Result<ShapeMatch> {
        let value = cx.eval_expr(expr.clone())?;
        self.check_value(cx, value)
    }

    fn describe(&self, _cx: &mut Cx) -> Result<ShapeDoc> {
        Ok(ShapeDoc::new(self.name).with_detail(self.detail))
    }
}
