//! One-copy adapters between reference host fields and canonical Tensors.

use std::sync::Arc;

use sim_kernel::{
    Cx, DefaultFactory, EagerPolicy, Expr, NumberLiteral, ObjectEncode, Result, Symbol, Value,
};
use sim_lib_interference_solve::HostPhasorField;
use sim_lib_numbers_tensor::{
    Tensor, TensorLocation, TensorStorage, TypedTensorStorage, domains, tensor_value_class_symbol,
};

use crate::citizen::{RecordCitizenSpec, encode_field, invalid, next_field, read_construct_parts};

/// Tensor-backed runtime phasor field.
///
/// The two component Tensors always have shape `[rows, cols]`, the same real
/// dtype, and compatible placement. No complex or grid storage is introduced.
#[derive(Clone)]
pub struct PhasorFieldDescriptor {
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub cols: usize,
    /// Canonical real-component Tensor.
    pub real: Tensor,
    /// Canonical imaginary-component Tensor.
    pub imag: Tensor,
}

impl PhasorFieldDescriptor {
    /// Admits two compatible canonical component Tensors.
    pub fn new(rows: usize, cols: usize, real: Tensor, imag: Tensor) -> Result<Self> {
        let value = Self {
            rows,
            cols,
            real,
            imag,
        };
        value.validate()?;
        Ok(value)
    }

    /// Moves a reference host field into two typed `f64` Tensor storages.
    ///
    /// The component buffers are consumed and shared directly by their Tensor;
    /// there is no intermediate grid or per-cell runtime-value allocation.
    pub fn from_host(field: HostPhasorField) -> Result<Self> {
        let (rows, cols, real, imag) = field.into_component_planes();
        let shape = vec![rows, cols];
        let real = Tensor::from_storage(
            shape.clone(),
            domains::f64(),
            Arc::new(TypedTensorStorage::<f64>::new(real)),
        )?;
        let imag = Tensor::from_storage(
            shape,
            domains::f64(),
            Arc::new(TypedTensorStorage::<f64>::new(imag)),
        )?;
        Self::new(rows, cols, real, imag)
    }

    /// Explicitly materializes both components into one host field.
    ///
    /// Each resident Tensor receives exactly one `materialize` call. Host
    /// storage remains zero-copy until the returned `HostPhasorField` takes its
    /// owned `f64` component buffers.
    pub fn materialize_host(&self, cx: &mut Cx) -> Result<HostPhasorField> {
        self.validate_metadata()?;
        let real_storage = self.real.materialize()?;
        let imag_storage = self.imag.materialize()?;
        let real = materialized_f64(cx, real_storage, self.real.dtype(), "real")?;
        let imag = materialized_f64(cx, imag_storage, self.imag.dtype(), "imaginary")?;
        HostPhasorField::from_component_planes(self.rows, self.cols, real, imag)
            .map_err(|error| invalid("PhasorField", format!("{error:?}")))
    }

    fn validate_metadata(&self) -> Result<()> {
        let cells = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| invalid("PhasorField", "rows * cols overflowed"))?;
        if self.rows == 0 || self.cols == 0 {
            return Err(invalid("PhasorField", "rows and cols must be non-zero"));
        }
        let expected = [self.rows, self.cols];
        if self.real.shape() != expected || self.imag.shape() != expected {
            return Err(invalid(
                "PhasorField",
                format!(
                    "component shapes must both be [{}, {}], found {:?} and {:?}",
                    self.rows,
                    self.cols,
                    self.real.shape(),
                    self.imag.shape()
                ),
            ));
        }
        if self.real.len() != cells || self.imag.len() != cells {
            return Err(invalid(
                "PhasorField",
                "component cell counts do not match rows * cols",
            ));
        }
        if self.real.dtype() != self.imag.dtype() {
            return Err(invalid(
                "PhasorField",
                "real and imaginary dtypes must match",
            ));
        }
        if !matches_real_dtype(self.real.dtype()) {
            return Err(invalid(
                "PhasorField",
                format!(
                    "component dtype must be numbers/f32 or numbers/f64, found {}",
                    self.real.dtype()
                ),
            ));
        }
        if !compatible_locations(self.real.location(), self.imag.location()) {
            return Err(invalid(
                "PhasorField",
                "component locations must both be host or resident at the same site",
            ));
        }
        Ok(())
    }
}

impl core::fmt::Debug for PhasorFieldDescriptor {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PhasorFieldDescriptor")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("real_shape", &self.real.shape())
            .field("imag_shape", &self.imag.shape())
            .field("dtype", &self.real.dtype())
            .field("real_location", &self.real.location())
            .field("imag_location", &self.imag.location())
            .finish()
    }
}

impl PartialEq for PhasorFieldDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && tensor_eq(&self.real, &other.real)
            && tensor_eq(&self.imag, &other.imag)
    }
}

impl RecordCitizenSpec for PhasorFieldDescriptor {
    const FIELDS: &'static [&'static str] = &["rows", "cols", "real", "imag"];

    fn encode_fields(&self, cx: &mut Cx) -> Result<Vec<Expr>> {
        Ok(vec![
            encode_field(&self.rows),
            encode_field(&self.cols),
            tensor_expr(cx, &self.real)?,
            tensor_expr(cx, &self.imag)?,
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let rows = next_field(cx, &mut fields, "rows")?;
        let cols = next_field(cx, &mut fields, "cols")?;
        let real = decode_tensor(
            cx,
            fields
                .next()
                .ok_or_else(|| invalid("PhasorField", "missing real Tensor"))?,
            "real",
        )?;
        let imag = decode_tensor(
            cx,
            fields
                .next()
                .ok_or_else(|| invalid("PhasorField", "missing imaginary Tensor"))?,
            "imag",
        )?;
        Self::new(rows, cols, real, imag)
    }

    fn example() -> Self {
        Self::from_host(
            HostPhasorField::from_component_planes(
                2,
                2,
                vec![1.0, 0.0, -1.0, 0.5],
                vec![0.0, 1.0, 0.0, -0.5],
            )
            .expect("example host field"),
        )
        .expect("example Tensor field")
    }

    fn validate(&self) -> Result<()> {
        self.validate_metadata()?;
        if self.real.location() == TensorLocation::Host {
            validate_host_finite(&self.real, "real")?;
            validate_host_finite(&self.imag, "imaginary")?;
        }
        Ok(())
    }
}

impl_record_citizen!(PhasorFieldDescriptor, "interference/PhasorField", 4);

pub(crate) fn tensor_expr(cx: &mut Cx, tensor: &Tensor) -> Result<Expr> {
    match tensor.object_encoding(cx)? {
        sim_kernel::ObjectEncoding::Constructor { class, args } => Ok(Expr::Extension {
            tag: Symbol::qualified("citizen", "read-construct"),
            payload: Box::new(Expr::Vector(
                std::iter::once(Expr::Symbol(class)).chain(args).collect(),
            )),
        }),
        _ => Err(invalid(
            "Tensor",
            "canonical Tensor did not expose a constructor encoding",
        )),
    }
}

pub(crate) fn decode_tensor(cx: &mut Cx, value: Value, field: &'static str) -> Result<Tensor> {
    let expr = sim_citizen::value_to_expr(cx, value, field)?;
    decode_tensor_expr(cx, &expr, field)
}

pub(crate) fn decode_tensor_expr(cx: &mut Cx, expr: &Expr, field: &'static str) -> Result<Tensor> {
    let (class, args) = read_construct_parts(expr, field)?;
    if class != tensor_value_class_symbol() {
        return Err(invalid(
            field,
            format!(
                "expected nested {}, found {class}",
                tensor_value_class_symbol()
            ),
        ));
    }
    let [Expr::Symbol(version), shape, data, Expr::Symbol(dtype)] = args else {
        return Err(invalid(
            field,
            "Tensor constructor must contain version, shape, data, and dtype",
        ));
    };
    if *version != Symbol::new("v1") {
        return Err(invalid(field, "Tensor constructor version must be v1"));
    }
    let Expr::List(dimensions) = shape else {
        return Err(invalid(field, "Tensor shape must be a list"));
    };
    let shape = dimensions
        .iter()
        .map(|dimension| decode_dimension(dimension, field))
        .collect::<Result<Vec<_>>>()?;
    let Expr::List(cells) = data else {
        return Err(invalid(field, "Tensor data must be a list"));
    };
    let cells = cells
        .iter()
        .map(|cell| sim_citizen::value_from_expr(cx, cell))
        .collect::<Result<Vec<_>>>()?;
    Tensor::new_exact(shape, dtype.clone(), cells)
}

fn decode_dimension(expr: &Expr, field: &'static str) -> Result<usize> {
    let Expr::Number(NumberLiteral { domain, canonical }) = expr else {
        return Err(invalid(field, "Tensor dimensions must be integers"));
    };
    if *domain != Symbol::qualified("citizen", "int") {
        return Err(invalid(field, "Tensor dimensions must use citizen/int"));
    }
    canonical
        .parse::<usize>()
        .map_err(|_| invalid(field, "Tensor dimension is not a usize"))
}

fn compatible_locations(real: TensorLocation, imag: TensorLocation) -> bool {
    match (real, imag) {
        (TensorLocation::Host, TensorLocation::Host) => true,
        (
            TensorLocation::Resident { site: real, .. },
            TensorLocation::Resident { site: imag, .. },
        ) => real == imag,
        _ => false,
    }
}

fn matches_real_dtype(dtype: &Symbol) -> bool {
    *dtype == domains::f32() || *dtype == domains::f64()
}

fn validate_host_finite(tensor: &Tensor, component: &'static str) -> Result<()> {
    let mut cx = bare_cx();
    let storage = tensor.materialize()?;
    for index in 0..storage.len() {
        let value = scalar_to_f64(
            &mut cx,
            storage.cell(index)?,
            tensor.dtype(),
            component,
            index,
        )?;
        if !value.is_finite() {
            return Err(invalid(
                "PhasorField",
                format!("{component} cell {index} must be finite"),
            ));
        }
    }
    Ok(())
}

fn materialized_f64(
    cx: &mut Cx,
    storage: Arc<dyn TensorStorage>,
    dtype: &Symbol,
    component: &'static str,
) -> Result<Vec<f64>> {
    (0..storage.len())
        .map(|index| scalar_to_f64(cx, storage.cell(index)?, dtype, component, index))
        .collect()
}

fn scalar_to_f64(
    cx: &mut Cx,
    value: Value,
    dtype: &Symbol,
    component: &'static str,
    index: usize,
) -> Result<f64> {
    let Expr::Number(number) = value.object().as_expr(cx)? else {
        return Err(invalid(
            "PhasorField",
            format!("{component} cell {index} is not a scalar number"),
        ));
    };
    if number.domain != *dtype {
        return Err(invalid(
            "PhasorField",
            format!(
                "{component} cell {index} domain {} does not match {dtype}",
                number.domain
            ),
        ));
    }
    let value = number.canonical.parse::<f64>().map_err(|_| {
        invalid(
            "PhasorField",
            format!("{component} cell {index} is not a finite real literal"),
        )
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(
            "PhasorField",
            format!("{component} cell {index} must be finite"),
        ))
    }
}

pub(crate) fn tensor_eq(left: &Tensor, right: &Tensor) -> bool {
    if left.shape() != right.shape()
        || left.dtype() != right.dtype()
        || left.location() != right.location()
    {
        return false;
    }
    if left.location() != TensorLocation::Host {
        return Arc::ptr_eq(left.storage(), right.storage());
    }
    let mut cx = bare_cx();
    let (Ok(left), Ok(right)) = (left.cells(), right.cells()) else {
        return false;
    };
    left.len() == right.len()
        && left.iter().zip(right.iter()).all(|(left, right)| {
            let left = left.object().as_expr(&mut cx);
            let right = right.object().as_expr(&mut cx);
            matches!((left, right), (Ok(left), Ok(right)) if sim_citizen::expr_citizen_eq(&left, &right))
        })
}

fn bare_cx() -> Cx {
    Cx::new(
        Arc::new(EagerPolicy),
        Arc::new(DefaultFactory),
        sim_kernel::HandleSeed::new(0x4954_4252),
    )
}
