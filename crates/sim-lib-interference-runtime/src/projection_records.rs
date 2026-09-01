//! Projection requests, certificates, masks, and scalar Tensor records.

use std::sync::Arc;

use sim_kernel::{Cx, Result, Symbol, Value};
use sim_lib_interference_solve::{
    LossClass, Observable, ProjectionCertificate, ReductionRule, ScalarProjection, ScalarSample,
};
use sim_lib_numbers_tensor::{Tensor, TensorLocation, TypedTensorStorage, domains};

use crate::{
    PhasorFieldDescriptor, SamplingCertificateDescriptor,
    citizen::{RecordCitizenSpec, decode_record, encode_field, encode_record, invalid, next_field},
    evidence::SamplingCertificateDescriptor as SamplingDescriptor,
    tensor_bridge::{decode_tensor, tensor_eq, tensor_expr},
};

/// Shape-checked request for scalar observation and optional detector reduction.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionRequestDescriptor {
    /// Observable kind.
    pub observable: Symbol,
    /// Angular time, present only for `interference/instant`.
    pub angular_time_rad: Option<f64>,
    /// Non-negative amplitude floor used for phase masks.
    pub phase_floor: f64,
    /// Requested target rows.
    pub target_rows: usize,
    /// Requested target columns.
    pub target_columns: usize,
    /// Detector reduction rule.
    pub reduction: Symbol,
}

impl ProjectionRequestDescriptor {
    /// Builds and validates a projection request.
    pub fn new(
        observable: Observable,
        phase_floor: f64,
        target_rows: usize,
        target_columns: usize,
        reduction: ReductionRule,
    ) -> Result<Self> {
        let (observable, angular_time_rad) = encode_observable(observable);
        let value = Self {
            observable,
            angular_time_rad,
            phase_floor,
            target_rows,
            target_columns,
            reduction: reduction_symbol(reduction),
        };
        value.validate()?;
        Ok(value)
    }

    /// Returns the checked observable.
    pub fn observable(&self) -> Result<Observable> {
        decode_observable(&self.observable, self.angular_time_rad)
    }

    /// Returns the checked detector rule.
    pub fn reduction(&self) -> Result<ReductionRule> {
        decode_reduction(&self.reduction)
    }
}

impl RecordCitizenSpec for ProjectionRequestDescriptor {
    const FIELDS: &'static [&'static str] = &[
        "observable",
        "angular-time-rad",
        "phase-floor",
        "target-rows",
        "target-columns",
        "reduction",
    ];

    fn encode_fields(&self, _cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_field(&self.observable),
            encode_field(&self.angular_time_rad),
            encode_field(&self.phase_floor),
            encode_field(&self.target_rows),
            encode_field(&self.target_columns),
            encode_field(&self.reduction),
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let value = Self {
            observable: next_field(cx, &mut fields, "observable")?,
            angular_time_rad: next_field(cx, &mut fields, "angular-time-rad")?,
            phase_floor: next_field(cx, &mut fields, "phase-floor")?,
            target_rows: next_field(cx, &mut fields, "target-rows")?,
            target_columns: next_field(cx, &mut fields, "target-columns")?,
            reduction: next_field(cx, &mut fields, "reduction")?,
        };
        value.validate()?;
        Ok(value)
    }

    fn example() -> Self {
        Self::new(Observable::Amplitude, 0.0, 2, 2, ReductionRule::Detail)
            .expect("example projection request")
    }

    fn validate(&self) -> Result<()> {
        let observable = self.observable()?;
        let reduction = self.reduction()?;
        if !self.phase_floor.is_finite() || self.phase_floor < 0.0 {
            return Err(invalid(
                "ProjectionRequest",
                "phase floor must be finite and non-negative",
            ));
        }
        if self.target_rows == 0 || self.target_columns == 0 {
            return Err(invalid(
                "ProjectionRequest",
                "target dimensions must be non-zero",
            ));
        }
        compatible_reduction(observable, reduction)
    }
}

impl_record_citizen!(
    ProjectionRequestDescriptor,
    "interference/ProjectionRequest",
    6
);

/// Citizen descriptor for immutable detector and sampling provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionCertificateDescriptor {
    /// Source rows.
    pub source_rows: usize,
    /// Source columns.
    pub source_columns: usize,
    /// Target rows.
    pub target_rows: usize,
    /// Target columns.
    pub target_columns: usize,
    /// Minimum source rows represented by one target cell.
    pub footprint_min_rows: usize,
    /// Maximum source rows represented by one target cell.
    pub footprint_max_rows: usize,
    /// Minimum source columns represented by one target cell.
    pub footprint_min_columns: usize,
    /// Maximum source columns represented by one target cell.
    pub footprint_max_columns: usize,
    /// Observable kind.
    pub observable: Symbol,
    /// Angular time for an instantaneous observable.
    pub angular_time_rad: Option<f64>,
    /// Phase mask amplitude floor.
    pub phase_floor: f64,
    /// Detector reduction rule.
    pub reduction: Symbol,
    /// `interference/lossless` or `interference/detector-integration`.
    pub loss_class: Symbol,
    /// Unchanged source sampling truth.
    pub source_sampling: SamplingCertificateDescriptor,
    /// Number of masked target cells.
    pub mask_count: usize,
}

impl ProjectionCertificateDescriptor {
    /// Projects a domain certificate without losing sampling evidence.
    pub fn from_certificate(certificate: ProjectionCertificate) -> Self {
        let source = certificate.source_dimensions();
        let target = certificate.target_dimensions();
        let footprint = certificate.footprint();
        let (observable, angular_time_rad) = encode_observable(certificate.observable());
        Self {
            source_rows: source.rows(),
            source_columns: source.columns(),
            target_rows: target.rows(),
            target_columns: target.columns(),
            footprint_min_rows: footprint.min_rows(),
            footprint_max_rows: footprint.max_rows(),
            footprint_min_columns: footprint.min_columns(),
            footprint_max_columns: footprint.max_columns(),
            observable,
            angular_time_rad,
            phase_floor: certificate.phase_floor(),
            reduction: reduction_symbol(certificate.rule()),
            loss_class: loss_symbol(certificate.loss_class()),
            source_sampling: SamplingDescriptor::from_certificate(
                certificate.source_sampling_certificate(),
            ),
            mask_count: certificate.mask_count(),
        }
    }

    /// Returns the checked observable.
    pub fn observable(&self) -> Result<Observable> {
        decode_observable(&self.observable, self.angular_time_rad)
    }

    /// Returns the checked reduction rule.
    pub fn reduction(&self) -> Result<ReductionRule> {
        decode_reduction(&self.reduction)
    }
}

impl RecordCitizenSpec for ProjectionCertificateDescriptor {
    const FIELDS: &'static [&'static str] = &[
        "source-rows",
        "source-columns",
        "target-rows",
        "target-columns",
        "footprint-min-rows",
        "footprint-max-rows",
        "footprint-min-columns",
        "footprint-max-columns",
        "observable",
        "angular-time-rad",
        "phase-floor",
        "reduction",
        "loss-class",
        "source-sampling",
        "mask-count",
    ];

    fn encode_fields(&self, cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_field(&self.source_rows),
            encode_field(&self.source_columns),
            encode_field(&self.target_rows),
            encode_field(&self.target_columns),
            encode_field(&self.footprint_min_rows),
            encode_field(&self.footprint_max_rows),
            encode_field(&self.footprint_min_columns),
            encode_field(&self.footprint_max_columns),
            encode_field(&self.observable),
            encode_field(&self.angular_time_rad),
            encode_field(&self.phase_floor),
            encode_field(&self.reduction),
            encode_field(&self.loss_class),
            encode_record(cx, &self.source_sampling)?,
            encode_field(&self.mask_count),
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let source_rows = next_field(cx, &mut fields, "source-rows")?;
        let source_columns = next_field(cx, &mut fields, "source-columns")?;
        let target_rows = next_field(cx, &mut fields, "target-rows")?;
        let target_columns = next_field(cx, &mut fields, "target-columns")?;
        let footprint_min_rows = next_field(cx, &mut fields, "footprint-min-rows")?;
        let footprint_max_rows = next_field(cx, &mut fields, "footprint-max-rows")?;
        let footprint_min_columns = next_field(cx, &mut fields, "footprint-min-columns")?;
        let footprint_max_columns = next_field(cx, &mut fields, "footprint-max-columns")?;
        let observable = next_field(cx, &mut fields, "observable")?;
        let angular_time_rad = next_field(cx, &mut fields, "angular-time-rad")?;
        let phase_floor = next_field(cx, &mut fields, "phase-floor")?;
        let reduction = next_field(cx, &mut fields, "reduction")?;
        let loss_class = next_field(cx, &mut fields, "loss-class")?;
        let source_sampling = decode_record(
            cx,
            fields
                .next()
                .ok_or_else(|| invalid("ProjectionCertificate", "missing source sampling"))?,
            "source-sampling",
        )?;
        let mask_count = next_field(cx, &mut fields, "mask-count")?;
        let value = Self {
            source_rows,
            source_columns,
            target_rows,
            target_columns,
            footprint_min_rows,
            footprint_max_rows,
            footprint_min_columns,
            footprint_max_columns,
            observable,
            angular_time_rad,
            phase_floor,
            reduction,
            loss_class,
            source_sampling,
            mask_count,
        };
        value.validate()?;
        Ok(value)
    }

    fn example() -> Self {
        sample_projection().certificate
    }

    fn validate(&self) -> Result<()> {
        if self.source_rows == 0
            || self.source_columns == 0
            || self.target_rows == 0
            || self.target_columns == 0
            || self.target_rows > self.source_rows
            || self.target_columns > self.source_columns
        {
            return Err(invalid(
                "ProjectionCertificate",
                "target dimensions must be non-zero and no larger than source dimensions",
            ));
        }
        let expected_footprint = (
            self.source_rows / self.target_rows,
            self.source_rows.div_ceil(self.target_rows),
            self.source_columns / self.target_columns,
            self.source_columns.div_ceil(self.target_columns),
        );
        if (
            self.footprint_min_rows,
            self.footprint_max_rows,
            self.footprint_min_columns,
            self.footprint_max_columns,
        ) != expected_footprint
        {
            return Err(invalid(
                "ProjectionCertificate",
                "detector footprint does not match the integer axis partitions",
            ));
        }
        let observable = self.observable()?;
        let reduction = self.reduction()?;
        compatible_reduction(observable, reduction)?;
        if !self.phase_floor.is_finite() || self.phase_floor < 0.0 {
            return Err(invalid(
                "ProjectionCertificate",
                "phase floor must be finite and non-negative",
            ));
        }
        self.source_sampling.to_certificate()?;
        let expected_loss =
            if self.source_rows == self.target_rows && self.source_columns == self.target_columns {
                loss_symbol(LossClass::Lossless)
            } else {
                loss_symbol(LossClass::DetectorIntegration)
            };
        if self.loss_class != expected_loss {
            return Err(invalid(
                "ProjectionCertificate",
                "loss class does not match source and target dimensions",
            ));
        }
        let cells = self
            .target_rows
            .checked_mul(self.target_columns)
            .ok_or_else(|| invalid("ProjectionCertificate", "target cell count overflowed"))?;
        if self.mask_count > cells || (observable != Observable::Phase && self.mask_count != 0) {
            return Err(invalid(
                "ProjectionCertificate",
                "mask count is incompatible with the target or observable",
            ));
        }
        Ok(())
    }
}

impl_record_citizen!(
    ProjectionCertificateDescriptor,
    "interference/ProjectionCertificate",
    15
);

/// One-Tensor scalar projection with an explicit phase mask.
#[derive(Clone)]
pub struct ScalarProjectionDescriptor {
    /// Target rows.
    pub rows: usize,
    /// Target columns.
    pub columns: usize,
    /// Canonical scalar Tensor, with masked cells stored as zero.
    pub values: Tensor,
    /// Row-major phase mask.
    pub mask: Vec<bool>,
    /// Inseparable projection and sampling evidence.
    pub certificate: ProjectionCertificateDescriptor,
}

impl ScalarProjectionDescriptor {
    /// Projects a dependency-free scalar result into one typed `f64` Tensor.
    pub fn from_projection(projection: &ScalarProjection) -> Result<Self> {
        let rows = projection.rows();
        let columns = projection.columns();
        let mut values = Vec::with_capacity(projection.samples().len());
        let mut mask = Vec::with_capacity(projection.samples().len());
        for sample in projection.samples() {
            match sample {
                ScalarSample::Value(value) => {
                    values.push(*value);
                    mask.push(false);
                }
                ScalarSample::Masked => {
                    values.push(0.0);
                    mask.push(true);
                }
            }
        }
        let tensor = Tensor::from_storage(
            vec![rows, columns],
            domains::f64(),
            Arc::new(TypedTensorStorage::<f64>::new(values)),
        )?;
        let value = Self {
            rows,
            columns,
            values: tensor,
            mask,
            certificate: ProjectionCertificateDescriptor::from_certificate(
                projection.certificate(),
            ),
        };
        value.validate()?;
        Ok(value)
    }
}

impl core::fmt::Debug for ScalarProjectionDescriptor {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ScalarProjectionDescriptor")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .field("shape", &self.values.shape())
            .field("dtype", &self.values.dtype())
            .field("location", &self.values.location())
            .field("mask", &self.mask)
            .field("certificate", &self.certificate)
            .finish()
    }
}

impl PartialEq for ScalarProjectionDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.columns == other.columns
            && tensor_eq(&self.values, &other.values)
            && self.mask == other.mask
            && self.certificate == other.certificate
    }
}

impl RecordCitizenSpec for ScalarProjectionDescriptor {
    const FIELDS: &'static [&'static str] = &["rows", "columns", "values", "mask", "certificate"];

    fn encode_fields(&self, cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_field(&self.rows),
            encode_field(&self.columns),
            tensor_expr(cx, &self.values)?,
            encode_field(&self.mask),
            encode_record(cx, &self.certificate)?,
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let rows = next_field(cx, &mut fields, "rows")?;
        let columns = next_field(cx, &mut fields, "columns")?;
        let values = decode_tensor(
            cx,
            fields
                .next()
                .ok_or_else(|| invalid("Projection", "missing values Tensor"))?,
            "values",
        )?;
        let mask = next_field(cx, &mut fields, "mask")?;
        let certificate = decode_record(
            cx,
            fields
                .next()
                .ok_or_else(|| invalid("Projection", "missing certificate"))?,
            "certificate",
        )?;
        let value = Self {
            rows,
            columns,
            values,
            mask,
            certificate,
        };
        value.validate()?;
        Ok(value)
    }

    fn example() -> Self {
        sample_projection()
    }

    fn validate(&self) -> Result<()> {
        let cells = self
            .rows
            .checked_mul(self.columns)
            .ok_or_else(|| invalid("Projection", "rows * columns overflowed"))?;
        if self.rows == 0
            || self.columns == 0
            || self.values.shape() != [self.rows, self.columns]
            || self.values.len() != cells
        {
            return Err(invalid(
                "Projection",
                "Tensor shape must exactly equal non-zero rows and columns",
            ));
        }
        if self.values.dtype() != &domains::f64() && self.values.dtype() != &domains::f32() {
            return Err(invalid(
                "Projection",
                "scalar Tensor dtype must be numbers/f32 or numbers/f64",
            ));
        }
        if self.mask.len() != cells {
            return Err(invalid(
                "Projection",
                "mask length must equal rows * columns",
            ));
        }
        self.certificate.validate()?;
        if self.certificate.target_rows != self.rows
            || self.certificate.target_columns != self.columns
            || self.certificate.mask_count != self.mask.iter().filter(|masked| **masked).count()
        {
            return Err(invalid(
                "Projection",
                "certificate dimensions or mask count do not match the projection",
            ));
        }
        if self.values.location() == TensorLocation::Host {
            let paired = PhasorFieldDescriptor::new(
                self.rows,
                self.columns,
                self.values.clone(),
                self.values.clone(),
            )?;
            let mut cx = bare_cx();
            let host = paired.materialize_host(&mut cx)?;
            for (index, masked) in self.mask.iter().copied().enumerate() {
                if masked && host.real()[index] != 0.0 {
                    return Err(invalid(
                        "Projection",
                        "masked scalar Tensor cells must contain canonical zero",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl_record_citizen!(ScalarProjectionDescriptor, "interference/Projection", 5);

fn compatible_reduction(observable: Observable, reduction: ReductionRule) -> Result<()> {
    let compatible = matches!(
        (observable, reduction),
        (_, ReductionRule::Detail)
            | (
                Observable::Real
                    | Observable::Imaginary
                    | Observable::Phase
                    | Observable::Instant { .. },
                ReductionRule::DetectorComplexMean
            )
            | (Observable::Amplitude, ReductionRule::DetectorScalarAreaMean)
            | (
                Observable::MagnitudeSquared,
                ReductionRule::DetectorMagnitudeSquaredAreaMean
            )
    );
    compatible.then_some(()).ok_or_else(|| {
        invalid(
            "Projection",
            "observable and detector rule are incompatible",
        )
    })
}

fn encode_observable(observable: Observable) -> (Symbol, Option<f64>) {
    match observable {
        Observable::Real => (observable_symbol("real"), None),
        Observable::Imaginary => (observable_symbol("imaginary"), None),
        Observable::Amplitude => (observable_symbol("amplitude"), None),
        Observable::Phase => (observable_symbol("phase"), None),
        Observable::MagnitudeSquared => (observable_symbol("magnitude-squared"), None),
        Observable::Instant { wt } => (observable_symbol("instant"), Some(wt)),
    }
}

fn decode_observable(symbol: &Symbol, angular_time: Option<f64>) -> Result<Observable> {
    let value = if *symbol == observable_symbol("real") {
        Observable::Real
    } else if *symbol == observable_symbol("imaginary") {
        Observable::Imaginary
    } else if *symbol == observable_symbol("amplitude") {
        Observable::Amplitude
    } else if *symbol == observable_symbol("phase") {
        Observable::Phase
    } else if *symbol == observable_symbol("magnitude-squared") {
        Observable::MagnitudeSquared
    } else if *symbol == observable_symbol("instant") {
        let wt = angular_time
            .filter(|wt| wt.is_finite())
            .ok_or_else(|| invalid("Projection", "instant requires finite angular time"))?;
        return Ok(Observable::Instant { wt });
    } else {
        return Err(invalid(
            "Projection",
            format!("unknown observable {symbol}"),
        ));
    };
    if angular_time.is_some() {
        return Err(invalid(
            "Projection",
            "angular time is allowed only for instant",
        ));
    }
    Ok(value)
}

fn reduction_symbol(reduction: ReductionRule) -> Symbol {
    Symbol::qualified(
        "interference",
        match reduction {
            ReductionRule::Detail => "detail",
            ReductionRule::DetectorComplexMean => "detector-complex-mean",
            ReductionRule::DetectorScalarAreaMean => "detector-scalar-area-mean",
            ReductionRule::DetectorMagnitudeSquaredAreaMean => {
                "detector-magnitude-squared-area-mean"
            }
        },
    )
}

fn decode_reduction(symbol: &Symbol) -> Result<ReductionRule> {
    [
        ReductionRule::Detail,
        ReductionRule::DetectorComplexMean,
        ReductionRule::DetectorScalarAreaMean,
        ReductionRule::DetectorMagnitudeSquaredAreaMean,
    ]
    .into_iter()
    .find(|rule| reduction_symbol(*rule) == *symbol)
    .ok_or_else(|| invalid("Projection", format!("unknown reduction rule {symbol}")))
}

fn loss_symbol(loss: LossClass) -> Symbol {
    Symbol::qualified(
        "interference",
        match loss {
            LossClass::Lossless => "lossless",
            LossClass::DetectorIntegration => "detector-integration",
        },
    )
}

fn observable_symbol(name: &str) -> Symbol {
    Symbol::qualified("interference", name.to_owned())
}

pub(crate) fn sample_projection() -> ScalarProjectionDescriptor {
    use sim_lib_interference_core::SamplingCertificate;
    use sim_lib_interference_solve::project;

    let problem = crate::records::sample_problem()
        .to_problem()
        .expect("example problem");
    let plane = crate::records::sample_plane()
        .to_plane()
        .expect("example plane");
    let field = sim_lib_interference_solve::HostPhasorField::from_component_planes(
        2,
        2,
        vec![1.0, 0.0, -1.0, 0.5],
        vec![0.0, 1.0, 0.0, -0.5],
    )
    .expect("example field");
    let sampling = SamplingCertificate::measure(&problem, &plane).expect("example sampling");
    let projection = project(&field, sampling, Observable::Phase, 0.0).expect("example projection");
    ScalarProjectionDescriptor::from_projection(&projection).expect("example runtime projection")
}

fn bare_cx() -> Cx {
    Cx::new(
        Arc::new(sim_kernel::EagerPolicy),
        Arc::new(sim_kernel::DefaultFactory),
        sim_kernel::HandleSeed::new(0x4950_524a),
    )
}
