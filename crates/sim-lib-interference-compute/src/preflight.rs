//! Public budgets, provider limits, and fail-closed diagnostics.

use std::{f64::consts::PI, fmt};

use sim_kernel::Symbol;
use sim_lib_numbers_tensor::{
    TensorExecutorCard, add_op_symbol, cos_op_symbol, div_op_symbol, exp_op_symbol, mul_op_symbol,
    sin_op_symbol, sqrt_op_symbol, sub_op_symbol,
};

/// Stable names of the only open Tensor operations emitted by this lowering.
pub const REQUIRED_TENSOR_OPERATION_NAMES: [&str; 8] = [
    "tensor/op/add",
    "tensor/op/sub",
    "tensor/op/mul",
    "tensor/op/div",
    "tensor/op/sqrt",
    "tensor/op/exp",
    "tensor/op/sin",
    "tensor/op/cos",
];

/// The validation boundary that refused a lowering request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreflightCheck {
    /// A budget or provider limit was internally invalid.
    Configuration,
    /// The selected executor does not advertise a required operation.
    Operator,
    /// The selected profile does not admit canonical `numbers/f32`.
    DType,
    /// A Tensor shape, element count, or tile layout was invalid.
    Shape,
    /// A point sample lies at or inside its exclusion radius.
    Singularity,
    /// A plane-wave sample lies behind its forward plane.
    ForwardPlane,
    /// A normalized-distance or gain denominator was not safely positive.
    Denominator,
    /// A Tensor or final result would exceed an allocation limit.
    ResultAllocation,
    /// Residual phase would exceed its declared interval.
    PhaseBudget,
    /// Predicted geometry error would exceed its declared limit.
    GeometryBudget,
    /// Predicted arithmetic roundoff would exceed its declared limit.
    RoundoffBudget,
    /// A host-derived constant could not be represented safely as `f32`.
    Constant,
    /// An accepted canonical Tensor operation failed.
    Execution,
}

/// A deterministic lowering diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoweringError {
    check: PreflightCheck,
    detail: String,
}

impl LoweringError {
    pub(crate) fn new(check: PreflightCheck, detail: impl Into<String>) -> Self {
        Self {
            check,
            detail: detail.into(),
        }
    }

    /// Returns the boundary that rejected or failed the request.
    pub fn check(&self) -> PreflightCheck {
        self.check
    }

    /// Returns the stable human-readable diagnostic detail.
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?} preflight: {}", self.check, self.detail)
    }
}

impl std::error::Error for LoweringError {}

/// Accuracy limits applied to every source on every tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseBudget {
    /// Largest absolute phase argument an executor may receive.
    pub max_abs_residual_phase_rad: f64,
    /// Largest predicted phase error from `f64` geometry lowered to local f32.
    pub max_predicted_geometry_error_rad: f64,
    /// Largest predicted phase error from the f32 arithmetic sequence.
    pub max_predicted_roundoff_error_rad: f64,
}

impl PhaseBudget {
    /// Constructs and validates a phase budget.
    pub fn new(
        max_abs_residual_phase_rad: f64,
        max_predicted_geometry_error_rad: f64,
        max_predicted_roundoff_error_rad: f64,
    ) -> Result<Self, LoweringError> {
        let budget = Self {
            max_abs_residual_phase_rad,
            max_predicted_geometry_error_rad,
            max_predicted_roundoff_error_rad,
        };
        budget.validate()?;
        Ok(budget)
    }

    pub(crate) fn validate(self) -> Result<(), LoweringError> {
        require_finite_positive(
            "max_abs_residual_phase_rad",
            self.max_abs_residual_phase_rad,
        )?;
        if self.max_abs_residual_phase_rad > PI {
            return Err(LoweringError::new(
                PreflightCheck::Configuration,
                format!(
                    "max_abs_residual_phase_rad {} exceeds pi",
                    self.max_abs_residual_phase_rad
                ),
            ));
        }
        require_finite_positive(
            "max_predicted_geometry_error_rad",
            self.max_predicted_geometry_error_rad,
        )?;
        require_finite_positive(
            "max_predicted_roundoff_error_rad",
            self.max_predicted_roundoff_error_rad,
        )
    }
}

impl Default for PhaseBudget {
    fn default() -> Self {
        Self {
            max_abs_residual_phase_rad: PI,
            max_predicted_geometry_error_rad: 1.0e-4,
            max_predicted_roundoff_error_rad: 1.0e-4,
        }
    }
}

/// Provider and allocation limits used to partition and admit a plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileProfile {
    /// Whether the selected provider admits canonical `numbers/f32`.
    pub supports_f32: bool,
    /// Largest logical element count for one tile Tensor.
    pub max_elements_per_tile: usize,
    /// Largest byte count in one resident segment.
    pub max_segment_bytes: u64,
    /// Largest segment count for one tile Tensor.
    pub max_segments_per_tensor: usize,
    /// Largest total byte count for one tile Tensor.
    pub max_tensor_bytes: u64,
    /// Largest combined byte count for the final real and imaginary planes.
    pub max_result_bytes: u64,
}

impl TileProfile {
    pub(crate) fn validate(self) -> Result<(), LoweringError> {
        if !self.supports_f32 {
            return Err(LoweringError::new(
                PreflightCheck::DType,
                "selected Tensor profile does not support numbers/f32",
            ));
        }
        for (name, value) in [
            ("max_elements_per_tile", self.max_elements_per_tile),
            ("max_segments_per_tensor", self.max_segments_per_tensor),
        ] {
            if value == 0 {
                return Err(LoweringError::new(
                    PreflightCheck::Configuration,
                    format!("{name} must be non-zero"),
                ));
            }
        }
        for (name, value, minimum) in [
            ("max_segment_bytes", self.max_segment_bytes, 4_u64),
            ("max_tensor_bytes", self.max_tensor_bytes, 4),
            ("max_result_bytes", self.max_result_bytes, 8),
        ] {
            if value < minimum {
                return Err(LoweringError::new(
                    PreflightCheck::Configuration,
                    format!("{name} must be at least {minimum}"),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn effective_element_limit(self) -> Result<usize, LoweringError> {
        self.validate()?;
        let segment_elements = self.max_segment_bytes / 4;
        let segmented = segment_elements
            .checked_mul(u64::try_from(self.max_segments_per_tensor).map_err(|_| {
                LoweringError::new(
                    PreflightCheck::ResultAllocation,
                    "segment count does not fit u64",
                )
            })?)
            .ok_or_else(|| {
                LoweringError::new(
                    PreflightCheck::ResultAllocation,
                    "segmented element limit overflowed",
                )
            })?;
        let tensor_elements = self.max_tensor_bytes / 4;
        let limit = u64::try_from(self.max_elements_per_tile)
            .unwrap_or(u64::MAX)
            .min(segmented)
            .min(tensor_elements);
        usize::try_from(limit).map_err(|_| {
            LoweringError::new(
                PreflightCheck::ResultAllocation,
                "effective tile element limit does not fit usize",
            )
        })
    }
}

impl Default for TileProfile {
    fn default() -> Self {
        Self {
            supports_f32: true,
            max_elements_per_tile: 262_144,
            max_segment_bytes: 64 * 1024,
            max_segments_per_tensor: 16,
            max_tensor_bytes: 16 * 1024 * 1024,
            max_result_bytes: 256 * 1024 * 1024,
        }
    }
}

pub(crate) fn required_operation_symbols() -> [Symbol; 8] {
    [
        add_op_symbol(),
        sub_op_symbol(),
        mul_op_symbol(),
        div_op_symbol(),
        sqrt_op_symbol(),
        exp_op_symbol(),
        sin_op_symbol(),
        cos_op_symbol(),
    ]
}

pub(crate) fn validate_executor(card: &TensorExecutorCard) -> Result<(), LoweringError> {
    for required in required_operation_symbols() {
        if !card
            .operations
            .iter()
            .any(|available| available == &required)
        {
            return Err(LoweringError::new(
                PreflightCheck::Operator,
                format!("executor {} does not advertise {required}", card.symbol),
            ));
        }
    }
    Ok(())
}

pub(crate) fn finite_nonzero_f32(name: &str, value: f64) -> Result<f32, LoweringError> {
    let lowered = finite_f32(name, value)?;
    if lowered == 0.0 {
        Err(LoweringError::new(
            PreflightCheck::Constant,
            format!("{name} underflowed to zero in f32"),
        ))
    } else {
        Ok(lowered)
    }
}

pub(crate) fn finite_f32(name: &str, value: f64) -> Result<f32, LoweringError> {
    let lowered = value as f32;
    if value.is_finite() && lowered.is_finite() {
        Ok(lowered)
    } else {
        Err(LoweringError::new(
            PreflightCheck::Constant,
            format!("{name} {value} is not representable as finite f32"),
        ))
    }
}

fn require_finite_positive(name: &str, value: f64) -> Result<(), LoweringError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(LoweringError::new(
            PreflightCheck::Configuration,
            format!("{name} must be finite and positive"),
        ))
    }
}
