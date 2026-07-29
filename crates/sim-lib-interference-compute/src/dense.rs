//! Checked dense host assembly for the canonical CPU Tensor lowering.

use std::sync::Arc;

use sim_kernel::{Cx, Symbol};
use sim_lib_interference_core::{InterferenceProblem, SamplingPlane};
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, Tensor, TypedTensorStorage, domains, parse_f32_literal_cell,
};

use crate::{LoweredTile, LoweringError, LoweringPlan, PhaseBudget, PreflightCheck, TileProfile};

/// Execution and phase-range evidence retained with a dense f32 field.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseExecutionEvidence {
    executor: Symbol,
    tiles: usize,
    max_segments_per_tensor: usize,
    submissions: usize,
    phase_limit_rad: f64,
    predicted_max_abs_psi_rad: f64,
    observed_max_abs_psi_rad: f64,
    predicted_geometry_error_rad: f64,
    predicted_roundoff_error_rad: f64,
}

impl DenseExecutionEvidence {
    /// Returns the canonical executor that accepted the lowering.
    pub fn executor(&self) -> &Symbol {
        &self.executor
    }

    /// Returns the number of physical plane tiles.
    pub fn tiles(&self) -> usize {
        self.tiles
    }

    /// Returns the largest admitted segment count for one tile Tensor.
    pub fn max_segments_per_tensor(&self) -> usize {
        self.max_segments_per_tensor
    }

    /// Returns the number of completed executor flushes.
    pub fn submissions(&self) -> usize {
        self.submissions
    }

    /// Returns the admitted upper bound for every residual phase argument.
    pub fn phase_limit_rad(&self) -> f64 {
        self.phase_limit_rad
    }

    /// Returns the conservative tile-radius phase bound.
    pub fn predicted_max_abs_psi_rad(&self) -> f64 {
        self.predicted_max_abs_psi_rad
    }

    /// Returns the largest residual phase computed during source preflight.
    pub fn observed_max_abs_psi_rad(&self) -> f64 {
        self.observed_max_abs_psi_rad
    }

    /// Returns the largest predicted phase error from lowered local geometry.
    pub fn predicted_geometry_error_rad(&self) -> f64 {
        self.predicted_geometry_error_rad
    }

    /// Returns the largest predicted phase error from f32 arithmetic.
    pub fn predicted_roundoff_error_rad(&self) -> f64 {
        self.predicted_roundoff_error_rad
    }
}

/// A complete finite row-major phasor field evaluated as canonical f32 Tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseF32Field {
    rows: usize,
    columns: usize,
    real: Vec<f32>,
    imaginary: Vec<f32>,
    evidence: DenseExecutionEvidence,
}

impl DenseF32Field {
    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the row-major real component plane.
    pub fn real(&self) -> &[f32] {
        &self.real
    }

    /// Returns the row-major imaginary component plane.
    pub fn imaginary(&self) -> &[f32] {
        &self.imaginary
    }

    /// Returns one cell's Cartesian components.
    pub fn cell(&self, row: usize, column: usize) -> Option<(f32, f32)> {
        let index = row.checked_mul(self.columns)?.checked_add(column)?;
        (row < self.rows && column < self.columns)
            .then(|| (self.real[index], self.imaginary[index]))
    }

    /// Returns the executor, tiling, and phase-bound evidence.
    pub fn evidence(&self) -> &DenseExecutionEvidence {
        &self.evidence
    }
}

/// Executes the normalized lowering through the canonical dense f32 CPU executor.
///
/// This path deliberately ignores any active environment executor. It is the
/// portable arithmetic baseline used to compare modeled and physical providers.
pub fn solve_dense_f32_cpu(
    cx: &mut Cx,
    problem: &InterferenceProblem,
    plane: SamplingPlane,
    phase_budget: PhaseBudget,
    tile_profile: TileProfile,
) -> Result<DenseF32Field, LoweringError> {
    let plan = LoweringPlan::preflight_with_executor(
        cx,
        problem,
        plane,
        phase_budget,
        tile_profile,
        Arc::new(CpuTensorExecutor::new()),
    )?;
    let tiles = plan.execute(cx)?;
    assemble_dense_field(plane, &plan, tiles)
}

fn assemble_dense_field(
    plane: SamplingPlane,
    plan: &LoweringPlan,
    tiles: Vec<LoweredTile>,
) -> Result<DenseF32Field, LoweringError> {
    let cells = plane.cell_count();
    let mut real = zeroed_component("real", cells)?;
    let mut imaginary = zeroed_component("imaginary", cells)?;
    let mut submissions = 0_usize;
    let mut max_segments_per_tensor = 0_usize;

    for lowered in &tiles {
        let tile = lowered.tile();
        max_segments_per_tensor = max_segments_per_tensor.max(tile.segments_per_tensor());
        submissions = submissions
            .checked_add(lowered.submissions().len())
            .ok_or_else(|| {
                LoweringError::new(
                    PreflightCheck::Execution,
                    "dense submission count overflowed usize",
                )
            })?;
        place_component(
            "real",
            plane,
            tile.row_start(),
            tile.column_start(),
            tile.rows(),
            tile.columns(),
            tensor_f32_cells(lowered.real())?.as_ref(),
            &mut real,
        )?;
        place_component(
            "imaginary",
            plane,
            tile.row_start(),
            tile.column_start(),
            tile.rows(),
            tile.columns(),
            tensor_f32_cells(lowered.imaginary())?.as_ref(),
            &mut imaginary,
        )?;
    }

    let estimate = plan.max_phase_estimate();
    let phase_budget = plan.phase_budget();
    let evidence = DenseExecutionEvidence {
        executor: plan.executor_card().symbol.clone(),
        tiles: tiles.len(),
        max_segments_per_tensor,
        submissions,
        phase_limit_rad: phase_budget.max_abs_residual_phase_rad,
        predicted_max_abs_psi_rad: plan.tile_plan().conservative_max_abs_phase_rad(),
        observed_max_abs_psi_rad: estimate.max_abs_residual_phase_rad,
        predicted_geometry_error_rad: estimate.max_predicted_geometry_error_rad,
        predicted_roundoff_error_rad: estimate.max_predicted_roundoff_error_rad,
    };
    Ok(DenseF32Field {
        rows: plane.rows(),
        columns: plane.columns(),
        real,
        imaginary,
        evidence,
    })
}

fn zeroed_component(name: &'static str, cells: usize) -> Result<Vec<f32>, LoweringError> {
    let mut values = Vec::new();
    values.try_reserve_exact(cells).map_err(|_| {
        LoweringError::new(
            PreflightCheck::ResultAllocation,
            format!("cannot reserve {cells} dense {name} cells"),
        )
    })?;
    values.resize(cells, 0.0);
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn place_component(
    name: &'static str,
    plane: SamplingPlane,
    row_start: usize,
    column_start: usize,
    rows: usize,
    columns: usize,
    source: &[f32],
    destination: &mut [f32],
) -> Result<(), LoweringError> {
    let expected = rows.checked_mul(columns).ok_or_else(|| {
        LoweringError::new(
            PreflightCheck::Shape,
            format!("{name} tile shape overflowed usize"),
        )
    })?;
    if source.len() != expected {
        return Err(LoweringError::new(
            PreflightCheck::Execution,
            format!(
                "{name} tile has {} cells for shape [{rows}, {columns}]",
                source.len()
            ),
        ));
    }
    for (local_row, row_cells) in source.chunks_exact(columns).enumerate() {
        let global_row = row_start + local_row;
        let start = global_row
            .checked_mul(plane.columns())
            .and_then(|offset| offset.checked_add(column_start))
            .ok_or_else(|| {
                LoweringError::new(
                    PreflightCheck::Shape,
                    format!("{name} tile destination offset overflowed usize"),
                )
            })?;
        let end = start.checked_add(columns).ok_or_else(|| {
            LoweringError::new(
                PreflightCheck::Shape,
                format!("{name} tile destination end overflowed usize"),
            )
        })?;
        let destination_row = destination.get_mut(start..end).ok_or_else(|| {
            LoweringError::new(
                PreflightCheck::Shape,
                format!("{name} tile lies outside the admitted plane"),
            )
        })?;
        destination_row.copy_from_slice(row_cells);
    }
    Ok(())
}

fn tensor_f32_cells(tensor: &Tensor) -> Result<Arc<[f32]>, LoweringError> {
    if tensor.dtype() != &domains::f32() {
        return Err(LoweringError::new(
            PreflightCheck::Execution,
            format!(
                "dense component dtype is {}, expected numbers/f32",
                tensor.dtype()
            ),
        ));
    }
    let storage = tensor.materialize().map_err(|error| {
        LoweringError::new(
            PreflightCheck::Execution,
            format!("cannot materialize dense component: {error}"),
        )
    })?;
    let cells = if let Some(typed) = storage.as_any().downcast_ref::<TypedTensorStorage<f32>>() {
        typed.cells()
    } else {
        (0..storage.len())
            .map(|index| {
                let value = storage.cell(index).map_err(|error| {
                    LoweringError::new(
                        PreflightCheck::Execution,
                        format!("cannot observe dense component cell {index}: {error}"),
                    )
                })?;
                parse_f32_literal_cell(&value).ok_or_else(|| {
                    LoweringError::new(
                        PreflightCheck::Execution,
                        format!("dense component cell {index} is not a canonical f32"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into()
    };
    if let Some((index, value)) = cells
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(LoweringError::new(
            PreflightCheck::Execution,
            format!("dense component cell {index} is non-finite: {value}"),
        ));
    }
    Ok(cells)
}
