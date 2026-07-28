//! Bounded physical plane tiling without large-world f32 coordinates.

use sim_lib_interference_core::{Point3M, SamplingPlane};
use sim_lib_numbers_tensor::bounded_element_count;

use crate::{LoweringError, PhaseBudget, PreflightCheck, TileProfile};

/// One rectangular row-major tile of a physical sampling plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneTile {
    row_start: usize,
    column_start: usize,
    rows: usize,
    columns: usize,
    center: Point3M,
    max_offset_metres: f64,
    segments_per_tensor: usize,
    tensor_bytes: u64,
}

impl PlaneTile {
    /// Returns the first global row in the tile.
    pub fn row_start(self) -> usize {
        self.row_start
    }

    /// Returns the first global column in the tile.
    pub fn column_start(self) -> usize {
        self.column_start
    }

    /// Returns the tile row count.
    pub fn rows(self) -> usize {
        self.rows
    }

    /// Returns the tile column count.
    pub fn columns(self) -> usize {
        self.columns
    }

    /// Returns the row-major Tensor shape.
    pub fn shape(self) -> [usize; 2] {
        [self.rows, self.columns]
    }

    /// Returns the host `f64` physical center used as the local origin.
    pub fn center(self) -> Point3M {
        self.center
    }

    /// Returns the farthest cell-center offset from the tile center.
    pub fn max_offset_metres(self) -> f64 {
        self.max_offset_metres
    }

    /// Returns the predicted segments needed by one f32 tile Tensor.
    pub fn segments_per_tensor(self) -> usize {
        self.segments_per_tensor
    }

    /// Returns the byte count of one f32 tile Tensor.
    pub fn tensor_bytes(self) -> u64 {
        self.tensor_bytes
    }
}

/// Complete deterministic tiling chosen before any Tensor submission.
#[derive(Clone, Debug, PartialEq)]
pub struct TilePlan {
    tiles: Vec<PlaneTile>,
    tile_rows: usize,
    tile_columns: usize,
    conservative_max_abs_phase_rad: f64,
}

impl TilePlan {
    /// Splits a plane under phase, element, segment, and byte limits.
    pub fn new(
        plane: SamplingPlane,
        wavenumber_rad_per_metre: f64,
        budget: PhaseBudget,
        profile: TileProfile,
    ) -> Result<Self, LoweringError> {
        budget.validate()?;
        let element_limit = profile.effective_element_limit()?;
        if !wavenumber_rad_per_metre.is_finite() || wavenumber_rad_per_metre <= 0.0 {
            return Err(LoweringError::new(
                PreflightCheck::Constant,
                "real wavenumber must be finite and positive",
            ));
        }
        if !(wavenumber_rad_per_metre as f32).is_finite() {
            return Err(LoweringError::new(
                PreflightCheck::Constant,
                "real wavenumber is not representable as finite f32",
            ));
        }
        admit_result_allocation(plane, profile)?;

        let phase_radius = budget.max_abs_residual_phase_rad / wavenumber_rad_per_metre;
        let (mut tile_rows, mut tile_columns) = phase_bounded_shape(plane, phase_radius);
        tile_columns = tile_columns.min(element_limit).max(1);
        tile_rows = tile_rows
            .min(element_limit.checked_div(tile_columns).unwrap_or(0))
            .max(1);

        let row_tiles = plane.rows().div_ceil(tile_rows);
        let column_tiles = plane.columns().div_ceil(tile_columns);
        let tile_count = row_tiles.checked_mul(column_tiles).ok_or_else(|| {
            LoweringError::new(PreflightCheck::Shape, "tile count overflowed usize")
        })?;
        let mut tiles = Vec::with_capacity(tile_count);
        let mut conservative_max_abs_phase_rad = 0.0_f64;

        for row_start in (0..plane.rows()).step_by(tile_rows) {
            let rows = tile_rows.min(plane.rows() - row_start);
            for column_start in (0..plane.columns()).step_by(tile_columns) {
                let columns = tile_columns.min(plane.columns() - column_start);
                let tile = build_tile(plane, profile, row_start, column_start, rows, columns)?;
                let phase = wavenumber_rad_per_metre * tile.max_offset_metres;
                if !phase.is_finite()
                    || phase > budget.max_abs_residual_phase_rad * (1.0 + 8.0 * f64::EPSILON)
                {
                    return Err(LoweringError::new(
                        PreflightCheck::PhaseBudget,
                        format!(
                            "tile ({row_start},{column_start}) predicts residual phase {phase} rad above {}",
                            budget.max_abs_residual_phase_rad
                        ),
                    ));
                }
                conservative_max_abs_phase_rad = conservative_max_abs_phase_rad.max(phase);
                tiles.push(tile);
            }
        }

        Ok(Self {
            tiles,
            tile_rows,
            tile_columns,
            conservative_max_abs_phase_rad,
        })
    }

    /// Returns tiles in deterministic global row-major order.
    pub fn tiles(&self) -> &[PlaneTile] {
        &self.tiles
    }

    /// Returns the nominal row count used before clipping edge tiles.
    pub fn tile_rows(&self) -> usize {
        self.tile_rows
    }

    /// Returns the nominal column count used before clipping edge tiles.
    pub fn tile_columns(&self) -> usize {
        self.tile_columns
    }

    /// Returns the conservative `k * tile_radius` bound.
    pub fn conservative_max_abs_phase_rad(&self) -> f64 {
        self.conservative_max_abs_phase_rad
    }
}

fn admit_result_allocation(
    plane: SamplingPlane,
    profile: TileProfile,
) -> Result<(), LoweringError> {
    let cells = u64::try_from(plane.cell_count()).map_err(|_| {
        LoweringError::new(
            PreflightCheck::ResultAllocation,
            "plane cell count does not fit u64",
        )
    })?;
    let bytes = cells.checked_mul(8).ok_or_else(|| {
        LoweringError::new(
            PreflightCheck::ResultAllocation,
            "two-component f32 result byte count overflowed",
        )
    })?;
    if bytes > profile.max_result_bytes {
        return Err(LoweringError::new(
            PreflightCheck::ResultAllocation,
            format!(
                "two-component f32 result requires {bytes} bytes above {}",
                profile.max_result_bytes
            ),
        ));
    }
    Ok(())
}

fn phase_bounded_shape(plane: SamplingPlane, radius: f64) -> (usize, usize) {
    let rows = plane.rows();
    let columns = plane.columns();
    if rows == 1 {
        return (
            1,
            axis_cells_within_radius(radius, plane.cell_size_u_m(), columns),
        );
    }
    if columns == 1 {
        return (
            axis_cells_within_radius(radius, plane.cell_size_v_m(), rows),
            1,
        );
    }
    let axis_radius = radius / 2.0_f64.sqrt();
    (
        axis_cells_within_radius(axis_radius, plane.cell_size_v_m(), rows),
        axis_cells_within_radius(axis_radius, plane.cell_size_u_m(), columns),
    )
}

fn axis_cells_within_radius(radius: f64, spacing: f64, extent: usize) -> usize {
    if radius >= (extent.saturating_sub(1) as f64) * spacing * 0.5 {
        return extent;
    }
    let span_cells = (2.0 * radius / spacing).floor();
    if span_cells >= usize::MAX as f64 {
        extent
    } else {
        (span_cells as usize).saturating_add(1).clamp(1, extent)
    }
}

fn build_tile(
    plane: SamplingPlane,
    profile: TileProfile,
    row_start: usize,
    column_start: usize,
    rows: usize,
    columns: usize,
) -> Result<PlaneTile, LoweringError> {
    let shape = [rows, columns];
    let elements = bounded_element_count(&shape).map_err(|error| {
        LoweringError::new(
            PreflightCheck::Shape,
            format!("tile shape {shape:?} is invalid: {error}"),
        )
    })?;
    let bytes = u64::try_from(elements)
        .ok()
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| {
            LoweringError::new(
                PreflightCheck::ResultAllocation,
                "tile Tensor byte count overflowed",
            )
        })?;
    if elements > profile.max_elements_per_tile || bytes > profile.max_tensor_bytes {
        return Err(LoweringError::new(
            PreflightCheck::ResultAllocation,
            format!("tile shape {shape:?} exceeds its element or byte limit"),
        ));
    }
    let segments_u64 = bytes.div_ceil(profile.max_segment_bytes);
    let segments = usize::try_from(segments_u64).map_err(|_| {
        LoweringError::new(
            PreflightCheck::ResultAllocation,
            "tile segment count does not fit usize",
        )
    })?;
    if segments > profile.max_segments_per_tensor {
        return Err(LoweringError::new(
            PreflightCheck::ResultAllocation,
            format!(
                "tile shape {shape:?} needs {segments} segments above {}",
                profile.max_segments_per_tensor
            ),
        ));
    }

    let center = tile_center(plane, row_start, column_start, rows, columns)?;
    let half_u = (columns.saturating_sub(1) as f64) * plane.cell_size_u_m() * 0.5;
    let half_v = (rows.saturating_sub(1) as f64) * plane.cell_size_v_m() * 0.5;
    let max_offset_metres = half_u.hypot(half_v);
    Ok(PlaneTile {
        row_start,
        column_start,
        rows,
        columns,
        center,
        max_offset_metres,
        segments_per_tensor: segments,
        tensor_bytes: bytes,
    })
}

fn tile_center(
    plane: SamplingPlane,
    row_start: usize,
    column_start: usize,
    rows: usize,
    columns: usize,
) -> Result<Point3M, LoweringError> {
    let offset_u = (column_start as f64 + columns as f64 * 0.5) * plane.cell_size_u_m();
    let offset_v = (row_start as f64 + rows as f64 * 0.5) * plane.cell_size_v_m();
    let [origin_x, origin_y, origin_z] = plane.origin().coordinates_metres();
    let [ux, uy, uz] = plane.u_axis().components();
    let [vx, vy, vz] = plane.v_axis().components();
    Point3M::from_metres(
        origin_x + offset_u * ux + offset_v * vx,
        origin_y + offset_u * uy + offset_v * vy,
        origin_z + offset_u * uz + offset_v * vz,
    )
    .map_err(|error| {
        LoweringError::new(
            PreflightCheck::Shape,
            format!("tile center is not finite: {error}"),
        )
    })
}
