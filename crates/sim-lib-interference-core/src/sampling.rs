//! Checked finite planes for two-dimensional physical sampling.

use crate::{InterferenceError, Point3M, PositiveMetres, UnitVector3};

/// Largest accepted absolute dot product between sampling axes.
///
/// Axes are normalized before they enter [`SamplingPlane`]. This tolerance
/// permits ordinary floating-point construction of rotated frames while still
/// rejecting skewed grids.
pub const SAMPLING_AXIS_ORTHOGONALITY_TOLERANCE: f64 = 1.0e-12;

/// A finite rectangular sampling plane embedded in three-dimensional space.
///
/// `origin` is the corner before the first cell. Columns advance along `u` and
/// rows advance along `v`. A cell at `(row, column)` is sampled at its centre:
///
/// `origin + (column + 1/2) * extent_u / columns * u`
/// `       + (row + 1/2) * extent_v / rows * v`.
///
/// Construction validates the orthonormal frame, dimensions, derived cell
/// sizes, and `rows * columns` before any caller needs to allocate grid
/// storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingPlane {
    origin: Point3M,
    u_axis: UnitVector3,
    v_axis: UnitVector3,
    normal: UnitVector3,
    extent_u: PositiveMetres,
    extent_v: PositiveMetres,
    rows: usize,
    columns: usize,
    cell_count: usize,
    cell_size_u_m: f64,
    cell_size_v_m: f64,
}

impl SamplingPlane {
    /// Constructs a checked finite sampling plane.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        origin: Point3M,
        u_axis: UnitVector3,
        v_axis: UnitVector3,
        extent_u: PositiveMetres,
        extent_v: PositiveMetres,
        rows: usize,
        columns: usize,
    ) -> Result<Self, InterferenceError> {
        if rows == 0 {
            return Err(InterferenceError::ZeroSamplingDimension { name: "rows" });
        }
        if columns == 0 {
            return Err(InterferenceError::ZeroSamplingDimension { name: "columns" });
        }
        let cell_count = rows
            .checked_mul(columns)
            .ok_or(InterferenceError::SamplingCellCountOverflow { rows, columns })?;

        let [ux, uy, uz] = u_axis.components();
        let [vx, vy, vz] = v_axis.components();
        let dot_product = ux * vx + uy * vy + uz * vz;
        if dot_product.abs() > SAMPLING_AXIS_ORTHOGONALITY_TOLERANCE {
            return Err(InterferenceError::NonOrthogonalSamplingAxes {
                dot_product,
                max_abs_dot_product: SAMPLING_AXIS_ORTHOGONALITY_TOLERANCE,
            });
        }

        let normal = UnitVector3::new(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)?;
        let cell_size_u_m = extent_u.get() / columns as f64;
        let cell_size_v_m = extent_v.get() / rows as f64;
        require_cell_size("u", cell_size_u_m)?;
        require_cell_size("v", cell_size_v_m)?;

        Ok(Self {
            origin,
            u_axis,
            v_axis,
            normal,
            extent_u,
            extent_v,
            rows,
            columns,
            cell_count,
            cell_size_u_m,
            cell_size_v_m,
        })
    }

    /// Returns the corner before the first cell.
    pub fn origin(self) -> Point3M {
        self.origin
    }

    /// Returns the direction in which columns advance.
    pub fn u_axis(self) -> UnitVector3 {
        self.u_axis
    }

    /// Returns the direction in which rows advance.
    pub fn v_axis(self) -> UnitVector3 {
        self.v_axis
    }

    /// Returns the right-handed unit normal `u cross v`.
    pub fn normal(self) -> UnitVector3 {
        self.normal
    }

    /// Returns the physical `u` extent.
    pub fn extent_u(self) -> PositiveMetres {
        self.extent_u
    }

    /// Returns the physical `v` extent.
    pub fn extent_v(self) -> PositiveMetres {
        self.extent_v
    }

    /// Returns the number of rows along `v`.
    pub fn rows(self) -> usize {
        self.rows
    }

    /// Returns the number of columns along `u`.
    pub fn columns(self) -> usize {
        self.columns
    }

    /// Returns the checked product `rows * columns`.
    pub fn cell_count(self) -> usize {
        self.cell_count
    }

    /// Returns the centre-to-centre spacing along `u` in metres.
    pub fn cell_size_u_m(self) -> f64 {
        self.cell_size_u_m
    }

    /// Returns the centre-to-centre spacing along `v` in metres.
    pub fn cell_size_v_m(self) -> f64 {
        self.cell_size_v_m
    }

    /// Returns the exact centre defined for one sampling cell.
    pub fn point_at(self, row: usize, column: usize) -> Result<Point3M, InterferenceError> {
        if row >= self.rows || column >= self.columns {
            return Err(InterferenceError::SamplingCellOutOfBounds {
                row,
                column,
                rows: self.rows,
                columns: self.columns,
            });
        }

        let offset_u = (column as f64 + 0.5) * self.cell_size_u_m;
        let offset_v = (row as f64 + 0.5) * self.cell_size_v_m;
        let [origin_x, origin_y, origin_z] = self.origin.coordinates_metres();
        let [ux, uy, uz] = self.u_axis.components();
        let [vx, vy, vz] = self.v_axis.components();

        Point3M::from_metres(
            origin_x + offset_u * ux + offset_v * vx,
            origin_y + offset_u * uy + offset_v * vy,
            origin_z + offset_u * uz + offset_v * vz,
        )
    }
}

fn require_cell_size(axis: &'static str, value: f64) -> Result<(), InterferenceError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(InterferenceError::InvalidSamplingCellSize { axis, value })
    }
}
