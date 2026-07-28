//! Immutable host storage for phasor component planes.

use crate::{ReferenceSolveError, complex::Complex64};

/// A complete two-dimensional host phasor field.
///
/// Real and imaginary components are stored in separate flat `f64` buffers.
/// Both buffers use the same row-major index, `row * columns + column`.
#[derive(Clone, Debug, PartialEq)]
pub struct HostPhasorField {
    rows: usize,
    columns: usize,
    real: Vec<f64>,
    imaginary: Vec<f64>,
}

impl HostPhasorField {
    pub(crate) fn try_zeroed(rows: usize, columns: usize) -> Result<Self, ReferenceSolveError> {
        let cells = rows
            .checked_mul(columns)
            .expect("SamplingPlane has already checked rows * columns");
        let real = zeroed_component("real", cells)?;
        let imaginary = zeroed_component("imaginary", cells)?;
        Ok(Self {
            rows,
            columns,
            real,
            imaginary,
        })
    }

    pub(crate) fn set_index(&mut self, index: usize, value: Complex64) {
        let (real, imaginary) = value.components();
        self.real[index] = real;
        self.imaginary[index] = imaginary;
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the number of cells in each component plane.
    pub fn len(&self) -> usize {
        self.real.len()
    }

    /// Returns whether the component planes contain no cells.
    ///
    /// Successfully solved fields are never empty because sampling-plane
    /// construction requires non-zero dimensions.
    pub fn is_empty(&self) -> bool {
        self.real.is_empty()
    }

    /// Borrows the flat row-major real component plane.
    pub fn real(&self) -> &[f64] {
        &self.real
    }

    /// Borrows the flat row-major imaginary component plane.
    pub fn imaginary(&self) -> &[f64] {
        &self.imaginary
    }

    /// Returns one cell's Cartesian components.
    pub fn cell(&self, row: usize, column: usize) -> Option<(f64, f64)> {
        let index = row.checked_mul(self.columns)?.checked_add(column)?;
        (row < self.rows && column < self.columns)
            .then(|| Complex64::new(self.real[index], self.imaginary[index]).components())
    }

    /// Consumes the field into its shape and component planes.
    pub fn into_component_planes(self) -> (usize, usize, Vec<f64>, Vec<f64>) {
        (self.rows, self.columns, self.real, self.imaginary)
    }
}

fn zeroed_component(
    component: &'static str,
    cells: usize,
) -> Result<Vec<f64>, ReferenceSolveError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(cells)
        .map_err(|_| ReferenceSolveError::AllocationFailed { component, cells })?;
    values.resize(cells, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::HostPhasorField;

    #[test]
    fn component_planes_share_one_row_major_shape() {
        let field = HostPhasorField {
            rows: 2,
            columns: 3,
            real: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            imaginary: vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        };

        assert_eq!(field.rows(), 2);
        assert_eq!(field.columns(), 3);
        assert_eq!(field.len(), 6);
        assert!(!field.is_empty());
        assert_eq!(field.cell(1, 2), Some((5.0, 15.0)));
        assert_eq!(field.cell(2, 0), None);
        assert_eq!(field.cell(0, 3), None);
        assert_eq!(field.real(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(field.imaginary(), &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    }
}
