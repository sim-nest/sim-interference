//! Crate-private Cartesian complex arithmetic.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Complex64 {
    real: f64,
    imaginary: f64,
}

impl Complex64 {
    pub(crate) fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }

    pub(crate) fn components(self) -> (f64, f64) {
        (self.real, self.imaginary)
    }
}
