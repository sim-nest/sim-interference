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

/// Neumaier compensated accumulation for deterministic scalar traversals.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    pub(crate) fn total(self) -> f64 {
        self.sum + self.correction
    }

    pub(crate) fn is_finite(self) -> bool {
        self.sum.is_finite() && self.correction.is_finite() && self.total().is_finite()
    }
}
