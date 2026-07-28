//! Fixed-tolerance differential reporting against the deterministic f64 oracle.

use std::{f64::consts::PI, fmt};

use sim_lib_interference_solve::HostPhasorField;

use crate::DenseF32Field;

/// A quantity checked by the portable f32 differential contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConformanceMetric {
    /// Real Cartesian component.
    Real,
    /// Imaginary Cartesian component.
    Imaginary,
    /// Complex amplitude.
    Amplitude,
    /// Wrapped phase in radians, for cells above the amplitude floor.
    Phase,
    /// Squared complex magnitude.
    MagnitudeSquared,
}

/// Absolute and relative tolerances for one scalar quantity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarTolerance {
    /// Fixed absolute error allowance.
    pub absolute: f64,
    /// Fixed relative error allowance.
    pub relative: f64,
}

/// Published fixed tolerances for dense f32 interference conformance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DifferentialTolerances {
    /// Real and imaginary component tolerance.
    pub component: ScalarTolerance,
    /// Complex amplitude tolerance.
    pub amplitude: ScalarTolerance,
    /// Wrapped phase tolerance in radians.
    pub phase: ScalarTolerance,
    /// Squared-magnitude tolerance.
    pub magnitude_squared: ScalarTolerance,
    /// Reference amplitude below which phase is undefined and not compared.
    pub phase_amplitude_floor: f64,
}

impl Default for DifferentialTolerances {
    fn default() -> Self {
        Self {
            component: ScalarTolerance {
                absolute: 2.0e-5,
                relative: 2.0e-4,
            },
            amplitude: ScalarTolerance {
                absolute: 2.0e-5,
                relative: 2.0e-4,
            },
            phase: ScalarTolerance {
                absolute: 3.0e-4,
                relative: 1.0e-4,
            },
            magnitude_squared: ScalarTolerance {
                absolute: 4.0e-5,
                relative: 4.0e-4,
            },
            phase_amplitude_floor: 1.0e-5,
        }
    }
}

/// Maximum observed error and its fixed tolerance at one cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DifferentialMaximum {
    /// Absolute error.
    pub error: f64,
    /// Absolute-plus-relative limit at this cell.
    pub limit: f64,
    /// Row containing the maximum.
    pub row: usize,
    /// Column containing the maximum.
    pub column: usize,
}

impl DifferentialMaximum {
    /// Returns whether the maximum satisfies its fixed tolerance.
    pub fn passed(self) -> bool {
        self.error <= self.limit
    }

    fn ratio(self) -> f64 {
        if self.limit == 0.0 {
            if self.error == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            self.error / self.limit
        }
    }
}

/// Complete comparison report shared by portable and accelerated providers.
#[derive(Clone, Debug, PartialEq)]
pub struct DifferentialReport {
    /// Maximum real-component error.
    pub real: DifferentialMaximum,
    /// Maximum imaginary-component error.
    pub imaginary: DifferentialMaximum,
    /// Maximum amplitude error.
    pub amplitude: DifferentialMaximum,
    /// Maximum wrapped-phase error, absent when every cell is below the floor.
    pub phase: Option<DifferentialMaximum>,
    /// Maximum squared-magnitude error.
    pub magnitude_squared: DifferentialMaximum,
    /// Number of cells whose phase was compared.
    pub phase_cells: usize,
    /// Metric with the largest error-to-limit ratio.
    pub worst_metric: ConformanceMetric,
    /// Row containing the worst normalized error.
    pub worst_row: usize,
    /// Column containing the worst normalized error.
    pub worst_column: usize,
}

impl DifferentialReport {
    /// Returns true when every compared quantity is within tolerance.
    pub fn passed(&self) -> bool {
        self.real.passed()
            && self.imaginary.passed()
            && self.amplitude.passed()
            && self.phase.is_none_or(DifferentialMaximum::passed)
            && self.magnitude_squared.passed()
    }

    /// Returns the largest absolute Cartesian-component error.
    pub fn max_component_absolute_error(&self) -> f64 {
        self.real.error.max(self.imaginary.error)
    }

    /// Returns the largest compared absolute wrapped-phase error.
    ///
    /// A field containing only below-floor cancellation cells returns zero
    /// because phase is deliberately undefined there.
    pub fn max_phase_absolute_error(&self) -> f64 {
        self.phase.map_or(0.0, |maximum| maximum.error)
    }
}

/// Structural or non-finite input rejected before differential reporting.
#[derive(Clone, Debug, PartialEq)]
pub struct DifferentialError {
    detail: String,
}

impl DifferentialError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }

    /// Returns the stable diagnostic.
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for DifferentialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for DifferentialError {}

/// Compares one dense f32 result with the deterministic f64 reference field.
pub fn compare_dense_to_reference(
    reference: &HostPhasorField,
    candidate: &DenseF32Field,
    tolerances: DifferentialTolerances,
) -> Result<DifferentialReport, DifferentialError> {
    compare_candidate(
        reference,
        CandidateField {
            name: "dense",
            rows: candidate.rows(),
            columns: candidate.columns(),
            real: CandidateComponent::F32(candidate.real()),
            imaginary: CandidateComponent::F32(candidate.imaginary()),
        },
        tolerances,
    )
}

/// Compares one materialized provider result with the deterministic f64 oracle.
///
/// Provider results use the same component, amplitude, phase-floor, wrapped
/// phase, and squared-magnitude reporting path as [`compare_dense_to_reference`].
/// The candidate is expected to be a materialized f32 Tensor field represented
/// by the runtime's host phasor container.
pub fn compare_materialized_to_reference(
    reference: &HostPhasorField,
    candidate: &HostPhasorField,
    tolerances: DifferentialTolerances,
) -> Result<DifferentialReport, DifferentialError> {
    compare_candidate(
        reference,
        CandidateField {
            name: "materialized",
            rows: candidate.rows(),
            columns: candidate.columns(),
            real: CandidateComponent::F64(candidate.real()),
            imaginary: CandidateComponent::F64(candidate.imaginary()),
        },
        tolerances,
    )
}

fn compare_candidate(
    reference: &HostPhasorField,
    candidate: CandidateField<'_>,
    tolerances: DifferentialTolerances,
) -> Result<DifferentialReport, DifferentialError> {
    validate_inputs(reference, candidate, tolerances)?;
    let mut maxima = Maxima::default();
    let columns = reference.columns();
    for index in 0..reference.len() {
        let row = index / columns;
        let column = index % columns;
        let rr = reference.real()[index];
        let ri = reference.imaginary()[index];
        let cr = candidate.real.get(index);
        let ci = candidate.imaginary.get(index);
        let reference_amplitude = rr.hypot(ri);
        let candidate_amplitude = cr.hypot(ci);
        maxima.observe(
            ConformanceMetric::Real,
            (rr - cr).abs(),
            limit(tolerances.component, rr, cr),
            row,
            column,
        );
        maxima.observe(
            ConformanceMetric::Imaginary,
            (ri - ci).abs(),
            limit(tolerances.component, ri, ci),
            row,
            column,
        );
        maxima.observe(
            ConformanceMetric::Amplitude,
            (reference_amplitude - candidate_amplitude).abs(),
            limit(
                tolerances.amplitude,
                reference_amplitude,
                candidate_amplitude,
            ),
            row,
            column,
        );
        if reference_amplitude >= tolerances.phase_amplitude_floor {
            maxima.phase_cells += 1;
            let reference_phase = ri.atan2(rr);
            let candidate_phase = ci.atan2(cr);
            maxima.observe(
                ConformanceMetric::Phase,
                wrapped_phase_error(reference_phase, candidate_phase),
                limit(tolerances.phase, reference_phase, candidate_phase),
                row,
                column,
            );
        }
        let reference_squared = rr.mul_add(rr, ri * ri);
        let candidate_squared = cr.mul_add(cr, ci * ci);
        maxima.observe(
            ConformanceMetric::MagnitudeSquared,
            (reference_squared - candidate_squared).abs(),
            limit(
                tolerances.magnitude_squared,
                reference_squared,
                candidate_squared,
            ),
            row,
            column,
        );
    }
    Ok(maxima.finish())
}

fn validate_inputs(
    reference: &HostPhasorField,
    candidate: CandidateField<'_>,
    tolerances: DifferentialTolerances,
) -> Result<(), DifferentialError> {
    if reference.rows() != candidate.rows || reference.columns() != candidate.columns {
        return Err(DifferentialError::new(format!(
            "reference shape [{}, {}] differs from {} shape [{}, {}]",
            reference.rows(),
            reference.columns(),
            candidate.name,
            candidate.rows,
            candidate.columns
        )));
    }
    if candidate.real.len() != reference.len() || candidate.imaginary.len() != reference.len() {
        return Err(DifferentialError::new(format!(
            "{} component lengths [{}, {}] differ from reference length {}",
            candidate.name,
            candidate.real.len(),
            candidate.imaginary.len(),
            reference.len()
        )));
    }
    for (name, tolerance) in [
        ("component", tolerances.component),
        ("amplitude", tolerances.amplitude),
        ("phase", tolerances.phase),
        ("magnitude-squared", tolerances.magnitude_squared),
    ] {
        if !tolerance.absolute.is_finite()
            || tolerance.absolute < 0.0
            || !tolerance.relative.is_finite()
            || tolerance.relative < 0.0
        {
            return Err(DifferentialError::new(format!(
                "{name} tolerances must be finite and non-negative"
            )));
        }
    }
    if !tolerances.phase_amplitude_floor.is_finite() || tolerances.phase_amplitude_floor <= 0.0 {
        return Err(DifferentialError::new(
            "phase amplitude floor must be finite and positive",
        ));
    }
    for (name, values) in [
        ("reference real", reference.real()),
        ("reference imaginary", reference.imaginary()),
    ] {
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(DifferentialError::new(format!(
                "{name} cell {index} is non-finite: {value}"
            )));
        }
    }
    for (component, values) in [("real", candidate.real), ("imaginary", candidate.imaginary)] {
        if let Some((index, value)) = values.first_non_finite() {
            return Err(DifferentialError::new(format!(
                "{} {component} cell {index} is non-finite: {value}",
                candidate.name
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum CandidateComponent<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl CandidateComponent<'_> {
    fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    fn get(self, index: usize) -> f64 {
        match self {
            Self::F32(values) => f64::from(values[index]),
            Self::F64(values) => values[index],
        }
    }

    fn first_non_finite(self) -> Option<(usize, f64)> {
        (0..self.len())
            .map(|index| (index, self.get(index)))
            .find(|(_, value)| !value.is_finite())
    }
}

#[derive(Clone, Copy)]
struct CandidateField<'a> {
    name: &'static str,
    rows: usize,
    columns: usize,
    real: CandidateComponent<'a>,
    imaginary: CandidateComponent<'a>,
}

fn limit(tolerance: ScalarTolerance, reference: f64, candidate: f64) -> f64 {
    tolerance.absolute + tolerance.relative * reference.abs().max(candidate.abs())
}

fn wrapped_phase_error(left: f64, right: f64) -> f64 {
    let difference = (left - right).abs().rem_euclid(2.0 * PI);
    difference.min(2.0 * PI - difference)
}

#[derive(Clone, Copy, Debug)]
struct ObservedMaximum {
    value: DifferentialMaximum,
    initialized: bool,
}

impl Default for ObservedMaximum {
    fn default() -> Self {
        Self {
            value: DifferentialMaximum {
                error: 0.0,
                limit: 0.0,
                row: 0,
                column: 0,
            },
            initialized: false,
        }
    }
}

impl ObservedMaximum {
    fn observe(&mut self, error: f64, limit: f64, row: usize, column: usize) {
        let candidate = DifferentialMaximum {
            error,
            limit,
            row,
            column,
        };
        if !self.initialized
            || candidate.error > self.value.error
            || (candidate.error == self.value.error && candidate.ratio() > self.value.ratio())
        {
            self.value = candidate;
            self.initialized = true;
        }
    }
}

#[derive(Default)]
struct Maxima {
    real: ObservedMaximum,
    imaginary: ObservedMaximum,
    amplitude: ObservedMaximum,
    phase: ObservedMaximum,
    magnitude_squared: ObservedMaximum,
    phase_cells: usize,
    worst: Option<(ConformanceMetric, DifferentialMaximum)>,
}

impl Maxima {
    fn observe(
        &mut self,
        metric: ConformanceMetric,
        error: f64,
        limit: f64,
        row: usize,
        column: usize,
    ) {
        let candidate = DifferentialMaximum {
            error,
            limit,
            row,
            column,
        };
        match metric {
            ConformanceMetric::Real => self.real.observe(error, limit, row, column),
            ConformanceMetric::Imaginary => self.imaginary.observe(error, limit, row, column),
            ConformanceMetric::Amplitude => self.amplitude.observe(error, limit, row, column),
            ConformanceMetric::Phase => self.phase.observe(error, limit, row, column),
            ConformanceMetric::MagnitudeSquared => {
                self.magnitude_squared.observe(error, limit, row, column);
            }
        }
        if self
            .worst
            .is_none_or(|(_, current)| candidate.ratio() > current.ratio())
        {
            self.worst = Some((metric, candidate));
        }
    }

    fn finish(self) -> DifferentialReport {
        let phase = (self.phase_cells != 0).then_some(self.phase.value);
        let (worst_metric, worst) = self
            .worst
            .expect("a valid interference field always has at least one cell");
        DifferentialReport {
            real: self.real.value,
            imaginary: self.imaginary.value,
            amplitude: self.amplitude.value,
            phase,
            magnitude_squared: self.magnitude_squared.value,
            phase_cells: self.phase_cells,
            worst_metric,
            worst_row: worst.row,
            worst_column: worst.column,
        }
    }
}
