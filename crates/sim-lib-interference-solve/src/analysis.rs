//! Deterministic scalar-field statistics and fringe candidates.

use std::fmt;

use sim_lib_interference_core::SamplingCertificate;

use crate::{Observable, ProjectionIdentity, ScalarProjection, ScalarSample};

/// Aggregate statistics for every cell in one scalar field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldStats {
    /// Number of analyzed cells.
    pub count: usize,
    /// Smallest scalar sample.
    pub minimum: f64,
    /// Largest scalar sample.
    pub maximum: f64,
    /// Row-major online mean.
    pub mean: f64,
    /// Population variance computed in deterministic row-major order.
    pub population_variance: f64,
}

/// Physical interpretation of a strict local scalar extremum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtremumKind {
    /// Strict local minimum and therefore a possible destructive node.
    NodeCandidate,
    /// Strict local maximum and therefore a possible constructive antinode.
    AntinodeCandidate,
}

/// One strict local extremum in target-grid coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extremum {
    /// Zero-based target row.
    pub row: usize,
    /// Zero-based target column.
    pub column: usize,
    /// Projected scalar value at this cell.
    pub value: f64,
    /// Node or antinode interpretation.
    pub kind: ExtremumKind,
}

/// Statistics and fringe candidates with inseparable physical provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct FringeReport {
    /// Sampling evidence inherited unchanged from the source solve.
    pub sampling: SamplingCertificate,
    /// Exact projection parameters of the analyzed scalar field.
    pub projection: ProjectionIdentity,
    /// Whole-field scalar statistics.
    pub stats: FieldStats,
    /// Strict local extrema in row-major order.
    pub extrema: Vec<Extremum>,
    /// `(maximum - minimum) / (maximum + minimum)`.
    ///
    /// This is `None` when the field's maximum amplitude is at or below the
    /// caller's amplitude floor.
    pub michelson_contrast: Option<f64>,
}

/// A scalar projection could not be honestly analyzed as a fringe field.
#[derive(Clone, Debug, PartialEq)]
pub enum AnalysisError {
    /// The amplitude floor was negative or non-finite.
    InvalidAmplitudeFloor {
        /// Rejected floor.
        value: f64,
    },
    /// Fringe analysis requires a non-negative amplitude-like observable.
    IncompatibleObservable {
        /// Rejected projection observable.
        observable: Observable,
    },
    /// An amplitude-like projection unexpectedly contained a masked cell.
    MaskedSample {
        /// Zero-based target row.
        row: usize,
        /// Zero-based target column.
        column: usize,
    },
    /// A derived statistic was not finite.
    NonFiniteStatistic {
        /// Stable statistic name.
        name: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// Extrema storage could not be reserved.
    AllocationFailed {
        /// Maximum number of extrema requested.
        cells: usize,
    },
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAmplitudeFloor { value } => write!(
                formatter,
                "analysis amplitude floor must be finite and non-negative: {value:?}"
            ),
            Self::IncompatibleObservable { observable } => write!(
                formatter,
                "fringe analysis requires amplitude or magnitude-squared, not {observable:?}"
            ),
            Self::MaskedSample { row, column } => {
                write!(formatter, "analysis input is masked at ({row}, {column})")
            }
            Self::NonFiniteStatistic { name, value } => {
                write!(
                    formatter,
                    "analysis statistic `{name}` is non-finite: {value:?}"
                )
            }
            Self::AllocationFailed { cells } => {
                write!(
                    formatter,
                    "could not reserve extrema storage for {cells} cells"
                )
            }
        }
    }
}

impl std::error::Error for AnalysisError {}

/// Analyzes an amplitude or normalized squared-magnitude projection.
///
/// Extrema use the fixed eight-cell Moore neighbourhood, clipped at field
/// edges. A candidate must be strictly less than or strictly greater than
/// every available neighbour, so plateaus never produce traversal-dependent
/// representatives. Results are emitted in row-major order. Michelson
/// contrast is omitted when the maximum represented amplitude is at or below
/// `amplitude_floor`.
pub fn analyze_fringes(
    projection: &ScalarProjection,
    amplitude_floor: f64,
) -> Result<FringeReport, AnalysisError> {
    if !amplitude_floor.is_finite() || amplitude_floor < 0.0 {
        return Err(AnalysisError::InvalidAmplitudeFloor {
            value: amplitude_floor,
        });
    }
    let certificate = projection.certificate();
    let observable = certificate.observable();
    if !matches!(
        observable,
        Observable::Amplitude | Observable::MagnitudeSquared
    ) {
        return Err(AnalysisError::IncompatibleObservable { observable });
    }

    let stats = field_stats(projection)?;
    let extrema = find_extrema(projection)?;
    let maximum_amplitude = match observable {
        Observable::Amplitude => stats.maximum,
        Observable::MagnitudeSquared => stats.maximum.sqrt(),
        _ => unreachable!("observable was admitted above"),
    };
    let michelson_contrast = if maximum_amplitude <= amplitude_floor {
        None
    } else {
        let contrast = (stats.maximum - stats.minimum) / (stats.maximum + stats.minimum);
        Some(require_finite("michelson-contrast", contrast)?)
    };

    Ok(FringeReport {
        sampling: certificate.source_sampling_certificate(),
        projection: certificate.identity(),
        stats,
        extrema,
        michelson_contrast,
    })
}

fn field_stats(projection: &ScalarProjection) -> Result<FieldStats, AnalysisError> {
    let mut minimum = scalar_value(projection, 0)?;
    let mut maximum = minimum;
    let mut mean = 0.0;
    let mut squared_deviation = 0.0;
    for index in 0..projection.samples().len() {
        let value = scalar_value(projection, index)?;
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        let count = (index + 1) as f64;
        let delta = value - mean;
        mean += delta / count;
        squared_deviation += delta * (value - mean);
        require_finite("mean", mean)?;
        require_finite("squared-deviation", squared_deviation)?;
    }
    let population_variance = require_finite(
        "population-variance",
        squared_deviation / projection.samples().len() as f64,
    )?;
    Ok(FieldStats {
        count: projection.samples().len(),
        minimum,
        maximum,
        mean,
        population_variance,
    })
}

fn find_extrema(projection: &ScalarProjection) -> Result<Vec<Extremum>, AnalysisError> {
    let rows = projection.rows();
    let columns = projection.columns();
    let mut extrema = Vec::new();
    extrema
        .try_reserve(projection.samples().len())
        .map_err(|_| AnalysisError::AllocationFailed {
            cells: projection.samples().len(),
        })?;
    for row in 0..rows {
        for column in 0..columns {
            let value = scalar_value(projection, row * columns + column)?;
            let mut has_neighbour = false;
            let mut below_all = true;
            let mut above_all = true;
            for neighbour_row in row.saturating_sub(1)..=(row + 1).min(rows - 1) {
                for neighbour_column in column.saturating_sub(1)..=(column + 1).min(columns - 1) {
                    if neighbour_row == row && neighbour_column == column {
                        continue;
                    }
                    has_neighbour = true;
                    let neighbour =
                        scalar_value(projection, neighbour_row * columns + neighbour_column)?;
                    below_all &= value < neighbour;
                    above_all &= value > neighbour;
                }
            }
            let kind = match (has_neighbour && below_all, has_neighbour && above_all) {
                (true, false) => Some(ExtremumKind::NodeCandidate),
                (false, true) => Some(ExtremumKind::AntinodeCandidate),
                _ => None,
            };
            if let Some(kind) = kind {
                extrema.push(Extremum {
                    row,
                    column,
                    value,
                    kind,
                });
            }
        }
    }
    extrema.shrink_to_fit();
    Ok(extrema)
}

fn scalar_value(projection: &ScalarProjection, index: usize) -> Result<f64, AnalysisError> {
    match projection.samples()[index] {
        ScalarSample::Value(value) => Ok(value),
        ScalarSample::Masked => Err(AnalysisError::MaskedSample {
            row: index / projection.columns(),
            column: index % projection.columns(),
        }),
    }
}

fn require_finite(name: &'static str, value: f64) -> Result<f64, AnalysisError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or(AnalysisError::NonFiniteStatistic { name, value })
}

#[cfg(test)]
#[path = "analysis_tests.rs"]
mod tests;
