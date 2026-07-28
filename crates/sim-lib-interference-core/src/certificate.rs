//! Sampling adequacy measurements carried with every field request.

use crate::{Emitter, InterferenceError, InterferenceProblem, SamplingPlane};

/// Explicit thresholds used to classify a [`SamplingCertificate`].
///
/// The default resolved carrier requirement is eight samples per wavelength,
/// which gives four samples across the worst-case half-wavelength power
/// fringe. Four carrier samples (two per power fringe) is the marginal Nyquist
/// floor. Point-source `1/r` envelope change defaults to five percent for
/// resolved and ten percent for marginal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingThresholds {
    /// Comfortable carrier samples per wavelength on both axes.
    pub resolved_min_samples_per_wavelength: f64,
    /// Marginal carrier samples per wavelength on both axes.
    pub marginal_min_samples_per_wavelength: f64,
    /// Comfortable maximum fractional `1/r` change across a cell.
    pub resolved_max_envelope_fraction_per_cell: f64,
    /// Marginal maximum fractional `1/r` change across a cell.
    pub marginal_max_envelope_fraction_per_cell: f64,
}

impl SamplingThresholds {
    /// Validates explicit resolved and marginal classification thresholds.
    pub fn new(
        resolved_min_samples_per_wavelength: f64,
        marginal_min_samples_per_wavelength: f64,
        resolved_max_envelope_fraction_per_cell: f64,
        marginal_max_envelope_fraction_per_cell: f64,
    ) -> Result<Self, InterferenceError> {
        Self {
            resolved_min_samples_per_wavelength,
            marginal_min_samples_per_wavelength,
            resolved_max_envelope_fraction_per_cell,
            marginal_max_envelope_fraction_per_cell,
        }
        .validate()
    }

    /// Revalidates a threshold record, including one built with field syntax.
    pub fn validate(self) -> Result<Self, InterferenceError> {
        positive_threshold(
            "resolved-min-samples-per-wavelength",
            self.resolved_min_samples_per_wavelength,
        )?;
        positive_threshold(
            "marginal-min-samples-per-wavelength",
            self.marginal_min_samples_per_wavelength,
        )?;
        non_negative_threshold(
            "resolved-max-envelope-fraction-per-cell",
            self.resolved_max_envelope_fraction_per_cell,
        )?;
        non_negative_threshold(
            "marginal-max-envelope-fraction-per-cell",
            self.marginal_max_envelope_fraction_per_cell,
        )?;
        if self.resolved_min_samples_per_wavelength < self.marginal_min_samples_per_wavelength
            || self.resolved_max_envelope_fraction_per_cell
                > self.marginal_max_envelope_fraction_per_cell
        {
            return Err(InterferenceError::InconsistentSamplingThresholds {
                resolved_min_samples_per_wavelength: self.resolved_min_samples_per_wavelength,
                marginal_min_samples_per_wavelength: self.marginal_min_samples_per_wavelength,
                resolved_max_envelope_fraction_per_cell: self
                    .resolved_max_envelope_fraction_per_cell,
                marginal_max_envelope_fraction_per_cell: self
                    .marginal_max_envelope_fraction_per_cell,
            });
        }
        Ok(self)
    }
}

impl Default for SamplingThresholds {
    fn default() -> Self {
        Self {
            resolved_min_samples_per_wavelength: 8.0,
            marginal_min_samples_per_wavelength: 4.0,
            resolved_max_envelope_fraction_per_cell: 0.05,
            marginal_max_envelope_fraction_per_cell: 0.10,
        }
    }
}

/// Whether a non-resolved request is refused or explicitly annotated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SamplingPolicy {
    /// Require a fully resolved certificate before solving.
    #[default]
    Strict,
    /// Admit the request while preserving its non-resolved certificate.
    Annotate,
}

impl SamplingPolicy {
    /// Applies this policy without allocating field storage.
    pub fn admit(self, certificate: &SamplingCertificate) -> Result<(), InterferenceError> {
        if self == Self::Strict && certificate.verdict != SamplingVerdict::Resolved {
            Err(InterferenceError::SamplingRefused {
                certificate: *certificate,
            })
        } else {
            Ok(())
        }
    }
}

/// Physical sampling classification for both phase and point-source envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingVerdict {
    /// Meets the comfortable carrier/fringe and envelope requirements.
    Resolved,
    /// Meets the minimum requirements but not the comfortable requirements.
    Marginal,
    /// Misses carrier/fringe Nyquist or the marginal envelope requirement.
    Aliased,
}

/// Measurements explaining whether a problem is resolved by a sampling plane.
///
/// Coherent squared magnitude can contain a worst-case fringe period of
/// `wavelength / 2`, so its samples-per-period values are half the carrier
/// samples-per-wavelength values. For point sources, the envelope bound covers
/// one complete cell diagonal relative to the nearest point on the finite
/// plane. A source touching that plane is rejected because no finite `1/r`
/// envelope certificate exists.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingCertificate {
    /// Explicit thresholds used for this classification.
    pub thresholds: SamplingThresholds,
    /// Unattenuated carrier wavelength.
    pub wavelength_m: f64,
    /// Carrier samples per wavelength along the plane's `u` axis.
    pub samples_per_wavelength_u: f64,
    /// Carrier samples per wavelength along the plane's `v` axis.
    pub samples_per_wavelength_v: f64,
    /// Samples per worst-case squared-magnitude fringe along `u`.
    pub samples_per_power_fringe_u: f64,
    /// Samples per worst-case squared-magnitude fringe along `v`.
    pub samples_per_power_fringe_v: f64,
    /// Distance from a point source to the nearest point on the finite plane.
    ///
    /// This is `None` when the problem has no point source.
    pub nearest_point_source_distance_m: Option<f64>,
    /// Conservative maximum fractional `1/r` change across one cell.
    pub max_envelope_fraction_per_cell: f64,
    /// Classification obtained from every preceding measurement.
    pub verdict: SamplingVerdict,
}

impl SamplingCertificate {
    /// Measures phase, worst-case power fringes, and point-source envelope.
    ///
    /// This performs only constant storage work and iterates over sources; it
    /// does not allocate a grid.
    pub fn measure(
        problem: &InterferenceProblem,
        plane: &SamplingPlane,
    ) -> Result<Self, InterferenceError> {
        Self::measure_with_thresholds(problem, plane, SamplingThresholds::default())
    }

    /// Measures using caller-supplied, validated classification thresholds.
    pub fn measure_with_thresholds(
        problem: &InterferenceProblem,
        plane: &SamplingPlane,
        thresholds: SamplingThresholds,
    ) -> Result<Self, InterferenceError> {
        let thresholds = thresholds.validate()?;
        let wavelength_m = finite_metric("wavelength-m", problem.wavelength_metres())?;
        let samples_per_wavelength_u = finite_metric(
            "samples-per-wavelength-u",
            wavelength_m / plane.cell_size_u_m(),
        )?;
        let samples_per_wavelength_v = finite_metric(
            "samples-per-wavelength-v",
            wavelength_m / plane.cell_size_v_m(),
        )?;
        let samples_per_power_fringe_u = samples_per_wavelength_u / 2.0;
        let samples_per_power_fringe_v = samples_per_wavelength_v / 2.0;
        let nearest_point_source_distance_m = nearest_point_source_distance(problem, plane)?;
        let max_envelope_fraction_per_cell = match nearest_point_source_distance_m {
            Some(distance) => conservative_envelope_change(distance, plane)?,
            None => 0.0,
        };

        let minimum_carrier_samples = samples_per_wavelength_u.min(samples_per_wavelength_v);
        let verdict = if minimum_carrier_samples >= thresholds.resolved_min_samples_per_wavelength
            && max_envelope_fraction_per_cell <= thresholds.resolved_max_envelope_fraction_per_cell
        {
            SamplingVerdict::Resolved
        } else if minimum_carrier_samples >= thresholds.marginal_min_samples_per_wavelength
            && max_envelope_fraction_per_cell <= thresholds.marginal_max_envelope_fraction_per_cell
        {
            SamplingVerdict::Marginal
        } else {
            SamplingVerdict::Aliased
        };

        Ok(Self {
            thresholds,
            wavelength_m,
            samples_per_wavelength_u,
            samples_per_wavelength_v,
            samples_per_power_fringe_u,
            samples_per_power_fringe_v,
            nearest_point_source_distance_m,
            max_envelope_fraction_per_cell,
            verdict,
        })
    }
}

fn positive_threshold(name: &'static str, value: f64) -> Result<(), InterferenceError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(InterferenceError::InvalidSamplingThreshold { name, value })
    }
}

fn non_negative_threshold(name: &'static str, value: f64) -> Result<(), InterferenceError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(InterferenceError::InvalidSamplingThreshold { name, value })
    }
}

fn finite_metric(name: &'static str, value: f64) -> Result<f64, InterferenceError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or(InterferenceError::NonFiniteSamplingMetric { name, value })
}

fn nearest_point_source_distance(
    problem: &InterferenceProblem,
    plane: &SamplingPlane,
) -> Result<Option<f64>, InterferenceError> {
    let mut nearest: Option<f64> = None;
    for source in &problem.sources {
        let Emitter::Point { position, .. } = source else {
            continue;
        };
        let [origin_x, origin_y, origin_z] = plane.origin().coordinates_metres();
        let [source_x, source_y, source_z] = position.coordinates_metres();
        let displacement = [
            source_x - origin_x,
            source_y - origin_y,
            source_z - origin_z,
        ];
        if displacement.iter().any(|component| !component.is_finite()) {
            return Err(InterferenceError::NonFiniteSamplingMetric {
                name: "point-source-plane-displacement-m",
                value: displacement
                    .into_iter()
                    .find(|component| !component.is_finite())
                    .unwrap_or(f64::NAN),
            });
        }
        let [ux, uy, uz] = plane.u_axis().components();
        let [vx, vy, vz] = plane.v_axis().components();
        let [nx, ny, nz] = plane.normal().components();
        let projected_u = displacement[0] * ux + displacement[1] * uy + displacement[2] * uz;
        let projected_v = displacement[0] * vx + displacement[1] * vy + displacement[2] * vz;
        let projected_normal = displacement[0] * nx + displacement[1] * ny + displacement[2] * nz;
        let outside_u = outside_extent_distance(projected_u, plane.extent_u().get());
        let outside_v = outside_extent_distance(projected_v, plane.extent_v().get());
        let distance = finite_metric(
            "nearest-point-source-distance-m",
            outside_u.hypot(outside_v).hypot(projected_normal),
        )?;
        nearest = Some(nearest.map_or(distance, |current| current.min(distance)));
    }
    Ok(nearest)
}

fn outside_extent_distance(coordinate: f64, extent: f64) -> f64 {
    if coordinate < 0.0 {
        -coordinate
    } else if coordinate > extent {
        coordinate - extent
    } else {
        0.0
    }
}

fn conservative_envelope_change(
    distance: f64,
    plane: &SamplingPlane,
) -> Result<f64, InterferenceError> {
    let cell_diagonal = finite_metric(
        "sampling-cell-diagonal-m",
        plane.cell_size_u_m().hypot(plane.cell_size_v_m()),
    )?;
    let value = cell_diagonal / distance;
    if distance == 0.0 || !value.is_finite() {
        Err(InterferenceError::UnboundedSamplingEnvelope {
            nearest_point_source_distance_m: distance,
            cell_diagonal_m: cell_diagonal,
        })
    } else {
        Ok(value)
    }
}
