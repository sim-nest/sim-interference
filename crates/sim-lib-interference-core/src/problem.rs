//! A complete, single-frequency coherent interference problem.

use std::f64::consts::TAU;

use crate::{Hertz, PositiveMetres, ScalarMedium, SourceSet, WaveNumber};

/// The distance at which a point emitter's field amplitude is specified.
///
/// Point-source propagation therefore uses a dimensionless spreading factor
/// `POINT_SOURCE_REFERENCE_DISTANCE_METRES / r`.
pub const POINT_SOURCE_REFERENCE_DISTANCE_METRES: f64 = 1.0;

/// A coherent scalar-wave problem in one homogeneous medium.
///
/// The model uses the real field
/// `u(x, t) = Re{U(x) * exp(-i * omega * t)}` and the complex wavenumber
/// `k_tilde = omega / c + i * alpha`. Away from point sources, the phasor
/// satisfies `laplacian(U) + k_tilde^2 * U = 0`.
///
/// The positive outgoing sign is `exp(i * k_tilde * distance)`: phase advances
/// with distance while non-negative `alpha` attenuates it. Point-source
/// amplitudes are stated at [`POINT_SOURCE_REFERENCE_DISTANCE_METRES`] and
/// samples at or inside `singularity_radius` are outside the model.
#[derive(Clone, Debug, PartialEq)]
pub struct InterferenceProblem {
    /// The one frequency shared by every coherent source.
    pub frequency: Hertz,
    /// The homogeneous scalar propagation medium.
    pub medium: ScalarMedium,
    /// The non-empty, canonically ordered coherent sources.
    pub sources: SourceSet,
    /// The positive exclusion radius around every point source.
    pub singularity_radius: PositiveMetres,
}

impl InterferenceProblem {
    /// Constructs a problem from checked components.
    pub fn new(
        frequency: Hertz,
        medium: ScalarMedium,
        sources: SourceSet,
        singularity_radius: PositiveMetres,
    ) -> Self {
        Self {
            frequency,
            medium,
            sources,
            singularity_radius,
        }
    }

    /// Returns the angular frequency `omega = 2 * pi * frequency`.
    pub fn angular_frequency_radians_per_second(&self) -> f64 {
        TAU * self.frequency.get()
    }

    /// Returns this problem's complex wavenumber.
    pub fn wavenumber(&self) -> WaveNumber {
        self.medium.wavenumber(self.frequency)
    }

    /// Returns the unattenuated wavelength `c / frequency` in metres.
    pub fn wavelength_metres(&self) -> f64 {
        self.medium.speed().get() / self.frequency.get()
    }
}
