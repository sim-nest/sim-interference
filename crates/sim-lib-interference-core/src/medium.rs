//! The homogeneous scalar medium and its complex wavenumber.

use std::f64::consts::TAU;

use crate::{Hertz, MetresPerSecond, NepersPerMetre};

/// The complex wavenumber `k_tilde = omega / c + i * alpha`.
///
/// For the crate's `exp(i * k_tilde * r)` propagation convention, the real
/// part advances phase and the non-negative imaginary part produces
/// `exp(-alpha * r)` attenuation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveNumber {
    real_radians_per_metre: f64,
    imaginary_nepers_per_metre: f64,
}

impl WaveNumber {
    /// Returns the real, phase-advancing component in radians per metre.
    pub fn real_radians_per_metre(self) -> f64 {
        self.real_radians_per_metre
    }

    /// Returns the imaginary, attenuating component in nepers per metre.
    pub fn imaginary_nepers_per_metre(self) -> f64 {
        self.imaginary_nepers_per_metre
    }
}

/// A three-dimensional, homogeneous, isotropic scalar propagation medium.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarMedium {
    speed: MetresPerSecond,
    attenuation: NepersPerMetre,
}

impl ScalarMedium {
    /// Constructs a medium from checked speed and attenuation quantities.
    pub fn new(speed: MetresPerSecond, attenuation: NepersPerMetre) -> Self {
        Self { speed, attenuation }
    }

    /// Returns the propagation speed.
    pub fn speed(self) -> MetresPerSecond {
        self.speed
    }

    /// Returns the attenuation coefficient.
    pub fn attenuation(self) -> NepersPerMetre {
        self.attenuation
    }

    /// Derives the complex wavenumber at `frequency`.
    pub fn wavenumber(self, frequency: Hertz) -> WaveNumber {
        WaveNumber {
            real_radians_per_metre: TAU * frequency.get() / self.speed.get(),
            imaginary_nepers_per_metre: self.attenuation.get(),
        }
    }
}
