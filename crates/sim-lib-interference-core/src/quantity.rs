//! Finite scalar wrappers with explicit physical admission rules.

use std::f64::consts::{PI, TAU};

use crate::InterferenceError;

fn invalid(name: &'static str, value: f64) -> InterferenceError {
    InterferenceError::InvalidQuantity { name, value }
}

fn finite(name: &'static str, value: f64) -> Result<f64, InterferenceError> {
    value
        .is_finite()
        .then_some(canonical_zero(value))
        .ok_or_else(|| invalid(name, value))
}

fn positive(name: &'static str, value: f64) -> Result<f64, InterferenceError> {
    (value.is_finite() && value > 0.0)
        .then_some(value)
        .ok_or_else(|| invalid(name, value))
}

fn non_negative(name: &'static str, value: f64) -> Result<f64, InterferenceError> {
    (value.is_finite() && value >= 0.0)
        .then_some(canonical_zero(value))
        .ok_or_else(|| invalid(name, value))
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// A finite signed distance or coordinate in metres.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Metres(f64);

impl Metres {
    /// Stable diagnostic name.
    pub const NAME: &'static str = "distance-m";

    /// Admits any finite signed coordinate, including zero.
    pub fn new(value: f64) -> Result<Self, InterferenceError> {
        finite(Self::NAME, value).map(Self)
    }

    /// Returns the validated value in metres.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A finite distance in metres that is strictly greater than zero.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct PositiveMetres(f64);

impl PositiveMetres {
    /// Stable diagnostic name.
    pub const NAME: &'static str = "positive-distance-m";

    /// Admits a finite, strictly positive distance.
    pub fn new(value: f64) -> Result<Self, InterferenceError> {
        positive(Self::NAME, value).map(Self)
    }

    /// Returns the validated value in metres.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A finite frequency in hertz that is strictly greater than zero.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Hertz(f64);

impl Hertz {
    /// Stable diagnostic name.
    pub const NAME: &'static str = "frequency-hz";

    /// Admits a finite, strictly positive frequency.
    pub fn new(value: f64) -> Result<Self, InterferenceError> {
        positive(Self::NAME, value).map(Self)
    }

    /// Returns the validated value in hertz.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A finite propagation speed in metres per second.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct MetresPerSecond(f64);

impl MetresPerSecond {
    /// Stable diagnostic name.
    pub const NAME: &'static str = "speed-m-s";

    /// Admits a finite, strictly positive propagation speed.
    pub fn new(value: f64) -> Result<Self, InterferenceError> {
        positive(Self::NAME, value).map(Self)
    }

    /// Returns the validated value in metres per second.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A finite non-negative attenuation coefficient in nepers per metre.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct NepersPerMetre(f64);

impl NepersPerMetre {
    /// Stable diagnostic name.
    pub const NAME: &'static str = "attenuation-np-m";

    /// Admits finite attenuation greater than or equal to zero.
    pub fn new(value: f64) -> Result<Self, InterferenceError> {
        non_negative(Self::NAME, value).map(Self)
    }

    /// Returns the validated value in nepers per metre.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A finite phase angle stored in the half-open interval `[-pi, pi)`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Radians(f64);

impl Radians {
    /// Stable diagnostic name.
    pub const NAME: &'static str = "phase-rad";

    /// Admits a finite angle and normalizes whole turns without rounding.
    pub fn new(value: f64) -> Result<Self, InterferenceError> {
        let value = finite(Self::NAME, value)?;
        let positive_turn = value.rem_euclid(TAU);
        let normalized = if positive_turn >= PI {
            positive_turn - TAU
        } else {
            positive_turn
        };
        Ok(Self(canonical_zero(normalized)))
    }

    /// Returns the normalized value in radians.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A finite non-negative scalar field amplitude.
///
/// A negative real contribution is represented by phase, not by a signed
/// amplitude, so zero is admitted while negative values are rejected.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct FieldAmplitude(f64);

impl FieldAmplitude {
    /// Stable diagnostic name.
    pub const NAME: &'static str = "field-amplitude";

    /// Admits a finite amplitude greater than or equal to zero.
    pub fn new(value: f64) -> Result<Self, InterferenceError> {
        non_negative(Self::NAME, value).map(Self)
    }

    /// Returns the validated field amplitude.
    pub fn get(self) -> f64 {
        self.0
    }
}
