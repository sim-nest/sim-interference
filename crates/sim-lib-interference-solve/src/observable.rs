//! Scalar observations derived from one coherent phasor field.

/// A scalar view of a coherent complex field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Observable {
    /// Cartesian real component.
    Real,
    /// Cartesian imaginary component.
    Imaginary,
    /// Euclidean amplitude `hypot(real, imaginary)`.
    Amplitude,
    /// Wrapped phase in the half-open interval `[-pi, pi)`.
    ///
    /// Phase at or below the projection's amplitude floor is represented by a
    /// masked sample rather than a number.
    Phase,
    /// Normalized `real * real + imaginary * imaginary` energy proxy.
    ///
    /// This is not physical intensity. Physical intensity requires a named
    /// field and medium impedance, which the scalar model does not carry.
    MagnitudeSquared,
    /// Instantaneous field `Re{U * exp(-i * wt)}`.
    ///
    /// Under the crate's time convention this is
    /// `real * cos(wt) + imaginary * sin(wt)`. At `wt == 0`, projection
    /// returns the real component directly so equality is bit-exact.
    Instant {
        /// Angular time in radians.
        wt: f64,
    },
}
