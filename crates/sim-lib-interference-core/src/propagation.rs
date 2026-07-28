//! Pure `f64` Green-function contributions at one spatial point.

use crate::{
    Emitter, FieldAmplitude, InterferenceError, InterferenceProblem,
    POINT_SOURCE_REFERENCE_DISTANCE_METRES, Point3M, Radians, UnitVector3,
};

fn require_finite(
    source_id: &str,
    name: &'static str,
    value: f64,
) -> Result<f64, InterferenceError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| InterferenceError::NonFinitePropagation {
            source_id: source_id.to_owned(),
            name,
            value,
        })
}

fn cartesian_components(
    source_id: &str,
    gain: f64,
    angle: f64,
) -> Result<(f64, f64), InterferenceError> {
    let gain = require_finite(source_id, "gain", gain)?;
    let angle = require_finite(source_id, "phase-radians", angle)?;
    Ok((gain * angle.cos(), gain * angle.sin()))
}

/// Evaluates one emitter's complex field contribution at one point.
///
/// The tuple is `(real, imaginary)`. This is the dependency-free `f64`
/// executable specification for a single source, not a grid solver or an
/// accumulation policy.
pub fn contribution_at(
    problem: &InterferenceProblem,
    source: &Emitter,
    at: Point3M,
) -> Result<(f64, f64), InterferenceError> {
    match source {
        Emitter::Point {
            id,
            position,
            amplitude_at_reference,
            phase,
        } => point_contribution_at(problem, id, *position, *amplitude_at_reference, *phase, at),
        Emitter::ForwardPlane {
            id,
            through,
            direction,
            amplitude,
            phase,
        } => {
            forward_plane_contribution_at(problem, id, *through, *direction, *amplitude, *phase, at)
        }
    }
}

/// Evaluates the outgoing point-source Green function at one point.
///
/// For distance `r`, this returns the Cartesian components of
/// `A * (R_ref / r) * exp(i * k_tilde * r + i * phase)`. A sample at or inside
/// the problem's singularity radius is rejected rather than clamped.
pub fn point_contribution_at(
    problem: &InterferenceProblem,
    source_id: &str,
    position: Point3M,
    amplitude_at_reference: FieldAmplitude,
    phase: Radians,
    at: Point3M,
) -> Result<(f64, f64), InterferenceError> {
    let distance = require_finite(source_id, "point-distance-metres", at.distance_to(position))?;
    if distance <= problem.singularity_radius.get() {
        return Err(InterferenceError::SingularPointSample {
            source_id: source_id.to_owned(),
            distance_metres: distance,
            singularity_radius_metres: problem.singularity_radius.get(),
        });
    }

    let wave_number = problem.wavenumber();
    let phase_advance = require_finite(
        source_id,
        "propagation-phase-radians",
        wave_number.real_radians_per_metre() * distance,
    )?;
    let attenuation_exponent = wave_number.imaginary_nepers_per_metre() * distance;
    let attenuation =
        if attenuation_exponent.is_infinite() && attenuation_exponent.is_sign_positive() {
            0.0
        } else {
            (-require_finite(source_id, "attenuation-exponent", attenuation_exponent)?).exp()
        };
    let gain = amplitude_at_reference.get() * POINT_SOURCE_REFERENCE_DISTANCE_METRES * attenuation
        / distance;

    cartesian_components(source_id, gain, phase_advance + phase.get())
}

/// Evaluates a forward-plane Green function at one point.
///
/// For signed distance `s`, this returns the Cartesian components of
/// `A * exp(i * k_tilde * s + i * phase)`. Only the forward half-space
/// `s >= 0` belongs to this emitter model.
pub fn forward_plane_contribution_at(
    problem: &InterferenceProblem,
    source_id: &str,
    through: Point3M,
    direction: UnitVector3,
    amplitude: FieldAmplitude,
    phase: Radians,
    at: Point3M,
) -> Result<(f64, f64), InterferenceError> {
    let signed_distance = require_finite(
        source_id,
        "plane-signed-distance-metres",
        direction.signed_distance_metres(through, at),
    )?;
    if signed_distance < 0.0 {
        return Err(InterferenceError::BehindForwardPlane {
            source_id: source_id.to_owned(),
            signed_distance_metres: signed_distance,
        });
    }

    let wave_number = problem.wavenumber();
    let phase_advance = require_finite(
        source_id,
        "propagation-phase-radians",
        wave_number.real_radians_per_metre() * signed_distance,
    )?;
    let attenuation_exponent = wave_number.imaginary_nepers_per_metre() * signed_distance;
    let attenuation =
        if attenuation_exponent.is_infinite() && attenuation_exponent.is_sign_positive() {
            0.0
        } else {
            (-require_finite(source_id, "attenuation-exponent", attenuation_exponent)?).exp()
        };
    let gain = amplitude.get() * attenuation;

    cartesian_components(source_id, gain, phase_advance + phase.get())
}
