//! Host f64 phase anchors and source-specific normalized constants.

use sim_lib_interference_core::{
    Emitter, InterferenceProblem, POINT_SOURCE_REFERENCE_DISTANCE_METRES, Point3M,
};

use crate::{LoweringError, PhaseBudget, PreflightCheck};

/// Tile-center constants for a spherical point source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointTileConstants {
    /// Unit direction from the source to the tile center.
    pub n0: [f32; 3],
    /// Reciprocal source-to-center distance.
    pub rho: f32,
    /// Host f64 cosine of the absolute phase, rounded once to f32.
    pub phase_cos: f32,
    /// Host f64 sine of the absolute phase, rounded once to f32.
    pub phase_sin: f32,
    /// Attenuated point-source gain at the tile center.
    pub gain0: f32,
}

/// Tile-center constants for a forward plane source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneTileConstants {
    /// Forward unit direction, rounded once from host f64.
    pub direction: [f32; 3],
    /// Host f64 signed distance from the phase plane to the tile center.
    pub signed_distance0_m: f64,
    /// Host f64 cosine of the absolute phase, rounded once to f32.
    pub phase_cos: f32,
    /// Host f64 sine of the absolute phase, rounded once to f32.
    pub phase_sin: f32,
    /// Host f64 attenuated gain at the tile center, rounded once to f32.
    pub gain0: f32,
}

/// Predicted phase-domain accuracy for one source on one tile.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SourcePhaseEstimate {
    /// Largest absolute f32 residual phase submitted to `sin` or `cos`.
    pub max_abs_residual_phase_rad: f64,
    /// Largest phase discrepancy caused by lowering local geometry to f32.
    pub max_predicted_geometry_error_rad: f64,
    /// Largest predicted f32 wavenumber, multiply, and operation roundoff.
    pub max_predicted_roundoff_error_rad: f64,
}

/// Prepared constants and phase estimate for one source on one tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SourceTileConstants {
    /// Spherical point-source constants.
    Point {
        /// Constants consumed by the Tensor lowering.
        constants: PointTileConstants,
        /// Preflight estimate for this source and tile.
        estimate: SourcePhaseEstimate,
    },
    /// Forward plane-wave constants.
    ForwardPlane {
        /// Constants consumed by the Tensor lowering.
        constants: PlaneTileConstants,
        /// Preflight estimate for this source and tile.
        estimate: SourcePhaseEstimate,
    },
}

impl SourceTileConstants {
    /// Returns the preflight phase estimate.
    pub fn estimate(self) -> SourcePhaseEstimate {
        match self {
            Self::Point { estimate, .. } | Self::ForwardPlane { estimate, .. } => estimate,
        }
    }
}

pub(crate) fn prepare_source_constants(
    problem: &InterferenceProblem,
    source: &Emitter,
    center: Point3M,
    local_f64: &[[f64; 3]],
    local_f32: &[[f32; 3]],
    budget: PhaseBudget,
) -> Result<SourceTileConstants, LoweringError> {
    let estimates = EstimateInputs {
        wavenumber: problem.wavenumber().real_radians_per_metre(),
        local_f64,
        local_f32,
        budget,
    };
    match source {
        Emitter::Point {
            id,
            position,
            amplitude_at_reference,
            phase,
        } => {
            let center_xyz = center.coordinates_metres();
            let source_xyz = position.coordinates_metres();
            let w0 = [
                center_xyz[0] - source_xyz[0],
                center_xyz[1] - source_xyz[1],
                center_xyz[2] - source_xyz[2],
            ];
            let r0 = norm(w0);
            if !r0.is_finite() || r0 <= problem.singularity_radius.get() {
                return Err(LoweringError::new(
                    PreflightCheck::Singularity,
                    format!(
                        "point source {id} tile center distance {r0} is at or inside {}",
                        problem.singularity_radius.get()
                    ),
                ));
            }
            let rho = 1.0 / r0;
            let n0 = [w0[0] / r0, w0[1] / r0, w0[2] / r0];
            let wave = problem.wavenumber();
            let phase0 = wave.real_radians_per_metre().mul_add(r0, phase.get());
            let gain0 = point_gain(
                amplitude_at_reference.get(),
                wave.imaginary_nepers_per_metre(),
                r0,
            )?;
            let constants = PointTileConstants {
                n0: [
                    finite_f32("point n0.x", n0[0], false)?,
                    finite_f32("point n0.y", n0[1], false)?,
                    finite_f32("point n0.z", n0[2], false)?,
                ],
                rho: finite_f32("point rho", rho, true)?,
                phase_cos: finite_f32("point phase cosine", phase0.cos(), false)?,
                phase_sin: finite_f32("point phase sine", phase0.sin(), false)?,
                gain0: finite_f32("point center gain", gain0, gain0 != 0.0)?,
            };
            let estimate = point_estimate(
                id,
                w0,
                problem.singularity_radius.get(),
                constants,
                estimates,
            )?;
            Ok(SourceTileConstants::Point {
                constants,
                estimate,
            })
        }
        Emitter::ForwardPlane {
            id,
            through,
            direction,
            amplitude,
            phase,
        } => {
            let signed_distance0_m = direction.signed_distance_metres(*through, center);
            if !signed_distance0_m.is_finite() {
                return Err(LoweringError::new(
                    PreflightCheck::ForwardPlane,
                    format!("forward plane {id} has a non-finite center distance"),
                ));
            }
            let wave = problem.wavenumber();
            let phase0 = wave
                .real_radians_per_metre()
                .mul_add(signed_distance0_m, phase.get());
            let gain0 = plane_gain(
                amplitude.get(),
                wave.imaginary_nepers_per_metre(),
                signed_distance0_m,
            )?;
            let direction64 = direction.components();
            let constants = PlaneTileConstants {
                direction: [
                    finite_f32("plane direction.x", direction64[0], false)?,
                    finite_f32("plane direction.y", direction64[1], false)?,
                    finite_f32("plane direction.z", direction64[2], false)?,
                ],
                signed_distance0_m,
                phase_cos: finite_f32("plane phase cosine", phase0.cos(), false)?,
                phase_sin: finite_f32("plane phase sine", phase0.sin(), false)?,
                gain0: finite_f32("plane center gain", gain0, gain0 != 0.0)?,
            };
            let estimate = plane_estimate(
                id,
                signed_distance0_m,
                direction64,
                constants.direction,
                estimates,
            )?;
            Ok(SourceTileConstants::ForwardPlane {
                constants,
                estimate,
            })
        }
    }
}

#[derive(Clone, Copy)]
struct EstimateInputs<'a> {
    wavenumber: f64,
    local_f64: &'a [[f64; 3]],
    local_f32: &'a [[f32; 3]],
    budget: PhaseBudget,
}

fn point_estimate(
    source_id: &str,
    w0: [f64; 3],
    singularity_radius: f64,
    constants: PointTileConstants,
    inputs: EstimateInputs<'_>,
) -> Result<SourcePhaseEstimate, LoweringError> {
    let mut estimate = SourcePhaseEstimate::default();
    let k32 = finite_f32("real wavenumber", inputs.wavenumber, true)?;
    for (offset64, offset32) in inputs.local_f64.iter().zip(inputs.local_f32) {
        let sample_vector = [
            w0[0] + offset64[0],
            w0[1] + offset64[1],
            w0[2] + offset64[2],
        ];
        let distance = norm(sample_vector);
        if !distance.is_finite() || distance <= singularity_radius {
            return Err(LoweringError::new(
                PreflightCheck::Singularity,
                format!(
                    "point source {source_id} sample distance {distance} is at or inside {singularity_radius}"
                ),
            ));
        }
        let exact_delta = normalized_point_delta_f64(w0, *offset64)?;
        let lowered_delta = normalized_point_delta_f32(constants, *offset32)?;
        update_estimate(
            &mut estimate,
            inputs.wavenumber,
            k32,
            exact_delta,
            lowered_delta,
        );
        let gain_denominator = 1.0_f32 + constants.rho * lowered_delta;
        if !gain_denominator.is_finite() || gain_denominator <= 0.0 {
            return Err(LoweringError::new(
                PreflightCheck::Denominator,
                format!("point source {source_id} gain denominator is {gain_denominator}"),
            ));
        }
    }
    admit_estimate(source_id, estimate, inputs.budget)?;
    Ok(estimate)
}

fn plane_estimate(
    source_id: &str,
    signed_distance0_m: f64,
    direction64: [f64; 3],
    direction32: [f32; 3],
    inputs: EstimateInputs<'_>,
) -> Result<SourcePhaseEstimate, LoweringError> {
    let mut estimate = SourcePhaseEstimate::default();
    let k32 = finite_f32("real wavenumber", inputs.wavenumber, true)?;
    for (offset64, offset32) in inputs.local_f64.iter().zip(inputs.local_f32) {
        let exact_delta = dot64(direction64, *offset64);
        let signed_distance = signed_distance0_m + exact_delta;
        if !signed_distance.is_finite() || signed_distance < 0.0 {
            return Err(LoweringError::new(
                PreflightCheck::ForwardPlane,
                format!("forward plane {source_id} sample signed distance is {signed_distance}"),
            ));
        }
        let lowered_delta = dot32(direction32, *offset32);
        update_estimate(
            &mut estimate,
            inputs.wavenumber,
            k32,
            exact_delta,
            lowered_delta,
        );
    }
    admit_estimate(source_id, estimate, inputs.budget)?;
    Ok(estimate)
}

fn update_estimate(
    estimate: &mut SourcePhaseEstimate,
    wavenumber: f64,
    wavenumber32: f32,
    exact_delta: f64,
    lowered_delta: f32,
) {
    let psi32 = wavenumber32 * lowered_delta;
    let geometry = wavenumber * (lowered_delta as f64 - exact_delta).abs();
    let ideal_with_lowered_geometry = wavenumber * lowered_delta as f64;
    let arithmetic = (psi32 as f64 - ideal_with_lowered_geometry).abs()
        + 16.0 * f32::EPSILON as f64 * (1.0 + (psi32 as f64).abs());
    estimate.max_abs_residual_phase_rad = estimate
        .max_abs_residual_phase_rad
        .max((psi32 as f64).abs());
    estimate.max_predicted_geometry_error_rad =
        estimate.max_predicted_geometry_error_rad.max(geometry);
    estimate.max_predicted_roundoff_error_rad =
        estimate.max_predicted_roundoff_error_rad.max(arithmetic);
}

fn admit_estimate(
    source_id: &str,
    estimate: SourcePhaseEstimate,
    budget: PhaseBudget,
) -> Result<(), LoweringError> {
    for (check, name, predicted, limit) in [
        (
            PreflightCheck::PhaseBudget,
            "absolute residual phase",
            estimate.max_abs_residual_phase_rad,
            budget.max_abs_residual_phase_rad,
        ),
        (
            PreflightCheck::GeometryBudget,
            "predicted geometry error",
            estimate.max_predicted_geometry_error_rad,
            budget.max_predicted_geometry_error_rad,
        ),
        (
            PreflightCheck::RoundoffBudget,
            "predicted roundoff error",
            estimate.max_predicted_roundoff_error_rad,
            budget.max_predicted_roundoff_error_rad,
        ),
    ] {
        if !predicted.is_finite() || predicted > limit * (1.0 + 8.0 * f64::EPSILON) {
            return Err(LoweringError::new(
                check,
                format!("source {source_id} {name} {predicted} exceeds {limit}"),
            ));
        }
    }
    Ok(())
}

fn normalized_point_delta_f64(w0: [f64; 3], offset: [f64; 3]) -> Result<f64, LoweringError> {
    let r0 = norm(w0);
    let rho = 1.0 / r0;
    let n0 = [w0[0] * rho, w0[1] * rho, w0[2] * rho];
    let a = dot64(n0, offset);
    let b = dot64(offset, offset);
    let z = 2.0 * a * rho + b * rho * rho;
    let denominator = (1.0 + z).sqrt() + 1.0;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(LoweringError::new(
            PreflightCheck::Denominator,
            format!("host normalized-distance denominator is {denominator}"),
        ));
    }
    Ok((2.0 * a + b * rho) / denominator)
}

fn normalized_point_delta_f32(
    constants: PointTileConstants,
    offset: [f32; 3],
) -> Result<f32, LoweringError> {
    let a = dot32(constants.n0, offset);
    let b = dot32(offset, offset);
    let z = (2.0 * a) * constants.rho + (b * constants.rho) * constants.rho;
    let sqrt_argument = 1.0 + z;
    if !sqrt_argument.is_finite() || sqrt_argument < 0.0 {
        return Err(LoweringError::new(
            PreflightCheck::Denominator,
            format!("normalized-distance sqrt argument is {sqrt_argument}"),
        ));
    }
    let denominator = sqrt_argument.sqrt() + 1.0;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(LoweringError::new(
            PreflightCheck::Denominator,
            format!("normalized-distance denominator is {denominator}"),
        ));
    }
    let delta = ((2.0 * a) + b * constants.rho) / denominator;
    if !delta.is_finite() {
        return Err(LoweringError::new(
            PreflightCheck::Denominator,
            "normalized-distance residual is not finite",
        ));
    }
    Ok(delta)
}

fn point_gain(amplitude: f64, alpha: f64, distance: f64) -> Result<f64, LoweringError> {
    let attenuation = attenuation(alpha, distance)?;
    let gain = amplitude * POINT_SOURCE_REFERENCE_DISTANCE_METRES * attenuation / distance;
    if gain.is_finite() {
        Ok(gain)
    } else {
        Err(LoweringError::new(
            PreflightCheck::Constant,
            "point center gain is not finite",
        ))
    }
}

fn plane_gain(amplitude: f64, alpha: f64, distance: f64) -> Result<f64, LoweringError> {
    if distance < 0.0 {
        return Err(LoweringError::new(
            PreflightCheck::ForwardPlane,
            format!("forward-plane center signed distance is {distance}"),
        ));
    }
    let gain = amplitude * attenuation(alpha, distance)?;
    if gain.is_finite() {
        Ok(gain)
    } else {
        Err(LoweringError::new(
            PreflightCheck::Constant,
            "plane center gain is not finite",
        ))
    }
}

fn attenuation(alpha: f64, distance: f64) -> Result<f64, LoweringError> {
    let exponent = alpha * distance;
    if exponent.is_infinite() && exponent.is_sign_positive() {
        Ok(0.0)
    } else if exponent.is_finite() {
        Ok((-exponent).exp())
    } else {
        Err(LoweringError::new(
            PreflightCheck::Constant,
            "attenuation exponent is invalid",
        ))
    }
}

fn finite_f32(name: &str, value: f64, preserve_nonzero: bool) -> Result<f32, LoweringError> {
    let lowered = value as f32;
    if !value.is_finite()
        || !lowered.is_finite()
        || (preserve_nonzero && value != 0.0 && lowered == 0.0)
    {
        Err(LoweringError::new(
            PreflightCheck::Constant,
            format!("{name} value {value} is not safely representable as f32"),
        ))
    } else {
        Ok(lowered)
    }
}

fn norm(vector: [f64; 3]) -> f64 {
    vector[0].hypot(vector[1]).hypot(vector[2])
}

fn dot64(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn dot32(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}
