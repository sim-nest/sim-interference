//! Metamorphic fixtures and complete reference-suite orchestration.

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, SamplingPolicy, SamplingThresholds, ScalarMedium,
    SourceSet, UnitVector3, WorkBudget,
};

use crate::{
    HostPhasorField, ReferencePhasorSolver, ReferenceSolveError,
    analytic::measure_analytic_fixtures,
    helmholtz::{MAX_OBSERVED_ORDER, MIN_OBSERVED_ORDER, measure_helmholtz_fixture_matrix},
    verification::{
        MAX_ANALYTIC_RELATIVE_ERROR, MAX_METAMORPHIC_RELATIVE_ERROR, VerificationError,
        VerificationReport,
    },
};

const FIXTURE_FREQUENCY_HERTZ: f64 = 7.0;
const FIXTURE_SPEED_METRES_PER_SECOND: f64 = 28.0;

/// Runs the analytic, metamorphic, time-sign, and Helmholtz fixture matrix.
///
/// The suite covers point-only, plane-only, mixed, attenuating, and
/// multi-source fields. It returns no report unless every relative error is
/// bounded, source permutations are bit-identical, and every seven-point
/// residual comparison has observed order in `[1.8, 2.2]`.
pub fn verify_reference_solver() -> Result<VerificationReport, VerificationError> {
    let analytic = measure_analytic_fixtures()?;
    let metamorphic = measure_metamorphic_fixtures()?;
    require_relative(
        "analytic",
        analytic.max_error(),
        MAX_ANALYTIC_RELATIVE_ERROR,
    )?;
    for (check, measured) in [
        ("point-reciprocity", metamorphic.reciprocity_relative),
        ("linearity", metamorphic.linearity_relative),
        ("rigid-motion", metamorphic.rigid_motion_relative),
        ("global-phase", metamorphic.global_phase_relative),
    ] {
        require_relative(check, measured, MAX_METAMORPHIC_RELATIVE_ERROR)?;
    }
    if !metamorphic.source_permutation_identical {
        return Err(VerificationError::SourcePermutationChanged);
    }

    let helmholtz = measure_helmholtz_fixture_matrix()?;
    for named in &helmholtz {
        if !(MIN_OBSERVED_ORDER..=MAX_OBSERVED_ORDER).contains(&named.measurement.observed_order) {
            return Err(VerificationError::HelmholtzOrderOutOfRange {
                fixture: named.name,
                observed: named.measurement.observed_order,
                minimum: MIN_OBSERVED_ORDER,
                maximum: MAX_OBSERVED_ORDER,
            });
        }
    }
    let least_ideal_order = helmholtz
        .iter()
        .max_by(|left, right| {
            (left.measurement.observed_order - 2.0)
                .abs()
                .total_cmp(&(right.measurement.observed_order - 2.0).abs())
        })
        .expect("the fixed Helmholtz fixture matrix is non-empty")
        .measurement
        .observed_order;

    Ok(VerificationReport {
        analytic_max_error: analytic.max_error(),
        reciprocity_relative: metamorphic.reciprocity_relative,
        linearity_relative: metamorphic.linearity_relative,
        rigid_motion_relative: metamorphic.rigid_motion_relative,
        global_phase_relative: metamorphic.global_phase_relative,
        source_permutation_identical: metamorphic.source_permutation_identical,
        helmholtz_observed_order: least_ideal_order,
    })
}

fn require_relative(
    check: &'static str,
    measured: f64,
    limit: f64,
) -> Result<(), VerificationError> {
    if measured.is_finite() && measured <= limit {
        Ok(())
    } else {
        Err(VerificationError::RelativeErrorExceeded {
            check,
            measured,
            limit,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MetamorphicMetrics {
    pub(crate) reciprocity_relative: f64,
    pub(crate) linearity_relative: f64,
    pub(crate) rigid_motion_relative: f64,
    pub(crate) global_phase_relative: f64,
    pub(crate) source_permutation_identical: bool,
}

pub(crate) fn measure_metamorphic_fixtures() -> Result<MetamorphicMetrics, ReferenceSolveError> {
    Ok(MetamorphicMetrics {
        reciprocity_relative: point_reciprocity_error()?,
        linearity_relative: linearity_error()?,
        rigid_motion_relative: rigid_motion_error()?,
        global_phase_relative: global_phase_error()?,
        source_permutation_identical: source_permutation_identical()?,
    })
}

fn point_reciprocity_error() -> Result<f64, ReferenceSolveError> {
    let first = point(-0.75, 0.5, -1.25)?;
    let second = point(1.25, -0.25, 0.75)?;
    let from_first = problem(vec![point_source("first", first, 1.0, 0.0)?], 0.08)?;
    let from_second = problem(vec![point_source("second", second, 1.0, 0.0)?], 0.08)?;
    let first_to_second = solve(&from_first, &point_plane(second)?)?;
    let second_to_first = solve(&from_second, &point_plane(first)?)?;

    Ok(relative_complex_error(
        first_to_second.cell(0, 0).unwrap(),
        second_to_first.cell(0, 0).unwrap(),
    ))
}

fn linearity_error() -> Result<f64, ReferenceSolveError> {
    let sample_plane = fixture_plane()?;
    let first_source = point_source("linear-point", point(-0.5, 0.25, -3.0)?, 2.0, 0.3)?;
    let second_source = forward_plane(
        "linear-plane",
        point(0.0, 0.0, -1.0)?,
        direction(0.0, 0.0, 1.0)?,
        0.75,
        -0.6,
    )?;
    let first = solve(&problem(vec![first_source.clone()], 0.02)?, &sample_plane)?;
    let second = solve(&problem(vec![second_source.clone()], 0.02)?, &sample_plane)?;
    let combined = solve(
        &problem(vec![second_source, first_source], 0.02)?,
        &sample_plane,
    )?;

    Ok(relative_field_sum_error(&combined, &first, &second))
}

fn rigid_motion_error() -> Result<f64, ReferenceSolveError> {
    let sources = vec![
        point_source("rigid-point", point(-1.0, 0.5, -4.0)?, 2.25, 0.4)?,
        forward_plane(
            "rigid-plane",
            point(0.0, 0.0, -2.0)?,
            direction(0.0, 0.0, 1.0)?,
            0.8,
            -0.7,
        )?,
    ];
    let transformed_sources = sources
        .iter()
        .map(rigid_emitter)
        .collect::<Result<Vec<_>, _>>()?;
    let original_plane = fixture_plane()?;
    let transformed_plane = rigid_plane(original_plane)?;
    let original = solve(&problem(sources, 0.03)?, &original_plane)?;
    let transformed = solve(&problem(transformed_sources, 0.03)?, &transformed_plane)?;

    Ok(relative_field_error(&original, &transformed))
}

fn global_phase_error() -> Result<f64, ReferenceSolveError> {
    let phase_shift = 0.625;
    let sources = vec![
        point_source("phase-point", point(-0.75, 0.25, -3.5)?, 1.5, 0.2)?,
        forward_plane(
            "phase-plane",
            point(0.0, 0.0, -1.5)?,
            direction(0.0, 0.0, 1.0)?,
            0.9,
            -0.4,
        )?,
    ];
    let shifted_sources = sources
        .iter()
        .map(|source| phase_shifted_emitter(source, phase_shift))
        .collect::<Result<Vec<_>, _>>()?;
    let sample_plane = fixture_plane()?;
    let original = solve(&problem(sources, 0.015)?, &sample_plane)?;
    let shifted = solve(&problem(shifted_sources, 0.015)?, &sample_plane)?;

    Ok(global_phase_field_error(&shifted, &original, phase_shift))
}

fn source_permutation_identical() -> Result<bool, ReferenceSolveError> {
    let sample_plane = fixture_plane()?;
    let solve_order = |order: [&str; 3]| -> Result<HostPhasorField, ReferenceSolveError> {
        let sources = order
            .into_iter()
            .map(permutation_source)
            .collect::<Result<Vec<_>, _>>()?;
        solve(&problem(sources, 0.025)?, &sample_plane)
    };
    let baseline = solve_order(["point-a", "point-b", "plane"])?;
    let reversed = solve_order(["plane", "point-b", "point-a"])?;
    let rotated = solve_order(["point-b", "plane", "point-a"])?;

    Ok(component_bits(&baseline) == component_bits(&reversed)
        && component_bits(&baseline) == component_bits(&rotated))
}

fn permutation_source(id: &str) -> Result<Emitter, ReferenceSolveError> {
    match id {
        "point-a" => point_source("point-a", point(-0.5, 0.75, -3.0)?, 1.5, 0.1),
        "point-b" => point_source("point-b", point(0.75, -0.25, -2.5)?, 0.75, -0.3),
        "plane" => forward_plane(
            "plane",
            point(0.0, 0.0, -1.0)?,
            direction(0.0, 0.0, 1.0)?,
            0.5,
            0.8,
        ),
        _ => unreachable!("the fixture lists only known source ids"),
    }
}

fn solve(
    problem: &InterferenceProblem,
    plane: &SamplingPlane,
) -> Result<HostPhasorField, ReferenceSolveError> {
    ReferencePhasorSolver::new(
        SamplingPolicy::Annotate,
        SamplingThresholds::default(),
        WorkBudget::default(),
    )
    .solve(problem, plane)
    .map(|(field, _)| field)
}

fn problem(
    sources: Vec<Emitter>,
    attenuation: f64,
) -> Result<InterferenceProblem, ReferenceSolveError> {
    Ok(InterferenceProblem::new(
        Hertz::new(FIXTURE_FREQUENCY_HERTZ)?,
        ScalarMedium::new(
            MetresPerSecond::new(FIXTURE_SPEED_METRES_PER_SECOND)?,
            NepersPerMetre::new(attenuation)?,
        ),
        SourceSet::new(sources)?,
        PositiveMetres::new(1.0e-6)?,
    ))
}

fn point_source(
    id: &str,
    position: Point3M,
    amplitude: f64,
    phase: f64,
) -> Result<Emitter, ReferenceSolveError> {
    Ok(Emitter::Point {
        id: id.to_owned(),
        position,
        amplitude_at_reference: FieldAmplitude::new(amplitude)?,
        phase: Radians::new(phase)?,
    })
}

fn forward_plane(
    id: &str,
    through: Point3M,
    direction: UnitVector3,
    amplitude: f64,
    phase: f64,
) -> Result<Emitter, ReferenceSolveError> {
    Ok(Emitter::ForwardPlane {
        id: id.to_owned(),
        through,
        direction,
        amplitude: FieldAmplitude::new(amplitude)?,
        phase: Radians::new(phase)?,
    })
}

fn phase_shifted_emitter(source: &Emitter, shift: f64) -> Result<Emitter, ReferenceSolveError> {
    Ok(match source {
        Emitter::Point {
            id,
            position,
            amplitude_at_reference,
            phase,
        } => Emitter::Point {
            id: id.clone(),
            position: *position,
            amplitude_at_reference: *amplitude_at_reference,
            phase: Radians::new(phase.get() + shift)?,
        },
        Emitter::ForwardPlane {
            id,
            through,
            direction,
            amplitude,
            phase,
        } => Emitter::ForwardPlane {
            id: id.clone(),
            through: *through,
            direction: *direction,
            amplitude: *amplitude,
            phase: Radians::new(phase.get() + shift)?,
        },
    })
}

fn rigid_emitter(source: &Emitter) -> Result<Emitter, ReferenceSolveError> {
    Ok(match source {
        Emitter::Point {
            id,
            position,
            amplitude_at_reference,
            phase,
        } => Emitter::Point {
            id: id.clone(),
            position: rigid_point(*position)?,
            amplitude_at_reference: *amplitude_at_reference,
            phase: *phase,
        },
        Emitter::ForwardPlane {
            id,
            through,
            direction,
            amplitude,
            phase,
        } => Emitter::ForwardPlane {
            id: id.clone(),
            through: rigid_point(*through)?,
            direction: rigid_direction(*direction)?,
            amplitude: *amplitude,
            phase: *phase,
        },
    })
}

fn rigid_plane(plane: SamplingPlane) -> Result<SamplingPlane, ReferenceSolveError> {
    SamplingPlane::new(
        rigid_point(plane.origin())?,
        rigid_direction(plane.u_axis())?,
        rigid_direction(plane.v_axis())?,
        plane.extent_u(),
        plane.extent_v(),
        plane.rows(),
        plane.columns(),
    )
    .map_err(ReferenceSolveError::from)
}

fn rigid_point(value: Point3M) -> Result<Point3M, ReferenceSolveError> {
    let [x, y, z] = value.coordinates_metres();
    point(-y + 3.0, x - 2.0, z + 5.0)
}

fn rigid_direction(value: UnitVector3) -> Result<UnitVector3, ReferenceSolveError> {
    let [x, y, z] = value.components();
    direction(-y, x, z)
}

fn fixture_plane() -> Result<SamplingPlane, ReferenceSolveError> {
    SamplingPlane::new(
        point(-0.5, -0.75, 0.0)?,
        direction(1.0, 0.0, 0.0)?,
        direction(0.0, 1.0, 0.0)?,
        PositiveMetres::new(1.0)?,
        PositiveMetres::new(1.5)?,
        3,
        4,
    )
    .map_err(ReferenceSolveError::from)
}

fn point_plane(at: Point3M) -> Result<SamplingPlane, ReferenceSolveError> {
    let [x, y, z] = at.coordinates_metres();
    SamplingPlane::new(
        point(x - 0.125, y - 0.125, z)?,
        direction(1.0, 0.0, 0.0)?,
        direction(0.0, 1.0, 0.0)?,
        PositiveMetres::new(0.25)?,
        PositiveMetres::new(0.25)?,
        1,
        1,
    )
    .map_err(ReferenceSolveError::from)
}

fn point(x: f64, y: f64, z: f64) -> Result<Point3M, ReferenceSolveError> {
    Point3M::from_metres(x, y, z).map_err(ReferenceSolveError::from)
}

fn direction(x: f64, y: f64, z: f64) -> Result<UnitVector3, ReferenceSolveError> {
    UnitVector3::new(x, y, z).map_err(ReferenceSolveError::from)
}

fn relative_complex_error(actual: (f64, f64), expected: (f64, f64)) -> f64 {
    let difference = (actual.0 - expected.0).hypot(actual.1 - expected.1);
    let scale = actual
        .0
        .hypot(actual.1)
        .max(expected.0.hypot(expected.1))
        .max(f64::MIN_POSITIVE);
    difference / scale
}

fn relative_field_error(actual: &HostPhasorField, expected: &HostPhasorField) -> f64 {
    actual
        .real()
        .iter()
        .zip(actual.imaginary())
        .zip(expected.real().iter().zip(expected.imaginary()))
        .map(
            |((&actual_real, &actual_imaginary), (&expected_real, &expected_imaginary))| {
                relative_complex_error(
                    (actual_real, actual_imaginary),
                    (expected_real, expected_imaginary),
                )
            },
        )
        .fold(0.0, f64::max)
}

fn relative_field_sum_error(
    actual: &HostPhasorField,
    first: &HostPhasorField,
    second: &HostPhasorField,
) -> f64 {
    (0..actual.len())
        .map(|index| {
            relative_complex_error(
                (actual.real()[index], actual.imaginary()[index]),
                (
                    first.real()[index] + second.real()[index],
                    first.imaginary()[index] + second.imaginary()[index],
                ),
            )
        })
        .fold(0.0, f64::max)
}

fn global_phase_field_error(
    actual: &HostPhasorField,
    original: &HostPhasorField,
    phase_shift: f64,
) -> f64 {
    let cosine = phase_shift.cos();
    let sine = phase_shift.sin();
    (0..actual.len())
        .map(|index| {
            let real = original.real()[index];
            let imaginary = original.imaginary()[index];
            relative_complex_error(
                (actual.real()[index], actual.imaginary()[index]),
                (
                    real * cosine - imaginary * sine,
                    real * sine + imaginary * cosine,
                ),
            )
        })
        .fold(0.0, f64::max)
}

fn component_bits(field: &HostPhasorField) -> (Vec<u64>, Vec<u64>) {
    (
        field.real().iter().map(|value| value.to_bits()).collect(),
        field
            .imaginary()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::measure_metamorphic_fixtures;

    #[test]
    fn reciprocity_linearity_rigid_motion_phase_and_permutation_laws_hold() {
        let metrics = measure_metamorphic_fixtures().unwrap();

        assert!(metrics.reciprocity_relative <= 2.0e-15, "{metrics:?}");
        assert!(metrics.linearity_relative <= 2.0e-15, "{metrics:?}");
        assert!(metrics.rigid_motion_relative <= 2.0e-14, "{metrics:?}");
        assert!(metrics.global_phase_relative <= 2.0e-15, "{metrics:?}");
        assert!(metrics.source_permutation_identical, "{metrics:?}");
    }
}
