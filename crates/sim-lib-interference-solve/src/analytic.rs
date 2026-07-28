//! Independent analytic identities for the reference phasor solver.

use std::f64::consts::PI;

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, SamplingPolicy, SamplingThresholds, ScalarMedium,
    SourceSet, UnitVector3, WorkBudget,
};

use crate::{HostPhasorField, ReferencePhasorSolver, ReferenceSolveError};

const SPEED_METRES_PER_SECOND: f64 = 8.0;
const FREQUENCY_HERTZ: f64 = 2.0;
const WAVELENGTH_METRES: f64 = SPEED_METRES_PER_SECOND / FREQUENCY_HERTZ;
const FIXTURE_EXTENT_METRES: f64 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AnalyticMetrics {
    pub(crate) plane_period: f64,
    pub(crate) point_spreading: f64,
    pub(crate) attenuation: f64,
    pub(crate) cancellation: f64,
    pub(crate) reinforcement: f64,
    pub(crate) standing_node: f64,
    pub(crate) outgoing_crest: f64,
}

impl AnalyticMetrics {
    pub(crate) fn max_error(self) -> f64 {
        [
            self.plane_period,
            self.point_spreading,
            self.attenuation,
            self.cancellation,
            self.reinforcement,
            self.standing_node,
            self.outgoing_crest,
        ]
        .into_iter()
        .fold(0.0, f64::max)
    }
}

pub(crate) fn measure_analytic_fixtures() -> Result<AnalyticMetrics, ReferenceSolveError> {
    Ok(AnalyticMetrics {
        plane_period: plane_period_error()?,
        point_spreading: point_spreading_error()?,
        attenuation: attenuation_error()?,
        cancellation: cancellation_error()?,
        reinforcement: reinforcement_error()?,
        standing_node: standing_node_error()?,
        outgoing_crest: outgoing_crest_time_sign_error()?,
    })
}

fn plane_period_error() -> Result<f64, ReferenceSolveError> {
    let problem = problem(
        0.0,
        vec![forward_plane(
            "period",
            point(-WAVELENGTH_METRES, 0.0, 0.0)?,
            direction(1.0, 0.0, 0.0)?,
            1.0,
            0.375,
        )?],
    )?;
    let field = solve(
        &problem,
        line_plane(
            point(-WAVELENGTH_METRES / 2.0, -0.5, 0.0)?,
            WAVELENGTH_METRES,
            2,
        )?,
    )?;
    relative_complex_error(field.cell(0, 0).unwrap(), field.cell(0, 1).unwrap())
}

fn point_spreading_error() -> Result<f64, ReferenceSolveError> {
    let problem = problem(
        0.0,
        vec![point_source("spreading", point(0.0, 0.0, 0.0)?, 3.0, 0.25)?],
    )?;
    let field = solve(&problem, line_plane(point(0.5, -0.5, 0.0)?, 1.0, 2)?)?;
    let near_scaled = magnitude(field.cell(0, 0).unwrap()) * 1.0;
    let far_scaled = magnitude(field.cell(0, 1).unwrap()) * 2.0;
    relative_scalar_error(near_scaled, far_scaled)
}

fn attenuation_error() -> Result<f64, ReferenceSolveError> {
    let attenuation = 0.35;
    let problem = problem(
        attenuation,
        vec![forward_plane(
            "attenuation",
            point(0.0, 0.0, 0.0)?,
            direction(1.0, 0.0, 0.0)?,
            2.0,
            -0.25,
        )?],
    )?;
    let field = solve(&problem, line_plane(point(0.5, -0.5, 0.0)?, 1.0, 2)?)?;
    let measured_ratio =
        magnitude(field.cell(0, 1).unwrap()) / magnitude(field.cell(0, 0).unwrap());
    let expected_ratio = (-attenuation).exp();
    relative_scalar_error(measured_ratio, expected_ratio)
}

fn cancellation_error() -> Result<f64, ReferenceSolveError> {
    let amplitude = 1.75;
    let problem = problem(
        0.0,
        vec![
            forward_plane(
                "cancel-a",
                point(0.0, 0.0, 0.0)?,
                direction(1.0, 0.0, 0.0)?,
                amplitude,
                0.0,
            )?,
            forward_plane(
                "cancel-b",
                point(0.0, 0.0, 0.0)?,
                direction(1.0, 0.0, 0.0)?,
                amplitude,
                PI,
            )?,
        ],
    )?;
    let field = solve(&problem, point_plane(point(0.5, 0.0, 0.0)?)?)?;
    Ok(magnitude(field.cell(0, 0).unwrap()) / (2.0 * amplitude))
}

fn reinforcement_error() -> Result<f64, ReferenceSolveError> {
    let amplitude = 1.25;
    let problem = problem(
        0.0,
        vec![
            forward_plane(
                "reinforce-a",
                point(0.0, 0.0, 0.0)?,
                direction(1.0, 0.0, 0.0)?,
                amplitude,
                0.5,
            )?,
            forward_plane(
                "reinforce-b",
                point(0.0, 0.0, 0.0)?,
                direction(1.0, 0.0, 0.0)?,
                amplitude,
                0.5,
            )?,
        ],
    )?;
    let field = solve(&problem, point_plane(point(0.75, 0.0, 0.0)?)?)?;
    relative_scalar_error(magnitude(field.cell(0, 0).unwrap()), 2.0 * amplitude)
}

fn standing_node_error() -> Result<f64, ReferenceSolveError> {
    let problem = problem(
        0.0,
        vec![
            forward_plane(
                "standing-forward",
                point(0.0, 0.0, 0.0)?,
                direction(1.0, 0.0, 0.0)?,
                1.0,
                0.0,
            )?,
            forward_plane(
                "standing-reverse",
                point(WAVELENGTH_METRES, 0.0, 0.0)?,
                direction(-1.0, 0.0, 0.0)?,
                1.0,
                0.0,
            )?,
        ],
    )?;
    let field = solve(
        &problem,
        point_plane(point(WAVELENGTH_METRES / 4.0, 0.0, 0.0)?)?,
    )?;
    Ok(magnitude(field.cell(0, 0).unwrap()) / 2.0)
}

pub(crate) fn outgoing_crest_time_sign_error() -> Result<f64, ReferenceSolveError> {
    let problem = problem(
        0.0,
        vec![forward_plane(
            "outgoing-crest",
            point(-WAVELENGTH_METRES / 2.0, 0.0, 0.0)?,
            direction(1.0, 0.0, 0.0)?,
            1.0,
            PI,
        )?],
    )?;
    let elapsed_phase = PI / 4.0;
    let forward = solve(
        &problem,
        point_plane(point(WAVELENGTH_METRES / 8.0, 0.0, 0.0)?)?,
    )?
    .cell(0, 0)
    .unwrap();
    let backward = solve(
        &problem,
        point_plane(point(-WAVELENGTH_METRES / 8.0, 0.0, 0.0)?)?,
    )?
    .cell(0, 0)
    .unwrap();

    let forward_crest = instantaneous_value(forward, elapsed_phase);
    let backward_value = instantaneous_value(backward, elapsed_phase);
    Ok((forward_crest - 1.0).abs().max(backward_value.abs()))
}

fn instantaneous_value(phasor: (f64, f64), angular_time: f64) -> f64 {
    phasor.0 * angular_time.cos() + phasor.1 * angular_time.sin()
}

fn solve(
    problem: &InterferenceProblem,
    plane: SamplingPlane,
) -> Result<HostPhasorField, ReferenceSolveError> {
    ReferencePhasorSolver::new(
        SamplingPolicy::Annotate,
        SamplingThresholds::default(),
        WorkBudget::default(),
    )
    .solve(problem, &plane)
    .map(|(field, _)| field)
}

fn problem(
    attenuation: f64,
    sources: Vec<Emitter>,
) -> Result<InterferenceProblem, ReferenceSolveError> {
    Ok(InterferenceProblem::new(
        Hertz::new(FREQUENCY_HERTZ)?,
        ScalarMedium::new(
            MetresPerSecond::new(SPEED_METRES_PER_SECOND)?,
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

fn point(x: f64, y: f64, z: f64) -> Result<Point3M, ReferenceSolveError> {
    Point3M::from_metres(x, y, z).map_err(ReferenceSolveError::from)
}

fn direction(x: f64, y: f64, z: f64) -> Result<UnitVector3, ReferenceSolveError> {
    UnitVector3::new(x, y, z).map_err(ReferenceSolveError::from)
}

fn point_plane(at: Point3M) -> Result<SamplingPlane, ReferenceSolveError> {
    let [x, y, z] = at.coordinates_metres();
    SamplingPlane::new(
        point(
            x - FIXTURE_EXTENT_METRES / 2.0,
            y - FIXTURE_EXTENT_METRES / 2.0,
            z,
        )?,
        direction(1.0, 0.0, 0.0)?,
        direction(0.0, 1.0, 0.0)?,
        PositiveMetres::new(FIXTURE_EXTENT_METRES)?,
        PositiveMetres::new(FIXTURE_EXTENT_METRES)?,
        1,
        1,
    )
    .map_err(ReferenceSolveError::from)
}

fn line_plane(
    origin: Point3M,
    spacing: f64,
    columns: usize,
) -> Result<SamplingPlane, ReferenceSolveError> {
    SamplingPlane::new(
        origin,
        direction(1.0, 0.0, 0.0)?,
        direction(0.0, 1.0, 0.0)?,
        PositiveMetres::new(spacing * columns as f64)?,
        PositiveMetres::new(1.0)?,
        1,
        columns,
    )
    .map_err(ReferenceSolveError::from)
}

fn magnitude(value: (f64, f64)) -> f64 {
    value.0.hypot(value.1)
}

fn relative_scalar_error(actual: f64, expected: f64) -> Result<f64, ReferenceSolveError> {
    let scale = actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE);
    Ok((actual - expected).abs() / scale)
}

fn relative_complex_error(
    actual: (f64, f64),
    expected: (f64, f64),
) -> Result<f64, ReferenceSolveError> {
    let difference = (actual.0 - expected.0).hypot(actual.1 - expected.1);
    let scale = magnitude(actual)
        .max(magnitude(expected))
        .max(f64::MIN_POSITIVE);
    Ok(difference / scale)
}

#[cfg(test)]
mod tests {
    use super::measure_analytic_fixtures;

    #[test]
    fn plane_point_attenuation_and_interference_identities_hold() {
        let metrics = measure_analytic_fixtures().unwrap();

        assert!(metrics.plane_period <= 2.0e-15, "{metrics:?}");
        assert!(metrics.point_spreading <= 2.0e-15, "{metrics:?}");
        assert!(metrics.attenuation <= 2.0e-15, "{metrics:?}");
        assert!(metrics.cancellation <= 2.0e-15, "{metrics:?}");
        assert!(metrics.reinforcement <= 2.0e-15, "{metrics:?}");
        assert!(metrics.standing_node <= 2.0e-15, "{metrics:?}");
        assert!(metrics.outgoing_crest <= 2.0e-15, "{metrics:?}");
        assert!(metrics.max_error() <= 2.0e-15, "{metrics:?}");
    }

    #[test]
    fn outgoing_crest_moves_in_the_positive_propagation_direction() {
        let error = super::outgoing_crest_time_sign_error().unwrap();

        assert!(error <= 2.0e-15, "outgoing crest error {error:e}");
    }
}
