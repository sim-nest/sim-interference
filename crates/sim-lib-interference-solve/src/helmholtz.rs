//! Seven-point Helmholtz residual convergence on parallel solved planes.

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, SamplingPolicy, SamplingThresholds, ScalarMedium,
    SourceSet, UnitVector3, WorkBudget,
};

use crate::{HostPhasorField, ReferencePhasorSolver, ReferenceSolveError};

pub(crate) const MIN_OBSERVED_ORDER: f64 = 1.8;
pub(crate) const MAX_OBSERVED_ORDER: f64 = 2.2;

const COARSE_POINTS_PER_AXIS: usize = 9;
const COARSE_SPACING_METRES: f64 = 0.125;
const LOWER_COORDINATE_METRES: f64 = -0.5;
const CENTRE_Z_METRES: f64 = 0.0;
const FIXTURE_FREQUENCY_HERTZ: f64 = 1.0;
const FIXTURE_SPEED_METRES_PER_SECOND: f64 = 4.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HelmholtzMeasurement {
    pub(crate) coarse_residual_rms: f64,
    pub(crate) fine_residual_rms: f64,
    pub(crate) observed_order: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NamedHelmholtzMeasurement {
    pub(crate) name: &'static str,
    pub(crate) measurement: HelmholtzMeasurement,
}

pub(crate) fn measure_helmholtz_fixture_matrix()
-> Result<Vec<NamedHelmholtzMeasurement>, ReferenceSolveError> {
    [
        ("point", point_problem()?),
        ("plane", plane_problem()?),
        ("mixed", mixed_problem()?),
        ("attenuating", attenuating_problem()?),
        ("multi-source", multi_source_problem()?),
    ]
    .into_iter()
    .map(|(name, problem)| {
        Ok(NamedHelmholtzMeasurement {
            name,
            measurement: measure_helmholtz_convergence(&problem)?,
        })
    })
    .collect()
}

fn measure_helmholtz_convergence(
    problem: &InterferenceProblem,
) -> Result<HelmholtzMeasurement, ReferenceSolveError> {
    let coarse = solve_stencil(problem, COARSE_SPACING_METRES, COARSE_POINTS_PER_AXIS)?;
    let fine = solve_stencil(
        problem,
        COARSE_SPACING_METRES / 2.0,
        2 * (COARSE_POINTS_PER_AXIS - 1) + 1,
    )?;
    let wave_number = problem.wavenumber();
    let k_real = wave_number.real_radians_per_metre();
    let k_imaginary = wave_number.imaginary_nepers_per_metre();
    let k_squared = (
        k_real * k_real - k_imaginary * k_imaginary,
        2.0 * k_real * k_imaginary,
    );
    let coarse_residual_rms =
        common_point_residual_rms(&coarse, COARSE_SPACING_METRES, k_squared, 1);
    let fine_residual_rms =
        common_point_residual_rms(&fine, COARSE_SPACING_METRES / 2.0, k_squared, 2);
    let observed_order = (coarse_residual_rms / fine_residual_rms).log2();

    Ok(HelmholtzMeasurement {
        coarse_residual_rms,
        fine_residual_rms,
        observed_order,
    })
}

struct StencilFields {
    below: HostPhasorField,
    centre: HostPhasorField,
    above: HostPhasorField,
}

fn solve_stencil(
    problem: &InterferenceProblem,
    spacing: f64,
    points_per_axis: usize,
) -> Result<StencilFields, ReferenceSolveError> {
    let solver = ReferencePhasorSolver::new(
        SamplingPolicy::Annotate,
        SamplingThresholds::default(),
        WorkBudget::default(),
    );
    let solve_at_z = |z| {
        solver
            .solve(problem, &parallel_plane(z, spacing, points_per_axis)?)
            .map(|(field, _)| field)
    };

    Ok(StencilFields {
        below: solve_at_z(CENTRE_Z_METRES - spacing)?,
        centre: solve_at_z(CENTRE_Z_METRES)?,
        above: solve_at_z(CENTRE_Z_METRES + spacing)?,
    })
}

fn parallel_plane(
    z: f64,
    spacing: f64,
    points_per_axis: usize,
) -> Result<SamplingPlane, ReferenceSolveError> {
    SamplingPlane::new(
        point(
            LOWER_COORDINATE_METRES - spacing / 2.0,
            LOWER_COORDINATE_METRES - spacing / 2.0,
            z,
        )?,
        direction(1.0, 0.0, 0.0)?,
        direction(0.0, 1.0, 0.0)?,
        PositiveMetres::new(spacing * points_per_axis as f64)?,
        PositiveMetres::new(spacing * points_per_axis as f64)?,
        points_per_axis,
        points_per_axis,
    )
    .map_err(ReferenceSolveError::from)
}

fn common_point_residual_rms(
    fields: &StencilFields,
    spacing: f64,
    k_squared: (f64, f64),
    fine_index_scale: usize,
) -> f64 {
    let coarse_interior = 1..(COARSE_POINTS_PER_AXIS - 1);
    let mut squared_norm_sum = 0.0;
    let mut count = 0_u64;

    for coarse_row in coarse_interior.clone() {
        for coarse_column in coarse_interior.clone() {
            let row = coarse_row * fine_index_scale;
            let column = coarse_column * fine_index_scale;
            let residual = seven_point_residual(fields, row, column, spacing, k_squared);
            squared_norm_sum += residual.0 * residual.0 + residual.1 * residual.1;
            count += 1;
        }
    }

    (squared_norm_sum / count as f64).sqrt()
}

fn seven_point_residual(
    fields: &StencilFields,
    row: usize,
    column: usize,
    spacing: f64,
    k_squared: (f64, f64),
) -> (f64, f64) {
    let centre = fields.centre.cell(row, column).unwrap();
    let neighbours = [
        fields.centre.cell(row, column - 1).unwrap(),
        fields.centre.cell(row, column + 1).unwrap(),
        fields.centre.cell(row - 1, column).unwrap(),
        fields.centre.cell(row + 1, column).unwrap(),
        fields.below.cell(row, column).unwrap(),
        fields.above.cell(row, column).unwrap(),
    ];
    let neighbour_sum = neighbours
        .into_iter()
        .fold((0.0, 0.0), |sum, value| (sum.0 + value.0, sum.1 + value.1));
    let inverse_spacing_squared = 1.0 / (spacing * spacing);
    let laplacian = (
        (neighbour_sum.0 - 6.0 * centre.0) * inverse_spacing_squared,
        (neighbour_sum.1 - 6.0 * centre.1) * inverse_spacing_squared,
    );
    let k_squared_field = (
        k_squared.0 * centre.0 - k_squared.1 * centre.1,
        k_squared.0 * centre.1 + k_squared.1 * centre.0,
    );
    (
        laplacian.0 + k_squared_field.0,
        laplacian.1 + k_squared_field.1,
    )
}

fn point_problem() -> Result<InterferenceProblem, ReferenceSolveError> {
    problem(
        0.0,
        vec![point_source("point", point(-0.2, 0.15, -2.5)?, 1.25, 0.3)?],
    )
}

fn plane_problem() -> Result<InterferenceProblem, ReferenceSolveError> {
    problem(
        0.0,
        vec![forward_plane(
            "plane",
            point(0.0, 0.0, -1.0)?,
            direction(0.0, 0.0, 1.0)?,
            0.6,
            -0.4,
        )?],
    )
}

fn mixed_problem() -> Result<InterferenceProblem, ReferenceSolveError> {
    problem(
        0.025,
        vec![
            point_source("mixed-point", point(-0.2, 0.15, -2.5)?, 1.25, 0.3)?,
            forward_plane(
                "mixed-plane",
                point(0.0, 0.0, -1.0)?,
                direction(0.0, 0.0, 1.0)?,
                0.6,
                -0.4,
            )?,
        ],
    )
}

fn attenuating_problem() -> Result<InterferenceProblem, ReferenceSolveError> {
    problem(
        0.12,
        vec![
            point_source("attenuating-point", point(0.3, -0.25, -3.0)?, 1.5, -0.2)?,
            forward_plane(
                "attenuating-plane",
                point(0.0, 0.0, -1.25)?,
                direction(0.0, 0.0, 1.0)?,
                0.4,
                0.55,
            )?,
        ],
    )
}

fn multi_source_problem() -> Result<InterferenceProblem, ReferenceSolveError> {
    problem(
        0.04,
        vec![
            point_source("multi-point-a", point(-0.75, 0.4, -2.75)?, 1.1, 0.15)?,
            point_source("multi-point-b", point(0.65, -0.5, -3.25)?, 0.8, -0.45)?,
            forward_plane(
                "multi-plane-x",
                point(-1.5, 0.0, 0.0)?,
                direction(1.0, 0.0, 0.0)?,
                0.35,
                0.7,
            )?,
            forward_plane(
                "multi-plane-z",
                point(0.0, 0.0, -1.5)?,
                direction(0.0, 0.0, 1.0)?,
                0.5,
                -0.3,
            )?,
        ],
    )
}

fn problem(
    attenuation: f64,
    sources: Vec<Emitter>,
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

fn point(x: f64, y: f64, z: f64) -> Result<Point3M, ReferenceSolveError> {
    Point3M::from_metres(x, y, z).map_err(ReferenceSolveError::from)
}

fn direction(x: f64, y: f64, z: f64) -> Result<UnitVector3, ReferenceSolveError> {
    UnitVector3::new(x, y, z).map_err(ReferenceSolveError::from)
}

#[cfg(test)]
mod tests {
    use super::{
        COARSE_POINTS_PER_AXIS, COARSE_SPACING_METRES, LOWER_COORDINATE_METRES, MAX_OBSERVED_ORDER,
        MIN_OBSERVED_ORDER, measure_helmholtz_fixture_matrix, parallel_plane,
    };

    #[test]
    fn coarse_and_fine_stencils_share_the_compared_physical_points() {
        let coarse = parallel_plane(0.0, COARSE_SPACING_METRES, COARSE_POINTS_PER_AXIS).unwrap();
        let fine = parallel_plane(
            0.0,
            COARSE_SPACING_METRES / 2.0,
            2 * (COARSE_POINTS_PER_AXIS - 1) + 1,
        )
        .unwrap();

        for row in 1..(COARSE_POINTS_PER_AXIS - 1) {
            for column in 1..(COARSE_POINTS_PER_AXIS - 1) {
                assert_eq!(
                    coarse.point_at(row, column).unwrap(),
                    fine.point_at(2 * row, 2 * column).unwrap()
                );
            }
        }
        assert_eq!(
            coarse.point_at(0, 0).unwrap().coordinates_metres(),
            [LOWER_COORDINATE_METRES, LOWER_COORDINATE_METRES, 0.0]
        );
    }

    #[test]
    fn seven_point_residual_converges_at_second_order() {
        let measurement = measure_helmholtz_fixture_matrix()
            .unwrap()
            .into_iter()
            .find(|measurement| measurement.name == "mixed")
            .unwrap()
            .measurement;
        eprintln!(
            "helmholtz coarse_rms={:.12e} fine_rms={:.12e} observed_order={:.12}",
            measurement.coarse_residual_rms,
            measurement.fine_residual_rms,
            measurement.observed_order
        );

        assert!(
            (MIN_OBSERVED_ORDER..=MAX_OBSERVED_ORDER).contains(&measurement.observed_order),
            "{measurement:?}"
        );
    }

    #[test]
    fn fixture_matrix_converges_at_second_order() {
        let measurements = measure_helmholtz_fixture_matrix().unwrap();
        assert_eq!(
            measurements
                .iter()
                .map(|measurement| measurement.name)
                .collect::<Vec<_>>(),
            ["point", "plane", "mixed", "attenuating", "multi-source"]
        );

        for named in measurements {
            eprintln!(
                "fixture={} coarse_rms={:.12e} fine_rms={:.12e} observed_order={:.12}",
                named.name,
                named.measurement.coarse_residual_rms,
                named.measurement.fine_residual_rms,
                named.measurement.observed_order
            );
            assert!(
                (MIN_OBSERVED_ORDER..=MAX_OBSERVED_ORDER)
                    .contains(&named.measurement.observed_order),
                "{named:?}"
            );
        }
    }
}
