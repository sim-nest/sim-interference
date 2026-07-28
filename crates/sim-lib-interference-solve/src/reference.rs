//! Configuration, traversal, and evidence for the CPU reference solver.

use sim_lib_interference_core::{
    Emitter, InterferenceError, InterferenceProblem, Point3M, RequestPreflight, SamplingPlane,
    SamplingPolicy, SamplingThresholds, WorkBudget, contribution_at,
};

use crate::{HostPhasorField, ReferenceSolveError, complex::Complex64};

/// Immutable evidence retained for one complete reference solve.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveEvidence {
    preflight: RequestPreflight,
    completed_cells: u64,
    completed_emitter_evaluations: u64,
}

impl SolveEvidence {
    /// Returns the sampling policy, certificate, and work admission record.
    pub fn preflight(&self) -> RequestPreflight {
        self.preflight
    }

    /// Returns the number of row-major output cells completed.
    pub fn completed_cells(&self) -> u64 {
        self.completed_cells
    }

    /// Returns the number of canonical source evaluations completed.
    pub fn completed_emitter_evaluations(&self) -> u64 {
        self.completed_emitter_evaluations
    }
}

/// Configuration for deterministic CPU `f64` reference solving.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferencePhasorSolver {
    sampling_policy: SamplingPolicy,
    sampling_thresholds: SamplingThresholds,
    work_budget: WorkBudget,
}

impl ReferencePhasorSolver {
    /// Constructs a solver from explicit admission policy.
    pub fn new(
        sampling_policy: SamplingPolicy,
        sampling_thresholds: SamplingThresholds,
        work_budget: WorkBudget,
    ) -> Self {
        Self {
            sampling_policy,
            sampling_thresholds,
            work_budget,
        }
    }

    /// Returns the configured sampling policy.
    pub fn sampling_policy(self) -> SamplingPolicy {
        self.sampling_policy
    }

    /// Returns the configured sampling thresholds.
    pub fn sampling_thresholds(self) -> SamplingThresholds {
        self.sampling_thresholds
    }

    /// Returns the configured work budget.
    pub fn work_budget(self) -> WorkBudget {
        self.work_budget
    }

    /// Solves one complete field or returns no field.
    ///
    /// Work and sampling admission run first, followed by an allocation-free
    /// row-major geometry pass. Component storage is reserved only after those
    /// checks pass.
    pub fn solve(
        self,
        problem: &InterferenceProblem,
        plane: &SamplingPlane,
    ) -> Result<(HostPhasorField, SolveEvidence), ReferenceSolveError> {
        let preflight = RequestPreflight::admit(
            problem,
            plane,
            self.sampling_policy,
            self.sampling_thresholds,
            self.work_budget,
        )
        .map_err(ReferenceSolveError::from)?;
        preflight_geometry(problem, plane)?;
        let field = evaluate_field(problem, plane)?;
        let evidence = SolveEvidence {
            preflight,
            completed_cells: preflight.work_estimate.cells,
            completed_emitter_evaluations: preflight.work_estimate.emitter_evaluations,
        };
        Ok((field, evidence))
    }
}

impl Default for ReferencePhasorSolver {
    fn default() -> Self {
        Self::new(
            SamplingPolicy::Strict,
            SamplingThresholds::default(),
            WorkBudget::default(),
        )
    }
}

fn preflight_geometry(
    problem: &InterferenceProblem,
    plane: &SamplingPlane,
) -> Result<(), ReferenceSolveError> {
    for row in 0..plane.rows() {
        for column in 0..plane.columns() {
            let at =
                plane
                    .point_at(row, column)
                    .map_err(|cause| ReferenceSolveError::CellGeometry {
                        row,
                        column,
                        cause: Box::new(cause),
                    })?;
            for source in &problem.sources {
                validate_source_geometry(problem, source, at).map_err(|cause| {
                    ReferenceSolveError::SourceAtCell {
                        source_id: source.id().to_owned(),
                        row,
                        column,
                        cause: Box::new(cause),
                    }
                })?;
            }
        }
    }
    Ok(())
}

fn validate_source_geometry(
    problem: &InterferenceProblem,
    source: &Emitter,
    at: Point3M,
) -> Result<(), InterferenceError> {
    match source {
        Emitter::Point { id, position, .. } => {
            let distance = at.distance_to(*position);
            if !distance.is_finite() {
                return Err(InterferenceError::NonFinitePropagation {
                    source_id: id.clone(),
                    name: "point-distance-metres",
                    value: distance,
                });
            }
            if distance <= problem.singularity_radius.get() {
                return Err(InterferenceError::SingularPointSample {
                    source_id: id.clone(),
                    distance_metres: distance,
                    singularity_radius_metres: problem.singularity_radius.get(),
                });
            }
        }
        Emitter::ForwardPlane {
            id,
            through,
            direction,
            ..
        } => {
            let signed_distance = direction.signed_distance_metres(*through, at);
            if !signed_distance.is_finite() {
                return Err(InterferenceError::NonFinitePropagation {
                    source_id: id.clone(),
                    name: "plane-signed-distance-metres",
                    value: signed_distance,
                });
            }
            if signed_distance < 0.0 {
                return Err(InterferenceError::BehindForwardPlane {
                    source_id: id.clone(),
                    signed_distance_metres: signed_distance,
                });
            }
        }
    }
    Ok(())
}

fn evaluate_field(
    problem: &InterferenceProblem,
    plane: &SamplingPlane,
) -> Result<HostPhasorField, ReferenceSolveError> {
    let mut field = HostPhasorField::try_zeroed(plane.rows(), plane.columns())?;
    for row in 0..plane.rows() {
        for column in 0..plane.columns() {
            let at =
                plane
                    .point_at(row, column)
                    .map_err(|cause| ReferenceSolveError::CellGeometry {
                        row,
                        column,
                        cause: Box::new(cause),
                    })?;
            let value = accumulate_cell(problem, at, row, column)?;
            let index = row * plane.columns() + column;
            field.set_index(index, value);
        }
    }
    Ok(field)
}

fn accumulate_cell(
    problem: &InterferenceProblem,
    at: Point3M,
    row: usize,
    column: usize,
) -> Result<Complex64, ReferenceSolveError> {
    let mut real_sum = 0.0;
    let mut real_correction = 0.0;
    let mut imaginary_sum = 0.0;
    let mut imaginary_correction = 0.0;

    for source in &problem.sources {
        let source_id = source.id();
        let (real, imaginary) = contribution_at(problem, source, at).map_err(|cause| {
            ReferenceSolveError::SourceAtCell {
                source_id: source_id.to_owned(),
                row,
                column,
                cause: Box::new(cause),
            }
        })?;
        compensated_add(&mut real_sum, &mut real_correction, real);
        require_finite_accumulation(source_id, row, column, "real", real_sum, real_correction)?;
        compensated_add(&mut imaginary_sum, &mut imaginary_correction, imaginary);
        require_finite_accumulation(
            source_id,
            row,
            column,
            "imaginary",
            imaginary_sum,
            imaginary_correction,
        )?;
    }

    Ok(Complex64::new(
        real_sum + real_correction,
        imaginary_sum + imaginary_correction,
    ))
}

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let next = *sum + value;
    *correction += if sum.abs() >= value.abs() {
        (*sum - next) + value
    } else {
        (value - next) + *sum
    };
    *sum = next;
}

fn require_finite_accumulation(
    source_id: &str,
    row: usize,
    column: usize,
    component: &'static str,
    sum: f64,
    correction: f64,
) -> Result<(), ReferenceSolveError> {
    let value = sum + correction;
    if sum.is_finite() && correction.is_finite() && value.is_finite() {
        Ok(())
    } else {
        Err(ReferenceSolveError::NonFiniteAccumulation {
            source_id: source_id.to_owned(),
            row,
            column,
            component,
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use sim_lib_interference_core::{
        Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre,
        Point3M, PositiveMetres, Radians, SamplingPlane, SamplingPolicy, SamplingThresholds,
        ScalarMedium, SourceSet, UnitVector3, WorkBudget,
    };

    use super::{ReferencePhasorSolver, evaluate_field};

    fn point(x: f64, y: f64, z: f64) -> Point3M {
        Point3M::from_metres(x, y, z).unwrap()
    }

    fn plane(rows: usize, columns: usize) -> SamplingPlane {
        SamplingPlane::new(
            point(0.0, 0.0, 0.0),
            UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
            UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
            PositiveMetres::new(1.0).unwrap(),
            PositiveMetres::new(1.0).unwrap(),
            rows,
            columns,
        )
        .unwrap()
    }

    fn problem(sources: Vec<Emitter>) -> InterferenceProblem {
        InterferenceProblem::new(
            Hertz::new(1.0).unwrap(),
            ScalarMedium::new(
                MetresPerSecond::new(100.0).unwrap(),
                NepersPerMetre::new(0.0).unwrap(),
            ),
            SourceSet::new(sources).unwrap(),
            PositiveMetres::new(0.001).unwrap(),
        )
    }

    fn plane_source(id: &str, amplitude: f64) -> Emitter {
        Emitter::ForwardPlane {
            id: id.to_owned(),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
            amplitude: FieldAmplitude::new(amplitude).unwrap(),
            phase: Radians::new(0.0).unwrap(),
        }
    }

    #[test]
    fn solver_configuration_is_explicit_and_immutable() {
        let thresholds = SamplingThresholds::new(12.0, 6.0, 0.025, 0.075).unwrap();
        let budget = WorkBudget {
            max_cells: 100,
            max_emitter_evaluations: 200,
            max_host_bytes: 3_200,
            max_result_bytes: 1_600,
            max_certificate_stencil_work: 700,
        };
        let solver = ReferencePhasorSolver::new(SamplingPolicy::Annotate, thresholds, budget);

        assert_eq!(solver.sampling_policy(), SamplingPolicy::Annotate);
        assert_eq!(solver.sampling_thresholds(), thresholds);
        assert_eq!(solver.work_budget(), budget);
    }

    #[test]
    fn row_major_cells_use_canonical_sources_and_neumaier_components() {
        let problem = problem(vec![
            plane_source("c-small", 1.0),
            plane_source("a-large", 1.0e16),
            plane_source("b-small", 1.0),
        ]);
        assert_eq!(
            problem.sources.iter().map(Emitter::id).collect::<Vec<_>>(),
            ["a-large", "b-small", "c-small"]
        );

        let field = evaluate_field(&problem, &plane(2, 3)).unwrap();
        assert_eq!(field.real(), &[1.0e16 + 2.0; 6]);
        assert_eq!(field.imaginary(), &[0.0; 6]);
        assert_eq!(field.cell(0, 2), Some((field.real()[2], 0.0)));
        assert_eq!(field.cell(1, 0), Some((field.real()[3], 0.0)));
    }

    #[test]
    fn solve_retains_complete_sampling_and_work_preflight() {
        let problem = problem(vec![plane_source("plane", 2.0)]);
        let (field, evidence) = ReferencePhasorSolver::default()
            .solve(&problem, &plane(2, 3))
            .unwrap();

        assert_eq!(field.len(), 6);
        assert_eq!(evidence.completed_cells(), 6);
        assert_eq!(evidence.completed_emitter_evaluations(), 6);
        assert_eq!(evidence.preflight().work_estimate.cells, 6);
        assert_eq!(evidence.preflight().work_estimate.host_bytes, 96);
        assert_eq!(evidence.preflight().sampling_policy, SamplingPolicy::Strict);
    }

    #[test]
    fn request_and_cell_geometry_fail_before_field_construction() {
        let problem = problem(vec![plane_source("plane", 1.0)]);
        let no_cells = WorkBudget {
            max_cells: 0,
            ..WorkBudget::default()
        };
        let budget_error = ReferencePhasorSolver::new(
            SamplingPolicy::Annotate,
            SamplingThresholds::default(),
            no_cells,
        )
        .solve(&problem, &plane(1, 1))
        .unwrap_err();
        assert!(matches!(
            budget_error,
            crate::ReferenceSolveError::Request { cause }
                if matches!(*cause, sim_lib_interference_core::InterferenceError::WorkBudgetExceeded { .. })
        ));

        let extreme_plane = SamplingPlane::new(
            point(f64::MAX, 0.0, 0.0),
            UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
            UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
            PositiveMetres::new(f64::MAX).unwrap(),
            PositiveMetres::new(1.0).unwrap(),
            1,
            1,
        )
        .unwrap();
        let geometry_error = ReferencePhasorSolver::new(
            SamplingPolicy::Annotate,
            SamplingThresholds::default(),
            WorkBudget::default(),
        )
        .solve(&problem, &extreme_plane)
        .unwrap_err();
        assert!(matches!(
            geometry_error,
            crate::ReferenceSolveError::CellGeometry {
                row: 0,
                column: 0,
                ..
            }
        ));
    }

    #[test]
    fn singular_and_behind_samples_name_source_and_cell() {
        let singular_problem = InterferenceProblem::new(
            Hertz::new(1.0).unwrap(),
            ScalarMedium::new(
                MetresPerSecond::new(100.0).unwrap(),
                NepersPerMetre::new(0.0).unwrap(),
            ),
            SourceSet::new(vec![Emitter::Point {
                id: "near-point".to_owned(),
                position: point(0.5, 0.5, 0.01),
                amplitude_at_reference: FieldAmplitude::new(1.0).unwrap(),
                phase: Radians::new(0.0).unwrap(),
            }])
            .unwrap(),
            PositiveMetres::new(0.02).unwrap(),
        );
        let solver = ReferencePhasorSolver::new(
            SamplingPolicy::Annotate,
            SamplingThresholds::default(),
            WorkBudget::default(),
        );
        let singular_error = solver.solve(&singular_problem, &plane(1, 1)).unwrap_err();
        assert!(matches!(
            singular_error,
            crate::ReferenceSolveError::SourceAtCell {
                source_id,
                row: 0,
                column: 0,
                cause,
            } if source_id == "near-point"
                && matches!(*cause, sim_lib_interference_core::InterferenceError::SingularPointSample { .. })
        ));

        let behind_problem = problem(vec![Emitter::ForwardPlane {
            id: "forward-only".to_owned(),
            through: point(0.0, 0.0, 1.0),
            direction: UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
            amplitude: FieldAmplitude::new(1.0).unwrap(),
            phase: Radians::new(0.0).unwrap(),
        }]);
        let behind_error = solver.solve(&behind_problem, &plane(2, 2)).unwrap_err();
        assert!(matches!(
            behind_error,
            crate::ReferenceSolveError::SourceAtCell {
                source_id,
                row: 0,
                column: 0,
                cause,
            } if source_id == "forward-only"
                && matches!(*cause, sim_lib_interference_core::InterferenceError::BehindForwardPlane { .. })
        ));
    }
}
