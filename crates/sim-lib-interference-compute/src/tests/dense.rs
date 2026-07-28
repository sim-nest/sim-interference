use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, SamplingPolicy, SamplingThresholds, ScalarMedium,
    SourceSet, UnitVector3, WorkBudget,
};
use sim_lib_interference_solve::ReferencePhasorSolver;

use crate::{
    DifferentialTolerances, PhaseBudget, PreflightCheck, TileProfile, compare_dense_to_reference,
    solve_dense_f32_cpu,
};

// conformance: dense f32 CPU execution follows the exact normalized lowering

fn point(x: f64, y: f64, z: f64) -> Point3M {
    Point3M::from_metres(x, y, z).unwrap()
}

fn plane(distance_m: f64, rows: usize, columns: usize) -> SamplingPlane {
    SamplingPlane::new(
        point(distance_m, -0.1, -0.1),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
        PositiveMetres::new(0.2).unwrap(),
        PositiveMetres::new(0.2).unwrap(),
        rows,
        columns,
    )
    .unwrap()
}

fn problem() -> InterferenceProblem {
    problem_from_sources(
        0.002,
        vec![
            Emitter::Point {
                id: "left".to_owned(),
                position: point(0.0, -0.03, 0.0),
                amplitude_at_reference: FieldAmplitude::new(1.0).unwrap(),
                phase: Radians::new(0.2).unwrap(),
            },
            Emitter::Point {
                id: "right".to_owned(),
                position: point(0.0, 0.04, 0.01),
                amplitude_at_reference: FieldAmplitude::new(0.8).unwrap(),
                phase: Radians::new(-0.4).unwrap(),
            },
        ],
    )
}

fn problem_from_sources(attenuation_np_m: f64, sources: Vec<Emitter>) -> InterferenceProblem {
    InterferenceProblem::new(
        Hertz::new(343.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(343.0).unwrap(),
            NepersPerMetre::new(attenuation_np_m).unwrap(),
        ),
        SourceSet::new(sources).unwrap(),
        PositiveMetres::new(0.001).unwrap(),
    )
}

fn test_cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::testing::eager_cx();
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_float::F32NumbersLib::new())
        .unwrap();
    cx
}

fn reference(
    problem: &InterferenceProblem,
    plane: &SamplingPlane,
) -> sim_lib_interference_solve::HostPhasorField {
    ReferencePhasorSolver::new(
        SamplingPolicy::Annotate,
        SamplingThresholds::default(),
        WorkBudget::default(),
    )
    .solve(problem, plane)
    .unwrap()
    .0
}

#[test]
fn exact_lowering_executes_through_dense_f32_cpu_and_assembles_tiles() {
    let problem = problem();
    let plane = plane(100.0, 5, 7);
    let profile = TileProfile {
        max_elements_per_tile: 6,
        max_segment_bytes: 8,
        max_segments_per_tensor: 3,
        max_tensor_bytes: 24,
        max_result_bytes: 8 * 35,
        ..TileProfile::default()
    };
    let mut cx = test_cx();
    let dense =
        solve_dense_f32_cpu(&mut cx, &problem, plane, PhaseBudget::default(), profile).unwrap();

    assert_eq!(dense.rows(), 5);
    assert_eq!(dense.columns(), 7);
    assert_eq!(dense.real().len(), 35);
    assert_eq!(dense.imaginary().len(), 35);
    assert!(
        dense
            .real()
            .iter()
            .chain(dense.imaginary())
            .all(|v| v.is_finite())
    );
    assert_eq!(
        dense.evidence().executor().to_string(),
        "tensor/executor/cpu"
    );
    assert!(dense.evidence().tiles() > 1);
    assert_eq!(dense.evidence().max_segments_per_tensor(), 3);
    assert!(dense.evidence().submissions() >= dense.evidence().tiles() * 3);
    assert!(dense.evidence().observed_max_abs_psi_rad() <= dense.evidence().phase_limit_rad());
    assert!(dense.evidence().predicted_max_abs_psi_rad() <= dense.evidence().phase_limit_rad());

    let report = compare_dense_to_reference(
        &reference(&problem, &plane),
        &dense,
        DifferentialTolerances::default(),
    )
    .unwrap();
    assert!(report.passed(), "{report:?}");
}

#[test]
fn fixed_tolerances_compare_every_published_quantity_above_phase_floor() {
    let problem = problem();
    let plane = plane(10.0, 4, 6);
    let mut cx = test_cx();
    let dense = solve_dense_f32_cpu(
        &mut cx,
        &problem,
        plane,
        PhaseBudget::default(),
        TileProfile::default(),
    )
    .unwrap();
    let tolerances = DifferentialTolerances::default();
    assert_eq!(tolerances.component.absolute, 2.0e-5);
    assert_eq!(tolerances.component.relative, 2.0e-4);
    assert_eq!(tolerances.amplitude.absolute, 2.0e-5);
    assert_eq!(tolerances.amplitude.relative, 2.0e-4);
    assert_eq!(tolerances.phase.absolute, 3.0e-4);
    assert_eq!(tolerances.phase.relative, 1.0e-4);
    assert_eq!(tolerances.magnitude_squared.absolute, 4.0e-5);
    assert_eq!(tolerances.magnitude_squared.relative, 4.0e-4);
    assert_eq!(tolerances.phase_amplitude_floor, 1.0e-5);

    let report = compare_dense_to_reference(&reference(&problem, &plane), &dense, tolerances)
        .expect("finite equal-shape fields compare");
    assert_eq!(report.phase_cells, plane.cell_count());
    assert!(report.real.passed(), "{report:?}");
    assert!(report.imaginary.passed(), "{report:?}");
    assert!(report.amplitude.passed(), "{report:?}");
    assert!(report.phase.unwrap().passed(), "{report:?}");
    assert!(report.magnitude_squared.passed(), "{report:?}");
    assert!(report.passed(), "{report:?}");
}

#[test]
fn fixture_matrix_covers_nodes_attenuation_edges_segments_permutation_and_refusals() {
    let attenuated = problem();
    let sample_plane = plane(100.0, 5, 7);
    let segmented_profile = TileProfile {
        max_elements_per_tile: 6,
        max_segment_bytes: 8,
        max_segments_per_tensor: 3,
        max_tensor_bytes: 24,
        max_result_bytes: 8 * 35,
        ..TileProfile::default()
    };
    let mut cx = test_cx();
    let attenuated_dense = solve_dense_f32_cpu(
        &mut cx,
        &attenuated,
        sample_plane,
        PhaseBudget::default(),
        segmented_profile,
    )
    .unwrap();
    let attenuated_report = compare_dense_to_reference(
        &reference(&attenuated, &sample_plane),
        &attenuated_dense,
        DifferentialTolerances::default(),
    )
    .unwrap();
    assert!(attenuated_report.passed(), "{attenuated_report:?}");
    assert!(attenuated_dense.evidence().tiles() > 1);
    assert_eq!(attenuated_dense.evidence().max_segments_per_tensor(), 3);
    assert!(
        attenuated_dense.evidence().tiles().saturating_mul(6) > sample_plane.cell_count(),
        "the final tile must be an awkward clipped edge"
    );

    let node_sources = vec![
        Emitter::ForwardPlane {
            id: "phase-0".to_owned(),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
            amplitude: FieldAmplitude::new(1.0).unwrap(),
            phase: Radians::new(0.0).unwrap(),
        },
        Emitter::ForwardPlane {
            id: "phase-pi".to_owned(),
            through: point(0.0, 0.0, 0.0),
            direction: UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
            amplitude: FieldAmplitude::new(1.0).unwrap(),
            phase: Radians::new(std::f64::consts::PI).unwrap(),
        },
    ];
    let node_problem = problem_from_sources(0.0, node_sources.clone());
    let node_plane = plane(10.0, 3, 5);
    let node_dense = solve_dense_f32_cpu(
        &mut cx,
        &node_problem,
        node_plane,
        PhaseBudget::default(),
        TileProfile::default(),
    )
    .unwrap();
    let node_report = compare_dense_to_reference(
        &reference(&node_problem, &node_plane),
        &node_dense,
        DifferentialTolerances::default(),
    )
    .unwrap();
    assert!(node_report.passed(), "{node_report:?}");
    assert_eq!(node_report.phase_cells, 0);
    assert_eq!(node_report.phase, None);

    let permuted_problem =
        problem_from_sources(0.0, node_sources.into_iter().rev().collect::<Vec<_>>());
    let permuted_dense = solve_dense_f32_cpu(
        &mut cx,
        &permuted_problem,
        node_plane,
        PhaseBudget::default(),
        TileProfile::default(),
    )
    .unwrap();
    assert_eq!(node_dense.real(), permuted_dense.real());
    assert_eq!(node_dense.imaginary(), permuted_dense.imaginary());

    assert!(FieldAmplitude::new(f64::NAN).is_err());
    assert!(Point3M::from_metres(f64::INFINITY, 0.0, 0.0).is_err());
    let invalid_tolerances = DifferentialTolerances {
        phase_amplitude_floor: f64::NAN,
        ..DifferentialTolerances::default()
    };
    assert!(
        compare_dense_to_reference(
            &reference(&node_problem, &node_plane),
            &node_dense,
            invalid_tolerances,
        )
        .unwrap_err()
        .detail()
        .contains("phase amplitude floor")
    );

    let behind_problem = problem_from_sources(
        0.0,
        vec![Emitter::ForwardPlane {
            id: "forward-only".to_owned(),
            through: point(11.0, 0.0, 0.0),
            direction: UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
            amplitude: FieldAmplitude::new(1.0).unwrap(),
            phase: Radians::new(0.0).unwrap(),
        }],
    );
    let error = solve_dense_f32_cpu(
        &mut cx,
        &behind_problem,
        node_plane,
        PhaseBudget::default(),
        TileProfile::default(),
    )
    .unwrap_err();
    assert_eq!(error.check(), PreflightCheck::ForwardPlane);
}

#[test]
fn normalized_phase_stays_bounded_while_naive_absolute_f32_grows() {
    let direction = UnitVector3::new(1.0, 0.125, 0.0).unwrap();
    let source_phase = Radians::new(0.37).unwrap();
    let sweep_problem = problem_from_sources(
        0.0,
        vec![Emitter::ForwardPlane {
            id: "tilted".to_owned(),
            through: point(0.0, 0.0, 0.0),
            direction,
            amplitude: FieldAmplitude::new(1.0).unwrap(),
            phase: source_phase,
        }],
    );
    let mut cx = test_cx();
    let mut rows = Vec::new();

    println!(
        "distance_m max_abs_psi normalized_phase_error naive_phase_error predicted_bound tiles worst_cell"
    );
    for distance_m in [1.0, 10.0, 100.0, 1_000.0] {
        let sample_plane = plane(distance_m, 3, 5);
        let reference = reference(&sweep_problem, &sample_plane);
        let dense = solve_dense_f32_cpu(
            &mut cx,
            &sweep_problem,
            sample_plane,
            PhaseBudget::default(),
            TileProfile::default(),
        )
        .unwrap();
        let report =
            compare_dense_to_reference(&reference, &dense, DifferentialTolerances::default())
                .unwrap();
        let phase = report.phase.expect("unit-amplitude phase is compared");
        let naive_phase_error =
            naive_forward_plane_phase_error(&sweep_problem, sample_plane, &reference);
        let evidence = dense.evidence();
        println!(
            "{distance_m:.0} {:.9e} {:.9e} {:.9e} {:.9e} {} ({},{})",
            evidence.observed_max_abs_psi_rad(),
            phase.error,
            naive_phase_error,
            evidence.predicted_max_abs_psi_rad(),
            evidence.tiles(),
            report.worst_row,
            report.worst_column,
        );
        assert!(report.passed(), "distance {distance_m}: {report:?}");
        assert!(phase.error <= phase.limit);
        assert!(evidence.observed_max_abs_psi_rad() <= evidence.phase_limit_rad());
        assert!(evidence.predicted_max_abs_psi_rad() <= evidence.phase_limit_rad());
        rows.push((phase.error, naive_phase_error));
    }

    assert!(
        rows.windows(2).all(|pair| pair[1].1 > pair[0].1),
        "naive absolute-f32 phase error must grow each decade: {rows:?}"
    );
    assert!(
        rows.last().unwrap().1 > rows.last().unwrap().0 * 1_000.0,
        "normalization must remain load-bearing at 1000 m: {rows:?}"
    );
}

fn naive_forward_plane_phase_error(
    problem: &InterferenceProblem,
    plane: SamplingPlane,
    reference: &sim_lib_interference_solve::HostPhasorField,
) -> f64 {
    let Emitter::ForwardPlane {
        through,
        direction,
        phase,
        ..
    } = problem.sources.iter().next().expect("phase sweep source")
    else {
        panic!("phase sweep fixture must contain one forward plane");
    };
    let origin = through.coordinates_metres().map(|value| value as f32);
    let direction = direction.components().map(|value| value as f32);
    let wavenumber = problem.wavenumber().real_radians_per_metre() as f32;
    let source_phase = phase.get() as f32;
    let mut maximum = 0.0_f64;
    for row in 0..plane.rows() {
        for column in 0..plane.columns() {
            let at = plane.point_at(row, column).unwrap().coordinates_metres();
            let x = direction[0] * (at[0] as f32 - origin[0]);
            let y = direction[1] * (at[1] as f32 - origin[1]);
            let z = direction[2] * (at[2] as f32 - origin[2]);
            let signed_distance = (x + y) + z;
            let absolute_phase = wavenumber * signed_distance + source_phase;
            let candidate_phase =
                f64::from(absolute_phase.sin()).atan2(f64::from(absolute_phase.cos()));
            let (reference_real, reference_imaginary) = reference.cell(row, column).unwrap();
            let reference_phase = reference_imaginary.atan2(reference_real);
            maximum = maximum.max(wrapped_phase_error(reference_phase, candidate_phase));
        }
    }
    maximum
}

fn wrapped_phase_error(left: f64, right: f64) -> f64 {
    let full_turn = 2.0 * std::f64::consts::PI;
    let difference = (left - right).abs().rem_euclid(full_turn);
    difference.min(full_turn - difference)
}
