use sim_lib_interference_solve::{
    MAX_ANALYTIC_RELATIVE_ERROR, MAX_METAMORPHIC_RELATIVE_ERROR, verify_reference_solver,
};

#[test]
fn full_reference_verification_matrix_passes() {
    let report = verify_reference_solver().unwrap();
    eprintln!("verification report: {report:?}");

    assert!(report.analytic_max_error <= MAX_ANALYTIC_RELATIVE_ERROR);
    assert!(report.reciprocity_relative <= MAX_METAMORPHIC_RELATIVE_ERROR);
    assert!(report.linearity_relative <= MAX_METAMORPHIC_RELATIVE_ERROR);
    assert!(report.rigid_motion_relative <= MAX_METAMORPHIC_RELATIVE_ERROR);
    assert!(report.global_phase_relative <= MAX_METAMORPHIC_RELATIVE_ERROR);
    assert!(report.source_permutation_identical);
    assert!((1.8..=2.2).contains(&report.helmholtz_observed_order));
}
// conformance: verification tests prove certificates, error bounds, and refusal paths.
