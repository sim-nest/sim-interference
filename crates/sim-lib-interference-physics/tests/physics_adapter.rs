// conformance: physics adaptation preserves evidence and refuses invented physical ports.

use sim_lib_interference_physics::{InterferenceAuditInput, adapt_interference};
use sim_lib_physics_adapter::{AdapterRefusal, DIMENSIONLESS, DataOrigin};
fn input() -> InterferenceAuditInput {
    InterferenceAuditInput {
        model_id: "wave/model/two-source".into(),
        state_id: "wave/field/1".into(),
        boundary_id: "wave/boundary/plane".into(),
        amplitude_squared: 4.0,
        sampling_certificate: "resolved:32-samples-per-wavelength".into(),
        compute_evidence: "cpu-reference:cells=64".into(),
        influenced_states: vec!["state/detector".into()],
        model_evidence: vec!["homogeneous-scalar-wave".into()],
        origin: DataOrigin::Modeled,
        explicit_port: None,
    }
}
#[test]
fn preserves_wave_certificates_without_inventing_a_port() {
    let out = adapt_interference(input()).unwrap();
    assert_eq!(out.observations[0].dimension, DIMENSIONLESS);
    assert_eq!(
        out.observations[0].kind,
        "wave:normalized-amplitude-squared"
    );
    assert!(out.ports.is_empty());
    assert_eq!(out.solver_evidence.len(), 2);
    assert_eq!(
        out.validate_lumped_audit(),
        Err(AdapterRefusal::MissingPort)
    );
}
#[test]
fn retains_origin_and_influences() {
    let mut value = input();
    value.origin = DataOrigin::Observed;
    let out = adapt_interference(value).unwrap();
    assert_eq!(out.observations[0].origin, DataOrigin::Observed);
    assert_eq!(out.influences[0].as_str(), "state/detector");
}
