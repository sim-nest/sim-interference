#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Lowers wave results and certificates to observations, never implicit ports.
use sim_lib_physics_adapter::{
    AdaptedModel, AdapterRefusal, DataOrigin, Dimension, Observation, PortDeclaration,
    StableIdentity,
};

/// Immutable wave result supplied by interference solve/compute owners.
#[derive(Clone, Debug, PartialEq)]
pub struct InterferenceAuditInput {
    /// Stable wave-model identity.
    pub model_id: String,
    /// Stable sampled-field identity.
    pub state_id: String,
    /// Explicit observation boundary.
    pub boundary_id: String,
    /// Normalized amplitude-squared observation.
    pub amplitude_squared: f64,
    /// Preserved sampling-certificate summary.
    pub sampling_certificate: String,
    /// Preserved solver/compute evidence summary.
    pub compute_evidence: String,
    /// State identities influenced by the wave result.
    pub influenced_states: Vec<String>,
    /// Model assumptions distinct from compute evidence.
    pub model_evidence: Vec<String>,
    /// Modeled or observed provider-record origin.
    pub origin: DataOrigin,
    /// Explicit physical conjugate pair, only when supplied by a boundary adapter.
    pub explicit_port: Option<(String, String, String)>,
}

/// Lowers a wave result to a compatible observation.
///
/// Squared magnitude is dimensionless and never called intensity, power, or a
/// lumped port. A port exists only when the caller supplies an explicit
/// boundary adapter with effort and flow kinds.
pub fn adapt_interference(input: InterferenceAuditInput) -> Result<AdaptedModel, AdapterRefusal> {
    let ports = input
        .explicit_port
        .map(|(id, effort_kind, flow_kind)| {
            Ok(PortDeclaration {
                id: StableIdentity::new(id)?,
                effort_kind,
                flow_kind,
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, AdapterRefusal>>()?;
    let result = AdaptedModel {
        model_id: StableIdentity::new(input.model_id)?,
        state_id: StableIdentity::new(input.state_id)?,
        boundary_id: StableIdentity::new(input.boundary_id)?,
        boundary_complete: true,
        ports,
        stores: vec![],
        events: vec![StableIdentity::new("event/interference-sample")?],
        observations: vec![Observation {
            id: StableIdentity::new("observation/amplitude-squared")?,
            kind: "wave:normalized-amplitude-squared".into(),
            dimension: Dimension([0; 7]),
            value: input.amplitude_squared,
            unit: "1".into(),
            origin: input.origin,
        }],
        influences: input
            .influenced_states
            .into_iter()
            .map(StableIdentity::new)
            .collect::<Result<_, _>>()?,
        model_evidence: input.model_evidence,
        solver_evidence: vec![input.sampling_certificate, input.compute_evidence],
    };
    result.validate()?;
    Ok(result)
}
