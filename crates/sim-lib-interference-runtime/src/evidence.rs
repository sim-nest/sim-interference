//! Truth-carrying sampling, work, solver, and complete-study records.

use sim_kernel::{Cx, Result, Symbol, Value};
use sim_lib_interference_core::{
    SamplingCertificate, SamplingPolicy, SamplingVerdict, WorkEstimate,
};
use sim_lib_interference_solve::{HostPhasorField, SolveEvidence};
use sim_lib_numbers_tensor::domains;

use crate::{
    PhasorFieldDescriptor, PlaneDescriptor, ProblemDescriptor,
    citizen::{RecordCitizenSpec, decode_record, encode_field, encode_record, invalid, next_field},
    records::{sample_plane, sample_problem},
};

/// Citizen descriptor for complete physical sampling evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingCertificateDescriptor {
    /// Comfortable carrier samples per wavelength.
    pub resolved_min_samples_per_wavelength: f64,
    /// Marginal carrier samples per wavelength.
    pub marginal_min_samples_per_wavelength: f64,
    /// Comfortable maximum fractional envelope change per cell.
    pub resolved_max_envelope_fraction_per_cell: f64,
    /// Marginal maximum fractional envelope change per cell.
    pub marginal_max_envelope_fraction_per_cell: f64,
    /// Carrier wavelength in metres.
    pub wavelength_m: f64,
    /// Carrier samples per wavelength on the `u` axis.
    pub samples_per_wavelength_u: f64,
    /// Carrier samples per wavelength on the `v` axis.
    pub samples_per_wavelength_v: f64,
    /// Squared-magnitude fringe samples on `u`.
    pub samples_per_power_fringe_u: f64,
    /// Squared-magnitude fringe samples on `v`.
    pub samples_per_power_fringe_v: f64,
    /// Nearest point-source distance, absent for plane-only problems.
    pub nearest_point_source_distance_m: Option<f64>,
    /// Conservative fractional `1/r` envelope change per cell.
    pub max_envelope_fraction_per_cell: f64,
    /// `interference/resolved`, `marginal`, or `aliased`.
    pub verdict: Symbol,
}

impl SamplingCertificateDescriptor {
    /// Projects a domain sampling certificate without dropping thresholds.
    pub fn from_certificate(certificate: SamplingCertificate) -> Self {
        Self {
            resolved_min_samples_per_wavelength: certificate
                .thresholds
                .resolved_min_samples_per_wavelength,
            marginal_min_samples_per_wavelength: certificate
                .thresholds
                .marginal_min_samples_per_wavelength,
            resolved_max_envelope_fraction_per_cell: certificate
                .thresholds
                .resolved_max_envelope_fraction_per_cell,
            marginal_max_envelope_fraction_per_cell: certificate
                .thresholds
                .marginal_max_envelope_fraction_per_cell,
            wavelength_m: certificate.wavelength_m,
            samples_per_wavelength_u: certificate.samples_per_wavelength_u,
            samples_per_wavelength_v: certificate.samples_per_wavelength_v,
            samples_per_power_fringe_u: certificate.samples_per_power_fringe_u,
            samples_per_power_fringe_v: certificate.samples_per_power_fringe_v,
            nearest_point_source_distance_m: certificate.nearest_point_source_distance_m,
            max_envelope_fraction_per_cell: certificate.max_envelope_fraction_per_cell,
            verdict: verdict_symbol(certificate.verdict),
        }
    }
}

/// Citizen descriptor for checked allocation and evaluation counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkEstimateDescriptor {
    /// Sampling cells.
    pub cells: u64,
    /// Coherent emitters.
    pub emitters: u64,
    /// Cell/emitter evaluations.
    pub emitter_evaluations: u64,
    /// Peak host bytes.
    pub host_bytes: u64,
    /// Result bytes.
    pub result_bytes: u64,
    /// Seven-point certificate stencil work.
    pub certificate_stencil_work: u64,
}

impl WorkEstimateDescriptor {
    /// Projects a checked domain estimate.
    pub fn from_estimate(estimate: WorkEstimate) -> Self {
        Self {
            cells: estimate.cells,
            emitters: estimate.emitters,
            emitter_evaluations: estimate.emitter_evaluations,
            host_bytes: estimate.host_bytes,
            result_bytes: estimate.result_bytes,
            certificate_stencil_work: estimate.certificate_stencil_work,
        }
    }

    /// Recomputes and checks every derived work dimension.
    pub fn to_estimate(self) -> Result<WorkEstimate> {
        let expected = WorkEstimate::new(self.cells, self.emitters)
            .map_err(|error| invalid("WorkEstimate", format!("{error:?}")))?;
        let actual = Self::from_estimate(expected);
        if actual != self {
            return Err(invalid(
                "WorkEstimate",
                "derived counts or byte sizes are inconsistent",
            ));
        }
        Ok(expected)
    }
}

impl RecordCitizenSpec for WorkEstimateDescriptor {
    const FIELDS: &'static [&'static str] = &[
        "cells",
        "emitters",
        "emitter-evaluations",
        "host-bytes",
        "result-bytes",
        "certificate-stencil-work",
    ];

    fn encode_fields(&self, _cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_field(&self.cells),
            encode_field(&self.emitters),
            encode_field(&self.emitter_evaluations),
            encode_field(&self.host_bytes),
            encode_field(&self.result_bytes),
            encode_field(&self.certificate_stencil_work),
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let value = Self {
            cells: next_field(cx, &mut fields, "cells")?,
            emitters: next_field(cx, &mut fields, "emitters")?,
            emitter_evaluations: next_field(cx, &mut fields, "emitter-evaluations")?,
            host_bytes: next_field(cx, &mut fields, "host-bytes")?,
            result_bytes: next_field(cx, &mut fields, "result-bytes")?,
            certificate_stencil_work: next_field(cx, &mut fields, "certificate-stencil-work")?,
        };
        value.validate()?;
        Ok(value)
    }

    fn example() -> Self {
        Self::from_estimate(WorkEstimate::new(4, 1).expect("example work"))
    }

    fn validate(&self) -> Result<()> {
        self.to_estimate().map(|_| ())
    }
}

impl_record_citizen!(WorkEstimateDescriptor, "interference/WorkEstimate", 6);

/// Citizen descriptor for one solver's immutable completion evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyEvidenceDescriptor {
    /// `interference/strict` or `interference/annotate`.
    pub sampling_policy: Symbol,
    /// Unchanged sampling truth.
    pub sampling: SamplingCertificateDescriptor,
    /// Preflight work estimate.
    pub work: WorkEstimateDescriptor,
    /// Provider identity.
    pub provider: Symbol,
    /// Component dtype.
    pub dtype: Symbol,
    /// Absolute Cartesian-component tolerance.
    pub component_absolute_tolerance: f64,
    /// Absolute squared-magnitude tolerance.
    pub squared_magnitude_absolute_tolerance: f64,
    /// Completed output cells.
    pub completed_cells: u64,
    /// Completed source evaluations.
    pub completed_emitter_evaluations: u64,
    /// Host-to-provider uploads.
    pub uploads: u64,
    /// Accepted provider submissions.
    pub submissions: u64,
    /// Intermediate host readbacks.
    pub intermediate_materializations: u64,
    /// Final component readbacks.
    pub final_materializations: u64,
    /// Ordered execution segments.
    pub segments: u64,
    /// Adapter/provider implementation identity.
    pub adapter: String,
    /// Optional measured hardware/profile identity.
    pub profile: Option<String>,
}

impl StudyEvidenceDescriptor {
    /// Projects evidence from the deterministic host reference solver.
    pub fn from_reference(evidence: &SolveEvidence) -> Self {
        let preflight = evidence.preflight();
        Self {
            sampling_policy: sampling_policy_symbol(preflight.sampling_policy),
            sampling: SamplingCertificateDescriptor::from_certificate(
                preflight.sampling_certificate,
            ),
            work: WorkEstimateDescriptor::from_estimate(preflight.work_estimate),
            provider: reference_provider_symbol(),
            dtype: domains::f64(),
            component_absolute_tolerance: 0.0,
            squared_magnitude_absolute_tolerance: 0.0,
            completed_cells: evidence.completed_cells(),
            completed_emitter_evaluations: evidence.completed_emitter_evaluations(),
            uploads: 0,
            submissions: 0,
            intermediate_materializations: 0,
            final_materializations: 0,
            segments: 1,
            adapter: "reference-f64".to_owned(),
            profile: None,
        }
    }

    /// Returns the checked sampling policy.
    pub fn sampling_policy(&self) -> Result<SamplingPolicy> {
        if self.sampling_policy == strict_policy_symbol() {
            Ok(SamplingPolicy::Strict)
        } else if self.sampling_policy == annotate_policy_symbol() {
            Ok(SamplingPolicy::Annotate)
        } else {
            Err(invalid(
                "StudyEvidence",
                format!("unknown sampling policy {}", self.sampling_policy),
            ))
        }
    }
}

impl RecordCitizenSpec for StudyEvidenceDescriptor {
    const FIELDS: &'static [&'static str] = &[
        "sampling-policy",
        "sampling",
        "work",
        "provider",
        "dtype",
        "component-absolute-tolerance",
        "squared-magnitude-absolute-tolerance",
        "completed-cells",
        "completed-emitter-evaluations",
        "uploads",
        "submissions",
        "intermediate-materializations",
        "final-materializations",
        "segments",
        "adapter",
        "profile",
    ];

    fn encode_fields(&self, cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_field(&self.sampling_policy),
            encode_record(cx, &self.sampling)?,
            encode_record(cx, &self.work)?,
            encode_field(&self.provider),
            encode_field(&self.dtype),
            encode_field(&self.component_absolute_tolerance),
            encode_field(&self.squared_magnitude_absolute_tolerance),
            encode_field(&self.completed_cells),
            encode_field(&self.completed_emitter_evaluations),
            encode_field(&self.uploads),
            encode_field(&self.submissions),
            encode_field(&self.intermediate_materializations),
            encode_field(&self.final_materializations),
            encode_field(&self.segments),
            encode_field(&self.adapter),
            encode_field(&self.profile),
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let sampling_policy = next_field(cx, &mut fields, "sampling-policy")?;
        let sampling = decode_record(
            cx,
            fields
                .next()
                .ok_or_else(|| invalid("StudyEvidence", "missing sampling evidence"))?,
            "sampling",
        )?;
        let work = decode_record(
            cx,
            fields
                .next()
                .ok_or_else(|| invalid("StudyEvidence", "missing work estimate"))?,
            "work",
        )?;
        let value = Self {
            sampling_policy,
            sampling,
            work,
            provider: next_field(cx, &mut fields, "provider")?,
            dtype: next_field(cx, &mut fields, "dtype")?,
            component_absolute_tolerance: next_field(
                cx,
                &mut fields,
                "component-absolute-tolerance",
            )?,
            squared_magnitude_absolute_tolerance: next_field(
                cx,
                &mut fields,
                "squared-magnitude-absolute-tolerance",
            )?,
            completed_cells: next_field(cx, &mut fields, "completed-cells")?,
            completed_emitter_evaluations: next_field(
                cx,
                &mut fields,
                "completed-emitter-evaluations",
            )?,
            uploads: next_field(cx, &mut fields, "uploads")?,
            submissions: next_field(cx, &mut fields, "submissions")?,
            intermediate_materializations: next_field(
                cx,
                &mut fields,
                "intermediate-materializations",
            )?,
            final_materializations: next_field(cx, &mut fields, "final-materializations")?,
            segments: next_field(cx, &mut fields, "segments")?,
            adapter: next_field(cx, &mut fields, "adapter")?,
            profile: next_field(cx, &mut fields, "profile")?,
        };
        value.validate()?;
        Ok(value)
    }

    fn example() -> Self {
        sample_study_evidence()
    }

    fn validate(&self) -> Result<()> {
        let policy = self.sampling_policy()?;
        let sampling = self.sampling.to_certificate()?;
        self.work.to_estimate()?;
        if policy == SamplingPolicy::Strict && sampling.verdict != SamplingVerdict::Resolved {
            return Err(invalid(
                "StudyEvidence",
                "strict policy cannot carry non-resolved sampling",
            ));
        }
        if self.provider.to_string().trim().is_empty() {
            return Err(invalid("StudyEvidence", "provider cannot be empty"));
        }
        if self.dtype != domains::f32() && self.dtype != domains::f64() {
            return Err(invalid(
                "StudyEvidence",
                "dtype must be numbers/f32 or numbers/f64",
            ));
        }
        for (name, value) in [
            (
                "component absolute tolerance",
                self.component_absolute_tolerance,
            ),
            (
                "squared-magnitude absolute tolerance",
                self.squared_magnitude_absolute_tolerance,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid(
                    "StudyEvidence",
                    format!("{name} must be finite and non-negative"),
                ));
            }
        }
        if self.completed_cells != self.work.cells
            || self.completed_emitter_evaluations != self.work.emitter_evaluations
        {
            return Err(invalid(
                "StudyEvidence",
                "completion counts must equal the admitted work",
            ));
        }
        if self.intermediate_materializations != 0 {
            return Err(invalid(
                "StudyEvidence",
                "intermediate materialization is forbidden",
            ));
        }
        if self.final_materializations > 2 {
            return Err(invalid(
                "StudyEvidence",
                "a two-component field permits at most two final materializations",
            ));
        }
        if self.segments == 0 {
            return Err(invalid(
                "StudyEvidence",
                "completed evidence must name at least one segment",
            ));
        }
        if self.adapter.trim().is_empty()
            || self
                .profile
                .as_ref()
                .is_some_and(|profile| profile.trim().is_empty())
        {
            return Err(invalid(
                "StudyEvidence",
                "adapter and present profile identities cannot be empty",
            ));
        }
        if self.provider == reference_provider_symbol()
            && (self.dtype != domains::f64()
                || self.component_absolute_tolerance != 0.0
                || self.squared_magnitude_absolute_tolerance != 0.0
                || self.uploads != 0
                || self.submissions != 0
                || self.final_materializations != 0
                || self.profile.is_some())
        {
            return Err(invalid(
                "StudyEvidence",
                "reference provider evidence must remain exact host f64",
            ));
        }
        Ok(())
    }
}

impl_record_citizen!(StudyEvidenceDescriptor, "interference/StudyEvidence", 16);

/// Complete Tensor-backed interference study with inseparable truth.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyDescriptor {
    /// Exact coherent problem.
    pub problem: ProblemDescriptor,
    /// Exact physical sampling plane.
    pub plane: PlaneDescriptor,
    /// Two-component Tensor field.
    pub field: PhasorFieldDescriptor,
    /// Sampling, work, provider, tolerance, and execution evidence.
    pub evidence: StudyEvidenceDescriptor,
}

impl StudyDescriptor {
    /// Builds and validates a complete study from canonical runtime records.
    ///
    /// Alternate study solvers use this constructor so they cannot bypass the
    /// same problem, plane, field, sampling, work, and dtype invariants as the
    /// reference provider.
    pub fn new(
        problem: ProblemDescriptor,
        plane: PlaneDescriptor,
        field: PhasorFieldDescriptor,
        evidence: StudyEvidenceDescriptor,
    ) -> Result<Self> {
        let value = Self {
            problem,
            plane,
            field,
            evidence,
        };
        value.validate()?;
        Ok(value)
    }

    /// Projects one completed reference solve into runtime records.
    pub fn from_reference(
        problem: &sim_lib_interference_core::InterferenceProblem,
        plane: sim_lib_interference_core::SamplingPlane,
        field: HostPhasorField,
        evidence: &SolveEvidence,
    ) -> Result<Self> {
        Self::new(
            ProblemDescriptor::from_problem(problem),
            PlaneDescriptor::from_plane(plane),
            PhasorFieldDescriptor::from_host(field)?,
            StudyEvidenceDescriptor::from_reference(evidence),
        )
    }
}

impl RecordCitizenSpec for StudyDescriptor {
    const FIELDS: &'static [&'static str] = &["problem", "plane", "field", "evidence"];

    fn encode_fields(&self, cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_record(cx, &self.problem)?,
            encode_record(cx, &self.plane)?,
            encode_record(cx, &self.field)?,
            encode_record(cx, &self.evidence)?,
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let value = Self {
            problem: decode_next_record(cx, &mut fields, "problem")?,
            plane: decode_next_record(cx, &mut fields, "plane")?,
            field: decode_next_record(cx, &mut fields, "field")?,
            evidence: decode_next_record(cx, &mut fields, "evidence")?,
        };
        value.validate()?;
        Ok(value)
    }

    fn example() -> Self {
        let value = Self {
            problem: sample_problem(),
            plane: sample_plane(),
            field: <PhasorFieldDescriptor as RecordCitizenSpec>::example(),
            evidence: sample_study_evidence(),
        };
        value.validate().expect("example study");
        value
    }

    fn validate(&self) -> Result<()> {
        let problem = self.problem.to_problem()?;
        let plane = self.plane.to_plane()?;
        self.field.validate()?;
        self.evidence.validate()?;
        if self.field.rows != self.plane.rows || self.field.cols != self.plane.columns {
            return Err(invalid(
                "Study",
                "field dimensions must equal the physical plane dimensions",
            ));
        }
        if self.field.real.dtype() != &self.evidence.dtype {
            return Err(invalid(
                "Study",
                "field dtype must equal the evidence dtype",
            ));
        }
        let expected_work = WorkEstimate::for_request(&problem, &plane)
            .map_err(|error| invalid("Study", format!("{error:?}")))?;
        if self.evidence.work != WorkEstimateDescriptor::from_estimate(expected_work) {
            return Err(invalid(
                "Study",
                "work evidence does not match the problem and plane",
            ));
        }
        let thresholds = self.evidence.sampling.to_certificate()?.thresholds;
        let expected_sampling =
            SamplingCertificate::measure_with_thresholds(&problem, &plane, thresholds)
                .map_err(|error| invalid("Study", format!("{error:?}")))?;
        if self.evidence.sampling
            != SamplingCertificateDescriptor::from_certificate(expected_sampling)
        {
            return Err(invalid(
                "Study",
                "sampling evidence does not match the problem and plane",
            ));
        }
        Ok(())
    }
}

impl_record_citizen!(StudyDescriptor, "interference/Study", 4);

fn decode_next_record<T>(
    cx: &mut Cx,
    fields: &mut impl Iterator<Item = Value>,
    name: &'static str,
) -> Result<T>
where
    T: sim_citizen::CitizenRuntime,
{
    decode_record(
        cx,
        fields
            .next()
            .ok_or_else(|| invalid("Study", format!("missing {name}")))?,
        name,
    )
}

pub(crate) fn sample_sampling() -> SamplingCertificateDescriptor {
    let problem = sample_problem().to_problem().expect("example problem");
    let plane = sample_plane().to_plane().expect("example plane");
    SamplingCertificateDescriptor::from_certificate(
        SamplingCertificate::measure(&problem, &plane).expect("example sampling"),
    )
}

fn sample_study_evidence() -> StudyEvidenceDescriptor {
    let problem = sample_problem().to_problem().expect("example problem");
    let plane = sample_plane().to_plane().expect("example plane");
    let work = WorkEstimate::for_request(&problem, &plane).expect("example work");
    let sampling =
        SamplingCertificate::measure(&problem, &plane).expect("example sampling certificate");
    let value = StudyEvidenceDescriptor {
        sampling_policy: annotate_policy_symbol(),
        sampling: SamplingCertificateDescriptor::from_certificate(sampling),
        work: WorkEstimateDescriptor::from_estimate(work),
        provider: reference_provider_symbol(),
        dtype: domains::f64(),
        component_absolute_tolerance: 0.0,
        squared_magnitude_absolute_tolerance: 0.0,
        completed_cells: work.cells,
        completed_emitter_evaluations: work.emitter_evaluations,
        uploads: 0,
        submissions: 0,
        intermediate_materializations: 0,
        final_materializations: 0,
        segments: 1,
        adapter: "reference-f64".to_owned(),
        profile: None,
    };
    value.validate().expect("example study evidence");
    value
}

pub(crate) fn verdict_symbol(verdict: SamplingVerdict) -> Symbol {
    Symbol::qualified(
        "interference",
        match verdict {
            SamplingVerdict::Resolved => "resolved",
            SamplingVerdict::Marginal => "marginal",
            SamplingVerdict::Aliased => "aliased",
        },
    )
}

fn sampling_policy_symbol(policy: SamplingPolicy) -> Symbol {
    match policy {
        SamplingPolicy::Strict => strict_policy_symbol(),
        SamplingPolicy::Annotate => annotate_policy_symbol(),
    }
}

fn strict_policy_symbol() -> Symbol {
    Symbol::qualified("interference", "strict")
}

fn annotate_policy_symbol() -> Symbol {
    Symbol::qualified("interference", "annotate")
}

fn reference_provider_symbol() -> Symbol {
    Symbol::qualified("interference", "reference-f64")
}
