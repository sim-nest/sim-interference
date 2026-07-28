//! Named-scenario construction and structured runtime outputs.

use std::sync::Arc;

use sim_kernel::{Cx, Error, Expr, Result, Symbol, Value};
use sim_lib_interference_core::{
    FieldAmplitude, Hertz, MetresPerSecond, NepersPerMetre, PositiveMetres, Radians, ScalarMedium,
};
use sim_lib_interference_solve::{
    AperturePolicy, ExtremumKind, FringeReport, MultiToneProjection, NamedScenario,
    ScenarioBuilder, ScenarioKind, ScenarioLimits,
};
use sim_lib_numbers_tensor::{Tensor, TypedTensorStorage, domains};

use crate::{PlaneDescriptor, ProblemDescriptor, SamplingCertificateDescriptor, ops_values::*};

pub(crate) fn build_scenario(map: &[(Expr, Expr)]) -> Result<NamedScenario> {
    let limits = optional_field(map, "limits")
        .map(|value| scenario_limits(expect_map(value, "scenario limits")?))
        .transpose()?
        .unwrap_or_default();
    let builder = ScenarioBuilder::new(
        Hertz::new(required_decode(map, "frequency-hz")?)
            .map_err(domain_error("scenario frequency-hz"))?,
        ScalarMedium::new(
            MetresPerSecond::new(required_decode(map, "speed-m-s")?)
                .map_err(domain_error("scenario speed-m-s"))?,
            NepersPerMetre::new(required_decode(map, "attenuation-np-m")?)
                .map_err(domain_error("scenario attenuation-np-m"))?,
        ),
        PositiveMetres::new(required_decode(map, "singularity-radius-m")?)
            .map_err(domain_error("scenario singularity-radius-m"))?,
        limits,
    );
    let kind = required_symbol(map, "kind", "interference/scenarios")?;
    let scenario = match kind.name.as_ref() {
        "two-point" => {
            reject_scenario_fields(
                map,
                &[
                    "first-m",
                    "second-m",
                    "amplitude-per-source",
                    "relative-phase-rad",
                ],
            )?;
            builder.two_point(
                point3(
                    required_field(map, "first-m", "two-point scenario")?,
                    "first-m",
                )?,
                point3(
                    required_field(map, "second-m", "two-point scenario")?,
                    "second-m",
                )?,
                FieldAmplitude::new(required_decode(map, "amplitude-per-source")?)
                    .map_err(domain_error("scenario amplitude-per-source"))?,
                Radians::new(required_decode(map, "relative-phase-rad")?)
                    .map_err(domain_error("scenario relative-phase-rad"))?,
            )
        }
        "counter-propagating-planes" => {
            reject_scenario_fields(
                map,
                &[
                    "centre-m",
                    "axis",
                    "source-separation-m",
                    "amplitude-per-source",
                    "relative-phase-rad",
                ],
            )?;
            builder.counter_propagating_planes(
                point3(
                    required_field(map, "centre-m", "counter-propagating scenario")?,
                    "centre-m",
                )?,
                unit_vector(
                    required_field(map, "axis", "counter-propagating scenario")?,
                    "axis",
                )?,
                PositiveMetres::new(required_decode(map, "source-separation-m")?)
                    .map_err(domain_error("scenario source-separation-m"))?,
                FieldAmplitude::new(required_decode(map, "amplitude-per-source")?)
                    .map_err(domain_error("scenario amplitude-per-source"))?,
                Radians::new(required_decode(map, "relative-phase-rad")?)
                    .map_err(domain_error("scenario relative-phase-rad"))?,
            )
        }
        "phased-array" => {
            reject_scenario_fields(
                map,
                &[
                    "centre-m",
                    "element-axis",
                    "elements",
                    "spacing-m",
                    "total-amplitude",
                    "first-phase-rad",
                    "progressive-phase-rad",
                    "aperture-policy",
                ],
            )?;
            builder.phased_array(
                point3(
                    required_field(map, "centre-m", "phased-array scenario")?,
                    "centre-m",
                )?,
                unit_vector(
                    required_field(map, "element-axis", "phased-array scenario")?,
                    "element-axis",
                )?,
                required_decode(map, "elements")?,
                PositiveMetres::new(required_decode(map, "spacing-m")?)
                    .map_err(domain_error("scenario spacing-m"))?,
                FieldAmplitude::new(required_decode(map, "total-amplitude")?)
                    .map_err(domain_error("scenario total-amplitude"))?,
                Radians::new(required_decode(map, "first-phase-rad")?)
                    .map_err(domain_error("scenario first-phase-rad"))?,
                Radians::new(required_decode(map, "progressive-phase-rad")?)
                    .map_err(domain_error("scenario progressive-phase-rad"))?,
                aperture_policy(map)?,
            )
        }
        "discrete-aperture" => {
            reject_scenario_fields(
                map,
                &[
                    "centre-m",
                    "u-axis",
                    "v-axis",
                    "rows",
                    "cols",
                    "spacing-u-m",
                    "spacing-v-m",
                    "total-amplitude",
                    "phase-rad",
                    "aperture-policy",
                ],
            )?;
            builder.discrete_aperture(
                point3(
                    required_field(map, "centre-m", "aperture scenario")?,
                    "centre-m",
                )?,
                unit_vector(
                    required_field(map, "u-axis", "aperture scenario")?,
                    "u-axis",
                )?,
                unit_vector(
                    required_field(map, "v-axis", "aperture scenario")?,
                    "v-axis",
                )?,
                required_decode(map, "rows")?,
                required_decode(map, "cols")?,
                PositiveMetres::new(required_decode(map, "spacing-u-m")?)
                    .map_err(domain_error("scenario spacing-u-m"))?,
                PositiveMetres::new(required_decode(map, "spacing-v-m")?)
                    .map_err(domain_error("scenario spacing-v-m"))?,
                FieldAmplitude::new(required_decode(map, "total-amplitude")?)
                    .map_err(domain_error("scenario total-amplitude"))?,
                Radians::new(required_decode(map, "phase-rad")?)
                    .map_err(domain_error("scenario phase-rad"))?,
                aperture_policy(map)?,
            )
        }
        other => {
            return Err(Error::Eval(format!(
                "unknown interference scenario {other}"
            )));
        }
    };
    scenario.map_err(|error| Error::Eval(format!("interference scenario failed: {error}")))
}

fn reject_scenario_fields(map: &[(Expr, Expr)], specific: &[&str]) -> Result<()> {
    const COMMON: [&str; 6] = [
        "kind",
        "frequency-hz",
        "speed-m-s",
        "attenuation-np-m",
        "singularity-radius-m",
        "limits",
    ];
    let allowed = COMMON
        .into_iter()
        .chain(specific.iter().copied())
        .collect::<Vec<_>>();
    reject_extra(map, &allowed, "interference/scenarios request")
}

fn scenario_limits(map: &[(Expr, Expr)]) -> Result<ScenarioLimits> {
    reject_extra(
        map,
        &[
            "max-sources",
            "max-generated-id-bytes",
            "max-total-id-bytes",
        ],
        "scenario limits",
    )?;
    ScenarioLimits::new(
        required_decode(map, "max-sources")?,
        required_decode(map, "max-generated-id-bytes")?,
        required_decode(map, "max-total-id-bytes")?,
    )
    .map_err(|error| Error::Eval(format!("invalid scenario limits: {error}")))
}

fn aperture_policy(map: &[(Expr, Expr)]) -> Result<AperturePolicy> {
    match required_symbol(map, "aperture-policy", "scenario aperture-policy")?
        .name
        .as_ref()
    {
        "strict" => Ok(AperturePolicy::Strict),
        "annotate" => Ok(AperturePolicy::Annotate),
        other => Err(Error::Eval(format!("unknown aperture policy {other}"))),
    }
}

pub(crate) fn scenario_value(cx: &mut Cx, scenario: NamedScenario) -> Result<Value> {
    let (problem, certificate) = scenario.into_parts();
    let spacing = certificate.element_spacing_wavelengths;
    let certificate_value = cx.factory().table(vec![
        (
            Symbol::new("kind"),
            symbol_value(cx, scenario_kind_symbol(certificate.kind))?,
        ),
        (
            Symbol::new("source-count"),
            field_value(cx, &certificate.source_count)?,
        ),
        (
            Symbol::new("total-source-amplitude"),
            field_value(cx, &certificate.total_source_amplitude)?,
        ),
        (
            Symbol::new("amplitude-per-source"),
            field_value(cx, &certificate.amplitude_per_source)?,
        ),
        (
            Symbol::new("spacing-u-wavelengths"),
            field_value(cx, &spacing.u)?,
        ),
        (
            Symbol::new("spacing-v-wavelengths"),
            field_value(cx, &spacing.v)?,
        ),
        (
            Symbol::new("aperture-policy"),
            field_value(
                cx,
                &certificate.aperture_policy.map(|policy| match policy {
                    AperturePolicy::Strict => Symbol::new("strict"),
                    AperturePolicy::Annotate => Symbol::new("annotate"),
                }),
            )?,
        ),
    ])?;
    cx.factory().table(vec![
        (
            Symbol::new("problem"),
            boxed(cx, ProblemDescriptor::from_problem(&problem))?,
        ),
        (Symbol::new("certificate"), certificate_value),
    ])
}

pub(crate) fn fringe_report_value(cx: &mut Cx, report: &FringeReport) -> Result<Value> {
    let stats = cx.factory().table(vec![
        (Symbol::new("count"), field_value(cx, &report.stats.count)?),
        (
            Symbol::new("minimum"),
            field_value(cx, &report.stats.minimum)?,
        ),
        (
            Symbol::new("maximum"),
            field_value(cx, &report.stats.maximum)?,
        ),
        (Symbol::new("mean"), field_value(cx, &report.stats.mean)?),
        (
            Symbol::new("population-variance"),
            field_value(cx, &report.stats.population_variance)?,
        ),
    ])?;
    let extrema = report
        .extrema
        .iter()
        .map(|extremum| {
            cx.factory().table(vec![
                (Symbol::new("row"), field_value(cx, &extremum.row)?),
                (Symbol::new("column"), field_value(cx, &extremum.column)?),
                (Symbol::new("value"), field_value(cx, &extremum.value)?),
                (
                    Symbol::new("kind"),
                    symbol_value(
                        cx,
                        Symbol::new(match extremum.kind {
                            ExtremumKind::NodeCandidate => "node-candidate",
                            ExtremumKind::AntinodeCandidate => "antinode-candidate",
                        }),
                    )?,
                ),
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    let identity = report.projection;
    let projection = cx.factory().table(vec![
        (
            Symbol::new("source-rows"),
            field_value(cx, &identity.source_dimensions().rows())?,
        ),
        (
            Symbol::new("source-cols"),
            field_value(cx, &identity.source_dimensions().columns())?,
        ),
        (
            Symbol::new("target-rows"),
            field_value(cx, &identity.target_dimensions().rows())?,
        ),
        (
            Symbol::new("target-cols"),
            field_value(cx, &identity.target_dimensions().columns())?,
        ),
        (
            Symbol::new("phase-floor"),
            field_value(cx, &identity.phase_floor())?,
        ),
    ])?;
    cx.factory().table(vec![
        (
            Symbol::new("sampling"),
            boxed(
                cx,
                SamplingCertificateDescriptor::from_certificate(report.sampling),
            )?,
        ),
        (Symbol::new("projection"), projection),
        (Symbol::new("stats"), stats),
        (Symbol::new("extrema"), cx.factory().list(extrema)?),
        (
            Symbol::new("michelson-contrast"),
            field_value(cx, &report.michelson_contrast)?,
        ),
    ])
}

pub(crate) fn multitone_value(cx: &mut Cx, projection: &MultiToneProjection) -> Result<Value> {
    let tensor = Tensor::from_storage(
        vec![projection.rows(), projection.columns()],
        domains::f64(),
        Arc::new(TypedTensorStorage::<f64>::new(
            projection.samples().to_vec(),
        )),
    )?;
    let certificate = projection.certificate();
    let requirements = certificate.sampling_requirements();
    let components = certificate
        .components()
        .iter()
        .map(|component| {
            cx.factory().table(vec![
                (
                    Symbol::new("frequency-hz"),
                    field_value(cx, &component.frequency().get())?,
                ),
                (Symbol::new("weight"), field_value(cx, &component.weight())?),
                (
                    Symbol::new("sampling"),
                    boxed(
                        cx,
                        SamplingCertificateDescriptor::from_certificate(
                            component.sampling_certificate(),
                        ),
                    )?,
                ),
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    let certificate_value = cx.factory().table(vec![
        (
            Symbol::new("plane"),
            boxed(cx, PlaneDescriptor::from_plane(certificate.plane()))?,
        ),
        (
            Symbol::new("highest-frequency-hz"),
            field_value(cx, &requirements.highest_frequency().get())?,
        ),
        (
            Symbol::new("minimum-temporal-samples-per-second"),
            field_value(cx, &requirements.minimum_temporal_samples_per_second())?,
        ),
        (
            Symbol::new("maximum-temporal-step-seconds"),
            field_value(cx, &requirements.maximum_temporal_step_seconds())?,
        ),
        (Symbol::new("components"), cx.factory().list(components)?),
    ])?;
    cx.factory().table(vec![
        (Symbol::new("rows"), field_value(cx, &projection.rows())?),
        (Symbol::new("cols"), field_value(cx, &projection.columns())?),
        (
            Symbol::new("values"),
            cx.factory().opaque(Arc::new(tensor))?,
        ),
        (Symbol::new("certificate"), certificate_value),
    ])
}

fn scenario_kind_symbol(kind: ScenarioKind) -> Symbol {
    Symbol::new(match kind {
        ScenarioKind::TwoPoint => "two-point",
        ScenarioKind::CounterPropagatingPlanes => "counter-propagating-planes",
        ScenarioKind::PhasedArray => "phased-array",
        ScenarioKind::DiscreteAperture => "discrete-aperture",
    })
}
