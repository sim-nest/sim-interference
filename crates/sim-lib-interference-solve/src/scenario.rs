//! Bounded named builders for canonical interference fixtures.

use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceError, InterferenceProblem, Point3M,
    PositiveMetres, Radians, ScalarMedium, SourceSet, UnitVector3,
};

use crate::ScenarioError;
use crate::scenario_admission::{
    IdPlan, allocate_sources, require_dimension, require_limit, sources_from, sources_from_fallible,
};

/// Absolute safety ceiling for a named scenario.
pub const ABSOLUTE_MAX_SCENARIO_SOURCES: usize = 1_000_000;
/// Absolute safety ceiling for one generated source identity.
pub const ABSOLUTE_MAX_GENERATED_ID_BYTES: usize = 1_024;
/// Absolute safety ceiling for all generated identities in one scenario.
pub const ABSOLUTE_MAX_TOTAL_ID_BYTES: usize = 64 * 1_024 * 1_024;
/// Maximum element spacing admitted by strict aperture policy.
pub const STRICT_MAX_SPACING_WAVELENGTHS: f64 = 0.5;

/// Explicit allocation limits applied before scenario construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScenarioLimits {
    pub(crate) max_sources: usize,
    pub(crate) max_generated_id_bytes: usize,
    pub(crate) max_total_id_bytes: usize,
}

impl ScenarioLimits {
    /// Constructs limits within the crate's absolute safety ceilings.
    pub fn new(
        max_sources: usize,
        max_generated_id_bytes: usize,
        max_total_id_bytes: usize,
    ) -> Result<Self, ScenarioError> {
        require_limit("max-sources", max_sources, ABSOLUTE_MAX_SCENARIO_SOURCES)?;
        require_limit(
            "max-generated-id-bytes",
            max_generated_id_bytes,
            ABSOLUTE_MAX_GENERATED_ID_BYTES,
        )?;
        require_limit(
            "max-total-id-bytes",
            max_total_id_bytes,
            ABSOLUTE_MAX_TOTAL_ID_BYTES,
        )?;
        Ok(Self {
            max_sources,
            max_generated_id_bytes,
            max_total_id_bytes,
        })
    }

    /// Returns the maximum number of sources.
    pub fn max_sources(self) -> usize {
        self.max_sources
    }

    /// Returns the maximum bytes in one generated identity.
    pub fn max_generated_id_bytes(self) -> usize {
        self.max_generated_id_bytes
    }

    /// Returns the maximum aggregate generated identity bytes.
    pub fn max_total_id_bytes(self) -> usize {
        self.max_total_id_bytes
    }
}

impl Default for ScenarioLimits {
    fn default() -> Self {
        Self {
            max_sources: 4_096,
            max_generated_id_bytes: 96,
            max_total_id_bytes: 256 * 1_024,
        }
    }
}

/// Whether sparse spatial sampling is rejected or retained as evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AperturePolicy {
    /// Refuse any active neighbour spacing above one half-wavelength.
    #[default]
    Strict,
    /// Construct the approximation and retain its spacing in the certificate.
    Annotate,
}

/// Named approximation created by a scenario builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScenarioKind {
    /// Two ideal isotropic point sources.
    TwoPoint,
    /// Two ideal plane waves directed into their shared slab.
    CounterPropagatingPlanes,
    /// A centred line of ideal isotropic point elements with progressive phase.
    PhasedArray,
    /// A centred rectangular grid of ideal isotropic point elements.
    DiscreteAperture,
}

/// Neighbour spacing expressed against the unattenuated wavelength.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementSpacingWavelengths {
    /// Line or aperture-column spacing; absent for a singleton axis.
    pub u: Option<f64>,
    /// Aperture-row spacing; absent for a singleton axis or line array.
    pub v: Option<f64>,
}

impl ElementSpacingWavelengths {
    /// Returns the largest active neighbour spacing, or `None` for one element.
    pub fn maximum(self) -> Option<f64> {
        match (self.u, self.v) {
            (Some(u), Some(v)) => Some(u.max(v)),
            (Some(value), None) | (None, Some(value)) => Some(value),
            (None, None) => None,
        }
    }
}

/// Evidence describing a named scenario and its discrete approximation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScenarioCertificate {
    /// Named builder used to create the problem.
    pub kind: ScenarioKind,
    /// Number of ideal sources created.
    pub source_count: usize,
    /// Requested coherent amplitude sum for array/aperture builders.
    ///
    /// Pair builders report the sum of their two equal per-source amplitudes.
    pub total_source_amplitude: f64,
    /// Equal amplitude carried by every source.
    pub amplitude_per_source: f64,
    /// Wavelength-relative neighbour spacing for discrete approximations.
    pub element_spacing_wavelengths: ElementSpacingWavelengths,
    /// Policy applied to the reported spacing.
    pub aperture_policy: Option<AperturePolicy>,
}

/// One checked problem plus the approximation evidence that created it.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedScenario {
    problem: InterferenceProblem,
    certificate: ScenarioCertificate,
}

impl NamedScenario {
    /// Returns the complete coherent problem.
    pub fn problem(&self) -> &InterferenceProblem {
        &self.problem
    }

    /// Returns construction and approximation evidence.
    pub fn certificate(&self) -> ScenarioCertificate {
        self.certificate
    }

    /// Consumes the scenario into its problem and evidence.
    pub fn into_parts(self) -> (InterferenceProblem, ScenarioCertificate) {
        (self.problem, self.certificate)
    }
}

/// Reusable physical context and limits for named scenario construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScenarioBuilder {
    frequency: Hertz,
    medium: ScalarMedium,
    singularity_radius: PositiveMetres,
    limits: ScenarioLimits,
}

impl ScenarioBuilder {
    /// Constructs a builder from checked physical values and explicit limits.
    pub fn new(
        frequency: Hertz,
        medium: ScalarMedium,
        singularity_radius: PositiveMetres,
        limits: ScenarioLimits,
    ) -> Self {
        Self {
            frequency,
            medium,
            singularity_radius,
            limits,
        }
    }

    /// Builds two equal-amplitude ideal isotropic point sources.
    pub fn two_point(
        self,
        first: Point3M,
        second: Point3M,
        amplitude_per_source: FieldAmplitude,
        relative_phase: Radians,
    ) -> Result<NamedScenario, ScenarioError> {
        let plan = IdPlan::admit("scenario/two-point/", 2, self.limits)?;
        let total_amplitude = checked_total_amplitude(amplitude_per_source, 2)?;
        let phases = [Radians::new(0.0)?, relative_phase];
        let positions = [first, second];
        let sources = sources_from(plan, |index, id| Emitter::Point {
            id,
            position: positions[index],
            amplitude_at_reference: amplitude_per_source,
            phase: phases[index],
        })?;
        self.finish(
            ScenarioKind::TwoPoint,
            sources,
            total_amplitude,
            amplitude_per_source.get(),
            ElementSpacingWavelengths { u: None, v: None },
            None,
        )
    }

    /// Builds equal plane waves directed into a slab around `centre`.
    pub fn counter_propagating_planes(
        self,
        centre: Point3M,
        axis: UnitVector3,
        source_separation: PositiveMetres,
        amplitude_per_source: FieldAmplitude,
        relative_phase: Radians,
    ) -> Result<NamedScenario, ScenarioError> {
        let plan = IdPlan::admit("scenario/counter-plane/", 2, self.limits)?;
        let total_amplitude = checked_total_amplitude(amplitude_per_source, 2)?;
        let mut sources = allocate_sources(plan.count)?;
        let half_separation = source_separation.get() / 2.0;
        let first = offset_point(centre, axis, -half_separation)?;
        let second = offset_point(centre, axis, half_separation)?;
        let [x, y, z] = axis.components();
        let reverse = UnitVector3::new(-x, -y, -z)?;
        sources.push(Emitter::ForwardPlane {
            id: plan.id(0),
            through: first,
            direction: axis,
            amplitude: amplitude_per_source,
            phase: Radians::new(0.0)?,
        });
        sources.push(Emitter::ForwardPlane {
            id: plan.id(1),
            through: second,
            direction: reverse,
            amplitude: amplitude_per_source,
            phase: relative_phase,
        });
        self.finish(
            ScenarioKind::CounterPropagatingPlanes,
            sources,
            total_amplitude,
            amplitude_per_source.get(),
            ElementSpacingWavelengths { u: None, v: None },
            None,
        )
    }

    /// Builds a centred line array of ideal point elements.
    #[allow(clippy::too_many_arguments)]
    pub fn phased_array(
        self,
        centre: Point3M,
        element_axis: UnitVector3,
        elements: usize,
        spacing: PositiveMetres,
        total_amplitude: FieldAmplitude,
        first_phase: Radians,
        progressive_phase: Radians,
        policy: AperturePolicy,
    ) -> Result<NamedScenario, ScenarioError> {
        require_dimension("elements", elements)?;
        let plan = IdPlan::admit("scenario/phased-array/", elements, self.limits)?;
        let reported_spacing = self.spacing((elements > 1).then_some(spacing), None, policy)?;
        let amplitude = normalized_amplitude(total_amplitude, elements)?;
        let sources = sources_from_fallible(plan, |index, id| {
            let offset = centred_offset(index, elements, spacing.get())?;
            Ok(Emitter::Point {
                id,
                position: offset_point(centre, element_axis, offset)?,
                amplitude_at_reference: amplitude,
                phase: Radians::new(first_phase.get() + index as f64 * progressive_phase.get())?,
            })
        })?;
        self.finish(
            ScenarioKind::PhasedArray,
            sources,
            total_amplitude.get(),
            amplitude.get(),
            reported_spacing,
            Some(policy),
        )
    }

    /// Builds a centred rectangular aperture of equal ideal point elements.
    #[allow(clippy::too_many_arguments)]
    pub fn discrete_aperture(
        self,
        centre: Point3M,
        u_axis: UnitVector3,
        v_axis: UnitVector3,
        rows: usize,
        columns: usize,
        spacing_u: PositiveMetres,
        spacing_v: PositiveMetres,
        total_amplitude: FieldAmplitude,
        phase: Radians,
        policy: AperturePolicy,
    ) -> Result<NamedScenario, ScenarioError> {
        require_dimension("rows", rows)?;
        require_dimension("columns", columns)?;
        let count = rows
            .checked_mul(columns)
            .ok_or(ScenarioError::SourceCountOverflow { rows, columns })?;
        let plan = IdPlan::admit("scenario/aperture/", count, self.limits)?;
        require_orthogonal_axes(u_axis, v_axis)?;
        let spacing = self.spacing(
            (columns > 1).then_some(spacing_u),
            (rows > 1).then_some(spacing_v),
            policy,
        )?;
        let amplitude = normalized_amplitude(total_amplitude, count)?;
        let sources = sources_from_fallible(plan, |index, id| {
            let row = index / columns;
            let column = index % columns;
            let offset_u = centred_offset(column, columns, spacing_u.get())?;
            let offset_v = centred_offset(row, rows, spacing_v.get())?;
            let along_u = offset_point(centre, u_axis, offset_u)?;
            Ok(Emitter::Point {
                id,
                position: offset_point(along_u, v_axis, offset_v)?,
                amplitude_at_reference: amplitude,
                phase,
            })
        })?;
        self.finish(
            ScenarioKind::DiscreteAperture,
            sources,
            total_amplitude.get(),
            amplitude.get(),
            spacing,
            Some(policy),
        )
    }

    fn spacing(
        self,
        u: Option<PositiveMetres>,
        v: Option<PositiveMetres>,
        policy: AperturePolicy,
    ) -> Result<ElementSpacingWavelengths, ScenarioError> {
        let wavelength = self.medium.speed().get() / self.frequency.get();
        let spacing = ElementSpacingWavelengths {
            u: u.map(|value| value.get() / wavelength),
            v: v.map(|value| value.get() / wavelength),
        };
        for (axis, value) in [("u", spacing.u), ("v", spacing.v)] {
            if let Some(value) = value
                && policy == AperturePolicy::Strict
                && value > STRICT_MAX_SPACING_WAVELENGTHS
            {
                return Err(ScenarioError::SparseAperture {
                    axis,
                    spacing_wavelengths: value,
                    maximum_wavelengths: STRICT_MAX_SPACING_WAVELENGTHS,
                });
            }
        }
        Ok(spacing)
    }

    fn finish(
        self,
        kind: ScenarioKind,
        sources: Vec<Emitter>,
        total_source_amplitude: f64,
        amplitude_per_source: f64,
        element_spacing_wavelengths: ElementSpacingWavelengths,
        aperture_policy: Option<AperturePolicy>,
    ) -> Result<NamedScenario, ScenarioError> {
        let source_count = sources.len();
        let problem = InterferenceProblem::new(
            self.frequency,
            self.medium,
            SourceSet::new(sources)?,
            self.singularity_radius,
        );
        Ok(NamedScenario {
            problem,
            certificate: ScenarioCertificate {
                kind,
                source_count,
                total_source_amplitude,
                amplitude_per_source,
                element_spacing_wavelengths,
                aperture_policy,
            },
        })
    }
}

fn normalized_amplitude(
    total: FieldAmplitude,
    count: usize,
) -> Result<FieldAmplitude, ScenarioError> {
    FieldAmplitude::new(total.get() / count as f64).map_err(ScenarioError::from)
}

fn checked_total_amplitude(per_source: FieldAmplitude, count: usize) -> Result<f64, ScenarioError> {
    FieldAmplitude::new(per_source.get() * count as f64)
        .map(FieldAmplitude::get)
        .map_err(ScenarioError::from)
}

fn centred_offset(index: usize, count: usize, spacing_metres: f64) -> Result<f64, ScenarioError> {
    let offset = (index as f64 - (count - 1) as f64 / 2.0) * spacing_metres;
    if offset.is_finite() {
        Ok(offset)
    } else {
        Err(InterferenceError::InvalidQuantity {
            name: "scenario-offset-m",
            value: offset,
        }
        .into())
    }
}

fn offset_point(
    point: Point3M,
    axis: UnitVector3,
    distance_metres: f64,
) -> Result<Point3M, ScenarioError> {
    let [x, y, z] = point.coordinates_metres();
    let [axis_x, axis_y, axis_z] = axis.components();
    Point3M::from_metres(
        x + distance_metres * axis_x,
        y + distance_metres * axis_y,
        z + distance_metres * axis_z,
    )
    .map_err(ScenarioError::from)
}

fn require_orthogonal_axes(u_axis: UnitVector3, v_axis: UnitVector3) -> Result<(), ScenarioError> {
    let [ux, uy, uz] = u_axis.components();
    let [vx, vy, vz] = v_axis.components();
    let dot_product = ux * vx + uy * vy + uz * vz;
    if dot_product.abs() > sim_lib_interference_core::SAMPLING_AXIS_ORTHOGONALITY_TOLERANCE {
        Err(InterferenceError::NonOrthogonalSamplingAxes {
            dot_product,
            max_abs_dot_product: sim_lib_interference_core::SAMPLING_AXIS_ORTHOGONALITY_TOLERANCE,
        }
        .into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "scenario_tests.rs"]
mod tests;
