//! Validated medium, emitter, problem, and physical-plane records.

use sim_kernel::{Cx, Result, Symbol, Value};
use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, ScalarMedium, SourceSet, UnitVector3,
};

use crate::citizen::{
    RecordCitizenSpec, decode_record, decode_records, encode_field, encode_record, encode_records,
    invalid, next_field,
};

/// Citizen descriptor for a homogeneous scalar propagation medium.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MediumDescriptor {
    /// Propagation speed in metres per second.
    pub speed_m_s: f64,
    /// Attenuation in nepers per metre.
    pub attenuation_np_m: f64,
}

impl MediumDescriptor {
    /// Checks physical units and builds a medium descriptor.
    pub fn new(speed_m_s: f64, attenuation_np_m: f64) -> Result<Self> {
        let value = Self {
            speed_m_s,
            attenuation_np_m,
        };
        value.validate()?;
        Ok(value)
    }

    /// Projects an admitted domain medium.
    pub fn from_medium(medium: ScalarMedium) -> Self {
        Self {
            speed_m_s: medium.speed().get(),
            attenuation_np_m: medium.attenuation().get(),
        }
    }

    /// Reconstructs the dependency-free domain medium.
    pub fn to_medium(self) -> Result<ScalarMedium> {
        Ok(ScalarMedium::new(
            MetresPerSecond::new(self.speed_m_s)
                .map_err(|error| invalid("Medium", format!("{error:?}")))?,
            NepersPerMetre::new(self.attenuation_np_m)
                .map_err(|error| invalid("Medium", format!("{error:?}")))?,
        ))
    }
}

impl RecordCitizenSpec for MediumDescriptor {
    const FIELDS: &'static [&'static str] = &["speed-m-s", "attenuation-np-m"];

    fn encode_fields(&self, _cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_field(&self.speed_m_s),
            encode_field(&self.attenuation_np_m),
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        Self::new(
            next_field(cx, &mut fields, "speed-m-s")?,
            next_field(cx, &mut fields, "attenuation-np-m")?,
        )
    }

    fn example() -> Self {
        Self::new(343.0, 0.0).expect("example medium")
    }

    fn validate(&self) -> Result<()> {
        self.to_medium().map(|_| ())
    }
}

impl_record_citizen!(MediumDescriptor, "interference/Medium", 2);

/// Citizen descriptor for a point or forward-plane coherent emitter.
#[derive(Clone, Debug, PartialEq)]
pub struct EmitterDescriptor {
    /// `interference/point` or `interference/forward-plane`.
    pub kind: Symbol,
    /// Stable source identity.
    pub id: String,
    /// Point position or a point on the zero-phase plane, in metres.
    pub anchor_m: Vec<f64>,
    /// Forward direction; absent for a point source.
    pub direction: Option<Vec<f64>>,
    /// Field amplitude at the point reference distance or zero-phase plane.
    pub amplitude: f64,
    /// Source phase in radians, canonically stored in `[-pi, pi)`.
    pub phase_rad: f64,
}

impl EmitterDescriptor {
    /// Projects a checked domain emitter.
    pub fn from_emitter(emitter: &Emitter) -> Self {
        match emitter {
            Emitter::Point {
                id,
                position,
                amplitude_at_reference,
                phase,
            } => Self {
                kind: point_kind(),
                id: id.clone(),
                anchor_m: position.coordinates_metres().to_vec(),
                direction: None,
                amplitude: amplitude_at_reference.get(),
                phase_rad: phase.get(),
            },
            Emitter::ForwardPlane {
                id,
                through,
                direction,
                amplitude,
                phase,
            } => Self {
                kind: forward_plane_kind(),
                id: id.clone(),
                anchor_m: through.coordinates_metres().to_vec(),
                direction: Some(direction.components().to_vec()),
                amplitude: amplitude.get(),
                phase_rad: phase.get(),
            },
        }
    }

    /// Reconstructs the checked domain emitter.
    pub fn to_emitter(&self) -> Result<Emitter> {
        if self.id.is_empty() {
            return Err(invalid("Emitter", "source id cannot be empty"));
        }
        let [x, y, z] = vector3(&self.anchor_m, "emitter anchor")?;
        let anchor = Point3M::from_metres(x, y, z)
            .map_err(|error| invalid("Emitter", format!("{error:?}")))?;
        let amplitude = FieldAmplitude::new(self.amplitude)
            .map_err(|error| invalid("Emitter", format!("{error:?}")))?;
        let phase = Radians::new(self.phase_rad)
            .map_err(|error| invalid("Emitter", format!("{error:?}")))?;
        if phase.get().to_bits() != self.phase_rad.to_bits() {
            return Err(invalid(
                "Emitter",
                "phase-rad must already be canonical in [-pi, pi)",
            ));
        }
        if self.kind == point_kind() {
            if self.direction.is_some() {
                return Err(invalid(
                    "Emitter",
                    "point emitters cannot carry a direction",
                ));
            }
            Ok(Emitter::Point {
                id: self.id.clone(),
                position: anchor,
                amplitude_at_reference: amplitude,
                phase,
            })
        } else if self.kind == forward_plane_kind() {
            let [dx, dy, dz] = vector3(
                self.direction
                    .as_deref()
                    .ok_or_else(|| invalid("Emitter", "forward-plane direction is required"))?,
                "emitter direction",
            )?;
            let direction = UnitVector3::new(dx, dy, dz)
                .map_err(|error| invalid("Emitter", format!("{error:?}")))?;
            let normalized = direction.components();
            if normalized
                .iter()
                .zip([dx, dy, dz])
                .any(|(actual, supplied)| actual.to_bits() != supplied.to_bits())
            {
                return Err(invalid(
                    "Emitter",
                    "forward-plane direction must already be normalized",
                ));
            }
            Ok(Emitter::ForwardPlane {
                id: self.id.clone(),
                through: anchor,
                direction,
                amplitude,
                phase,
            })
        } else {
            Err(invalid(
                "Emitter",
                format!("unknown emitter kind {}", self.kind),
            ))
        }
    }
}

impl RecordCitizenSpec for EmitterDescriptor {
    const FIELDS: &'static [&'static str] = &[
        "kind",
        "id",
        "anchor-m",
        "direction",
        "amplitude",
        "phase-rad",
    ];

    fn encode_fields(&self, _cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_field(&self.kind),
            encode_field(&self.id),
            encode_field(&self.anchor_m),
            encode_field(&self.direction),
            encode_field(&self.amplitude),
            encode_field(&self.phase_rad),
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let value = Self {
            kind: next_field(cx, &mut fields, "kind")?,
            id: next_field(cx, &mut fields, "id")?,
            anchor_m: next_field(cx, &mut fields, "anchor-m")?,
            direction: next_field(cx, &mut fields, "direction")?,
            amplitude: next_field(cx, &mut fields, "amplitude")?,
            phase_rad: next_field(cx, &mut fields, "phase-rad")?,
        };
        value.validate()?;
        Ok(value)
    }

    fn example() -> Self {
        Self::from_emitter(&Emitter::Point {
            id: "source".to_owned(),
            position: Point3M::from_metres(0.0, 0.0, 0.0).expect("example point"),
            amplitude_at_reference: FieldAmplitude::new(1.0).expect("example amplitude"),
            phase: Radians::new(0.0).expect("example phase"),
        })
    }

    fn validate(&self) -> Result<()> {
        self.to_emitter().map(|_| ())
    }
}

impl_record_citizen!(EmitterDescriptor, "interference/Emitter", 6);

/// Citizen descriptor for one complete coherent single-frequency problem.
#[derive(Clone, Debug, PartialEq)]
pub struct ProblemDescriptor {
    /// Shared coherent frequency in hertz.
    pub frequency_hz: f64,
    /// Homogeneous scalar medium.
    pub medium: MediumDescriptor,
    /// Non-empty sources in canonical stable-id order.
    pub emitters: Vec<EmitterDescriptor>,
    /// Point-source exclusion radius in metres.
    pub singularity_radius_m: f64,
}

impl ProblemDescriptor {
    /// Projects a checked domain problem.
    pub fn from_problem(problem: &InterferenceProblem) -> Self {
        Self {
            frequency_hz: problem.frequency.get(),
            medium: MediumDescriptor::from_medium(problem.medium),
            emitters: problem
                .sources
                .iter()
                .map(EmitterDescriptor::from_emitter)
                .collect(),
            singularity_radius_m: problem.singularity_radius.get(),
        }
    }

    /// Reconstructs the checked dependency-free problem.
    pub fn to_problem(&self) -> Result<InterferenceProblem> {
        let frequency = Hertz::new(self.frequency_hz)
            .map_err(|error| invalid("Problem", format!("{error:?}")))?;
        let sources = self
            .emitters
            .iter()
            .map(EmitterDescriptor::to_emitter)
            .collect::<Result<Vec<_>>>()?;
        let source_set =
            SourceSet::new(sources).map_err(|error| invalid("Problem", format!("{error:?}")))?;
        let canonical_ids = source_set.iter().map(Emitter::id);
        if !canonical_ids.eq(self.emitters.iter().map(|emitter| emitter.id.as_str())) {
            return Err(invalid(
                "Problem",
                "emitters must be in canonical stable-id order",
            ));
        }
        let radius = PositiveMetres::new(self.singularity_radius_m)
            .map_err(|error| invalid("Problem", format!("{error:?}")))?;
        Ok(InterferenceProblem::new(
            frequency,
            self.medium.to_medium()?,
            source_set,
            radius,
        ))
    }
}

impl RecordCitizenSpec for ProblemDescriptor {
    const FIELDS: &'static [&'static str] =
        &["frequency-hz", "medium", "emitters", "singularity-radius-m"];

    fn encode_fields(&self, cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_field(&self.frequency_hz),
            encode_record(cx, &self.medium)?,
            encode_records(cx, &self.emitters)?,
            encode_field(&self.singularity_radius_m),
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let frequency_hz = next_field(cx, &mut fields, "frequency-hz")?;
        let medium = decode_record(
            cx,
            fields
                .next()
                .ok_or_else(|| invalid("Problem", "missing medium"))?,
            "medium",
        )?;
        let emitters = decode_records(
            cx,
            fields
                .next()
                .ok_or_else(|| invalid("Problem", "missing emitters"))?,
            "emitters",
        )?;
        let singularity_radius_m = next_field(cx, &mut fields, "singularity-radius-m")?;
        let value = Self {
            frequency_hz,
            medium,
            emitters,
            singularity_radius_m,
        };
        value.validate()?;
        Ok(value)
    }

    fn example() -> Self {
        sample_problem()
    }

    fn validate(&self) -> Result<()> {
        self.to_problem().map(|_| ())
    }
}

impl_record_citizen!(ProblemDescriptor, "interference/Problem", 4);

/// Citizen descriptor for an orthonormal finite physical sampling plane.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneDescriptor {
    /// Corner before the first cell, in metres.
    pub origin_m: Vec<f64>,
    /// Unit direction along columns.
    pub u_axis: Vec<f64>,
    /// Unit direction along rows.
    pub v_axis: Vec<f64>,
    /// Physical column-axis extent in metres.
    pub extent_u_m: f64,
    /// Physical row-axis extent in metres.
    pub extent_v_m: f64,
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub columns: usize,
}

impl PlaneDescriptor {
    /// Projects a checked sampling plane.
    pub fn from_plane(plane: SamplingPlane) -> Self {
        Self {
            origin_m: plane.origin().coordinates_metres().to_vec(),
            u_axis: plane.u_axis().components().to_vec(),
            v_axis: plane.v_axis().components().to_vec(),
            extent_u_m: plane.extent_u().get(),
            extent_v_m: plane.extent_v().get(),
            rows: plane.rows(),
            columns: plane.columns(),
        }
    }

    /// Reconstructs and revalidates the physical plane.
    pub fn to_plane(&self) -> Result<SamplingPlane> {
        let [x, y, z] = vector3(&self.origin_m, "plane origin")?;
        let [ux, uy, uz] = vector3(&self.u_axis, "plane u-axis")?;
        let [vx, vy, vz] = vector3(&self.v_axis, "plane v-axis")?;
        let u_axis = canonical_unit_vector(ux, uy, uz, "u-axis")?;
        let v_axis = canonical_unit_vector(vx, vy, vz, "v-axis")?;
        SamplingPlane::new(
            Point3M::from_metres(x, y, z)
                .map_err(|error| invalid("Plane", format!("{error:?}")))?,
            u_axis,
            v_axis,
            PositiveMetres::new(self.extent_u_m)
                .map_err(|error| invalid("Plane", format!("{error:?}")))?,
            PositiveMetres::new(self.extent_v_m)
                .map_err(|error| invalid("Plane", format!("{error:?}")))?,
            self.rows,
            self.columns,
        )
        .map_err(|error| invalid("Plane", format!("{error:?}")))
    }
}

impl RecordCitizenSpec for PlaneDescriptor {
    const FIELDS: &'static [&'static str] = &[
        "origin-m",
        "u-axis",
        "v-axis",
        "extent-u-m",
        "extent-v-m",
        "rows",
        "columns",
    ];

    fn encode_fields(&self, _cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_field(&self.origin_m),
            encode_field(&self.u_axis),
            encode_field(&self.v_axis),
            encode_field(&self.extent_u_m),
            encode_field(&self.extent_v_m),
            encode_field(&self.rows),
            encode_field(&self.columns),
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let value = Self {
            origin_m: next_field(cx, &mut fields, "origin-m")?,
            u_axis: next_field(cx, &mut fields, "u-axis")?,
            v_axis: next_field(cx, &mut fields, "v-axis")?,
            extent_u_m: next_field(cx, &mut fields, "extent-u-m")?,
            extent_v_m: next_field(cx, &mut fields, "extent-v-m")?,
            rows: next_field(cx, &mut fields, "rows")?,
            columns: next_field(cx, &mut fields, "columns")?,
        };
        value.validate()?;
        Ok(value)
    }

    fn example() -> Self {
        sample_plane()
    }

    fn validate(&self) -> Result<()> {
        self.to_plane().map(|_| ())
    }
}

impl_record_citizen!(PlaneDescriptor, "interference/Plane", 7);

pub(crate) fn sample_problem() -> ProblemDescriptor {
    let medium = MediumDescriptor::new(343.0, 0.0).expect("example medium");
    let emitter = <EmitterDescriptor as RecordCitizenSpec>::example();
    let value = ProblemDescriptor {
        frequency_hz: 1_000.0,
        medium,
        emitters: vec![emitter],
        singularity_radius_m: 0.01,
    };
    value.validate().expect("example problem");
    value
}

pub(crate) fn sample_plane() -> PlaneDescriptor {
    let value = PlaneDescriptor {
        origin_m: vec![0.0, 0.0, 1.0],
        u_axis: vec![1.0, 0.0, 0.0],
        v_axis: vec![0.0, 1.0, 0.0],
        extent_u_m: 0.1,
        extent_v_m: 0.1,
        rows: 2,
        columns: 2,
    };
    value.validate().expect("example plane");
    value
}

fn vector3(values: &[f64], field: &str) -> Result<[f64; 3]> {
    let [x, y, z] = values else {
        return Err(invalid(field, "expected exactly three components"));
    };
    if [*x, *y, *z].iter().any(|value| !value.is_finite()) {
        return Err(invalid(field, "components must be finite"));
    }
    Ok([*x, *y, *z])
}

fn canonical_unit_vector(x: f64, y: f64, z: f64, field: &str) -> Result<UnitVector3> {
    let vector = UnitVector3::new(x, y, z)
        .map_err(|error| invalid("Plane", format!("{field}: {error:?}")))?;
    if vector
        .components()
        .iter()
        .zip([x, y, z])
        .any(|(actual, supplied)| actual.to_bits() != supplied.to_bits())
    {
        return Err(invalid(
            "Plane",
            format!("{field} must already be normalized"),
        ));
    }
    Ok(vector)
}

fn point_kind() -> Symbol {
    Symbol::qualified("interference", "point")
}

fn forward_plane_kind() -> Symbol {
    Symbol::qualified("interference", "forward-plane")
}
