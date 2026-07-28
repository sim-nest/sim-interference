//! Fail-closed reconstruction for sampling certificate Citizens.

use sim_kernel::{Cx, Result, Value};
use sim_lib_interference_core::{SamplingCertificate, SamplingThresholds, SamplingVerdict};

use crate::{
    SamplingCertificateDescriptor,
    citizen::{RecordCitizenSpec, encode_field, invalid, next_field},
    evidence::{sample_sampling, verdict_symbol},
};

impl SamplingCertificateDescriptor {
    /// Reconstructs and revalidates all sampling measurements.
    pub fn to_certificate(&self) -> Result<SamplingCertificate> {
        let thresholds = SamplingThresholds::new(
            self.resolved_min_samples_per_wavelength,
            self.marginal_min_samples_per_wavelength,
            self.resolved_max_envelope_fraction_per_cell,
            self.marginal_max_envelope_fraction_per_cell,
        )
        .map_err(|error| invalid("SamplingCertificate", format!("{error:?}")))?;
        for (name, value, positive) in [
            ("wavelength-m", self.wavelength_m, true),
            (
                "samples-per-wavelength-u",
                self.samples_per_wavelength_u,
                true,
            ),
            (
                "samples-per-wavelength-v",
                self.samples_per_wavelength_v,
                true,
            ),
            (
                "samples-per-power-fringe-u",
                self.samples_per_power_fringe_u,
                true,
            ),
            (
                "samples-per-power-fringe-v",
                self.samples_per_power_fringe_v,
                true,
            ),
            (
                "max-envelope-fraction-per-cell",
                self.max_envelope_fraction_per_cell,
                false,
            ),
        ] {
            if !value.is_finite() || (positive && value <= 0.0) || (!positive && value < 0.0) {
                return Err(invalid(
                    "SamplingCertificate",
                    format!("{name} is outside its finite physical range"),
                ));
            }
        }
        if self
            .nearest_point_source_distance_m
            .is_some_and(|distance| !distance.is_finite() || distance <= 0.0)
        {
            return Err(invalid(
                "SamplingCertificate",
                "nearest point-source distance must be finite and positive",
            ));
        }
        if self.samples_per_power_fringe_u.to_bits()
            != (self.samples_per_wavelength_u / 2.0).to_bits()
            || self.samples_per_power_fringe_v.to_bits()
                != (self.samples_per_wavelength_v / 2.0).to_bits()
        {
            return Err(invalid(
                "SamplingCertificate",
                "power-fringe samples must be exactly half the carrier samples",
            ));
        }
        if self.nearest_point_source_distance_m.is_none()
            && self.max_envelope_fraction_per_cell != 0.0
        {
            return Err(invalid(
                "SamplingCertificate",
                "plane-only evidence must have zero envelope change",
            ));
        }
        let minimum = self
            .samples_per_wavelength_u
            .min(self.samples_per_wavelength_v);
        let expected = if minimum >= thresholds.resolved_min_samples_per_wavelength
            && self.max_envelope_fraction_per_cell
                <= thresholds.resolved_max_envelope_fraction_per_cell
        {
            SamplingVerdict::Resolved
        } else if minimum >= thresholds.marginal_min_samples_per_wavelength
            && self.max_envelope_fraction_per_cell
                <= thresholds.marginal_max_envelope_fraction_per_cell
        {
            SamplingVerdict::Marginal
        } else {
            SamplingVerdict::Aliased
        };
        if self.verdict != verdict_symbol(expected) {
            return Err(invalid(
                "SamplingCertificate",
                "verdict does not follow the recorded thresholds and measurements",
            ));
        }
        Ok(SamplingCertificate {
            thresholds,
            wavelength_m: self.wavelength_m,
            samples_per_wavelength_u: self.samples_per_wavelength_u,
            samples_per_wavelength_v: self.samples_per_wavelength_v,
            samples_per_power_fringe_u: self.samples_per_power_fringe_u,
            samples_per_power_fringe_v: self.samples_per_power_fringe_v,
            nearest_point_source_distance_m: self.nearest_point_source_distance_m,
            max_envelope_fraction_per_cell: self.max_envelope_fraction_per_cell,
            verdict: expected,
        })
    }
}

impl RecordCitizenSpec for SamplingCertificateDescriptor {
    const FIELDS: &'static [&'static str] = &[
        "resolved-min-samples-per-wavelength",
        "marginal-min-samples-per-wavelength",
        "resolved-max-envelope-fraction-per-cell",
        "marginal-max-envelope-fraction-per-cell",
        "wavelength-m",
        "samples-per-wavelength-u",
        "samples-per-wavelength-v",
        "samples-per-power-fringe-u",
        "samples-per-power-fringe-v",
        "nearest-point-source-distance-m",
        "max-envelope-fraction-per-cell",
        "verdict",
    ];

    fn encode_fields(&self, _cx: &mut Cx) -> Result<Vec<sim_kernel::Expr>> {
        Ok(vec![
            encode_field(&self.resolved_min_samples_per_wavelength),
            encode_field(&self.marginal_min_samples_per_wavelength),
            encode_field(&self.resolved_max_envelope_fraction_per_cell),
            encode_field(&self.marginal_max_envelope_fraction_per_cell),
            encode_field(&self.wavelength_m),
            encode_field(&self.samples_per_wavelength_u),
            encode_field(&self.samples_per_wavelength_v),
            encode_field(&self.samples_per_power_fringe_u),
            encode_field(&self.samples_per_power_fringe_v),
            encode_field(&self.nearest_point_source_distance_m),
            encode_field(&self.max_envelope_fraction_per_cell),
            encode_field(&self.verdict),
        ])
    }

    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self> {
        let mut fields = fields.into_iter();
        let value = Self {
            resolved_min_samples_per_wavelength: next_field(
                cx,
                &mut fields,
                "resolved-min-samples-per-wavelength",
            )?,
            marginal_min_samples_per_wavelength: next_field(
                cx,
                &mut fields,
                "marginal-min-samples-per-wavelength",
            )?,
            resolved_max_envelope_fraction_per_cell: next_field(
                cx,
                &mut fields,
                "resolved-max-envelope-fraction-per-cell",
            )?,
            marginal_max_envelope_fraction_per_cell: next_field(
                cx,
                &mut fields,
                "marginal-max-envelope-fraction-per-cell",
            )?,
            wavelength_m: next_field(cx, &mut fields, "wavelength-m")?,
            samples_per_wavelength_u: next_field(cx, &mut fields, "samples-per-wavelength-u")?,
            samples_per_wavelength_v: next_field(cx, &mut fields, "samples-per-wavelength-v")?,
            samples_per_power_fringe_u: next_field(cx, &mut fields, "samples-per-power-fringe-u")?,
            samples_per_power_fringe_v: next_field(cx, &mut fields, "samples-per-power-fringe-v")?,
            nearest_point_source_distance_m: next_field(
                cx,
                &mut fields,
                "nearest-point-source-distance-m",
            )?,
            max_envelope_fraction_per_cell: next_field(
                cx,
                &mut fields,
                "max-envelope-fraction-per-cell",
            )?,
            verdict: next_field(cx, &mut fields, "verdict")?,
        };
        value.validate()?;
        Ok(value)
    }

    fn example() -> Self {
        sample_sampling()
    }

    fn validate(&self) -> Result<()> {
        self.to_certificate().map(|_| ())
    }
}

impl_record_citizen!(
    SamplingCertificateDescriptor,
    "interference/SamplingCertificate",
    12
);
