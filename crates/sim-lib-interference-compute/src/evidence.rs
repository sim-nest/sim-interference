//! Bounded, fail-closed physical-provider evidence.

use std::fmt;

use crate::DifferentialReport;

/// Number of same-profile executions required by the hardware contract.
pub const HARDWARE_DETERMINISM_REPEATS: usize = 100;

/// Whether a physical-provider observation can satisfy the hardware gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HardwareMeasurementResult {
    /// A physical adapter was measured and every contract passed.
    MeasuredPass,
    /// A physical adapter was measured and at least one contract failed.
    MeasuredFail,
    /// No qualifying physical adapter was measured.
    NotMeasured,
}

impl HardwareMeasurementResult {
    fn as_str(self) -> &'static str {
        match self {
            Self::MeasuredPass => "measured-pass",
            Self::MeasuredFail => "measured-fail",
            Self::NotMeasured => "not-measured",
        }
    }
}

/// Provider-independent measurements used to classify one hardware run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HardwareEvidenceMetrics {
    /// Number of admitted physical field tiles.
    pub tiles: usize,
    /// Number of resident final-component segments.
    pub segments: u64,
    /// Largest residual phase argument sent to Tensor trigonometry.
    pub max_abs_psi: f64,
    /// Intermediate host materializations.
    pub intermediate_materializations: u64,
    /// Final component host materializations.
    pub final_materializations: u64,
    /// Same-profile execution count.
    pub repeats: usize,
    /// Whether every repeat produced bit-identical component planes.
    pub deterministic: bool,
}

/// Sanitized differential, lifecycle, and determinism evidence for one profile.
#[derive(Clone, Debug, PartialEq)]
pub struct HardwareEvidenceReport {
    adapter_id: String,
    profile_id: String,
    tiles: usize,
    segments: u64,
    max_abs_psi: f64,
    max_component_abs: f64,
    max_phase_abs: f64,
    intermediate_materializations: u64,
    final_materializations: u64,
    repeats: usize,
    result: HardwareMeasurementResult,
}

impl HardwareEvidenceReport {
    /// Records that no qualifying physical adapter was measured.
    ///
    /// This result deliberately cannot satisfy [`Self::satisfies_hardware_gate`].
    pub fn not_measured(adapter_id: impl Into<String>, profile_id: impl Into<String>) -> Self {
        Self {
            adapter_id: nonempty_or_unavailable(adapter_id.into()),
            profile_id: nonempty_or_unavailable(profile_id.into()),
            tiles: 0,
            segments: 0,
            max_abs_psi: 0.0,
            max_component_abs: 0.0,
            max_phase_abs: 0.0,
            intermediate_materializations: 0,
            final_materializations: 0,
            repeats: 0,
            result: HardwareMeasurementResult::NotMeasured,
        }
    }

    /// Builds measured evidence from the shared differential report.
    pub fn measured(
        adapter_id: impl Into<String>,
        profile_id: impl Into<String>,
        metrics: HardwareEvidenceMetrics,
        differential: &DifferentialReport,
    ) -> Self {
        let lifecycle =
            metrics.intermediate_materializations == 0 && metrics.final_materializations == 2;
        let finite = metrics.max_abs_psi.is_finite()
            && differential.max_component_absolute_error().is_finite()
            && differential.max_phase_absolute_error().is_finite();
        let passed = differential.passed()
            && metrics.deterministic
            && lifecycle
            && finite
            && metrics.repeats == HARDWARE_DETERMINISM_REPEATS;
        Self {
            adapter_id: nonempty_or_unavailable(adapter_id.into()),
            profile_id: nonempty_or_unavailable(profile_id.into()),
            tiles: metrics.tiles,
            segments: metrics.segments,
            max_abs_psi: metrics.max_abs_psi,
            max_component_abs: differential.max_component_absolute_error(),
            max_phase_abs: differential.max_phase_absolute_error(),
            intermediate_materializations: metrics.intermediate_materializations,
            final_materializations: metrics.final_materializations,
            repeats: metrics.repeats,
            result: if passed {
                HardwareMeasurementResult::MeasuredPass
            } else {
                HardwareMeasurementResult::MeasuredFail
            },
        }
    }

    /// Returns the explicit measurement classification.
    pub fn result(&self) -> HardwareMeasurementResult {
        self.result
    }

    /// Returns true only for a qualifying 100-repeat measured pass.
    pub fn satisfies_hardware_gate(&self) -> bool {
        self.result == HardwareMeasurementResult::MeasuredPass
            && self.repeats == HARDWARE_DETERMINISM_REPEATS
    }
}

impl fmt::Display for HardwareEvidenceReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "provider=wgpu adapter={} profile={} dtype=f32 tiles={} segments={}",
            self.adapter_id, self.profile_id, self.tiles, self.segments
        )?;
        writeln!(
            formatter,
            "max_abs_psi={:.9e} max_component_abs={:.9e} max_phase_abs={:.9e}",
            self.max_abs_psi, self.max_component_abs, self.max_phase_abs
        )?;
        write!(
            formatter,
            "intermediate_materializations={} final_materializations={} repeats={}\nresult={}",
            self.intermediate_materializations,
            self.final_materializations,
            self.repeats,
            self.result.as_str()
        )
    }
}

fn nonempty_or_unavailable(value: String) -> String {
    if value.trim().is_empty() {
        "unavailable".to_owned()
    } else {
        value
    }
}
