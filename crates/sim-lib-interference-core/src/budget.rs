//! Checked work accounting and allocation-free request admission.

use std::fmt;

use crate::{
    InterferenceError, InterferenceProblem, SamplingCertificate, SamplingPlane, SamplingPolicy,
    SamplingThresholds,
};

const PHASOR_RESULT_BYTES_PER_CELL: u64 = 2 * size_of::<f64>() as u64;
const CERTIFICATE_STENCIL_POINTS_PER_CELL: u64 = 7;

/// A separately limited work or storage dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkMetric {
    /// Number of output cells.
    Cells,
    /// Number of source Green-function evaluations.
    EmitterEvaluations,
    /// Peak bytes reserved for host phasor components.
    HostBytes,
    /// Bytes in the two-component result.
    ResultBytes,
    /// Seven-point certificate stencil evaluations.
    CertificateStencilWork,
}

impl fmt::Display for WorkMetric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cells => "cells",
            Self::EmitterEvaluations => "emitter-evaluations",
            Self::HostBytes => "host-bytes",
            Self::ResultBytes => "result-bytes",
            Self::CertificateStencilWork => "certificate-stencil-work",
        })
    }
}

/// Allocation and evaluation counts known before a field solve starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkEstimate {
    /// Number of sampling cells.
    pub cells: u64,
    /// Number of coherent emitters.
    pub emitters: u64,
    /// Product of cells and emitters.
    pub emitter_evaluations: u64,
    /// Peak bytes for the reference host phasor result.
    pub host_bytes: u64,
    /// Bytes in the two `f64` result components.
    pub result_bytes: u64,
    /// Work for a seven-point certificate stencil at every cell.
    pub certificate_stencil_work: u64,
}

impl WorkEstimate {
    /// Computes every work dimension with checked integer arithmetic.
    ///
    /// This function allocates nothing and is also useful for testing or
    /// admitting decoded counts before a [`SamplingPlane`] is constructed.
    pub fn new(cells: u64, emitters: u64) -> Result<Self, InterferenceError> {
        let emitter_evaluations = checked_product(cells, emitters, WorkMetric::EmitterEvaluations)?;
        let result_bytes =
            checked_product(cells, PHASOR_RESULT_BYTES_PER_CELL, WorkMetric::ResultBytes)?;
        let host_bytes = result_bytes;
        let certificate_stencil_work = checked_product(
            cells,
            CERTIFICATE_STENCIL_POINTS_PER_CELL,
            WorkMetric::CertificateStencilWork,
        )?;
        Ok(Self {
            cells,
            emitters,
            emitter_evaluations,
            host_bytes,
            result_bytes,
            certificate_stencil_work,
        })
    }

    /// Computes a request estimate from checked domain records.
    pub fn for_request(
        problem: &InterferenceProblem,
        plane: &SamplingPlane,
    ) -> Result<Self, InterferenceError> {
        let cells = u64::try_from(plane.cell_count()).map_err(|_| {
            InterferenceError::WorkEstimateOverflow {
                metric: WorkMetric::Cells,
            }
        })?;
        let emitters = u64::try_from(problem.sources.len()).map_err(|_| {
            InterferenceError::WorkEstimateOverflow {
                metric: WorkMetric::EmitterEvaluations,
            }
        })?;
        Self::new(cells, emitters)
    }
}

/// Explicit upper bounds for every preflight work dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkBudget {
    /// Maximum output cells.
    pub max_cells: u64,
    /// Maximum source Green-function evaluations.
    pub max_emitter_evaluations: u64,
    /// Maximum host bytes.
    pub max_host_bytes: u64,
    /// Maximum result bytes.
    pub max_result_bytes: u64,
    /// Maximum certificate stencil evaluations.
    pub max_certificate_stencil_work: u64,
}

impl WorkBudget {
    /// Admits an estimate or names the first estimate and limit exceeded.
    ///
    /// Checks follow the field order, making diagnostics deterministic.
    pub fn admit(self, estimate: &WorkEstimate) -> Result<(), InterferenceError> {
        for (metric, requested, limit) in [
            (WorkMetric::Cells, estimate.cells, self.max_cells),
            (
                WorkMetric::EmitterEvaluations,
                estimate.emitter_evaluations,
                self.max_emitter_evaluations,
            ),
            (
                WorkMetric::HostBytes,
                estimate.host_bytes,
                self.max_host_bytes,
            ),
            (
                WorkMetric::ResultBytes,
                estimate.result_bytes,
                self.max_result_bytes,
            ),
            (
                WorkMetric::CertificateStencilWork,
                estimate.certificate_stencil_work,
                self.max_certificate_stencil_work,
            ),
        ] {
            if requested > limit {
                return Err(InterferenceError::WorkBudgetExceeded {
                    metric,
                    estimate: requested,
                    limit,
                });
            }
        }
        Ok(())
    }
}

impl Default for WorkBudget {
    fn default() -> Self {
        const MAX_CELLS: u64 = 4_096 * 4_096;
        Self {
            max_cells: MAX_CELLS,
            max_emitter_evaluations: 2_000_000_000,
            max_host_bytes: 1 << 30,
            max_result_bytes: 1 << 29,
            max_certificate_stencil_work: MAX_CELLS * CERTIFICATE_STENCIL_POINTS_PER_CELL,
        }
    }
}

/// Successful sampling and work evidence obtained before field allocation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RequestPreflight {
    /// Policy applied to the sampling certificate.
    pub sampling_policy: SamplingPolicy,
    /// Physical sampling measurements and thresholds.
    pub sampling_certificate: SamplingCertificate,
    /// Checked work and storage counts.
    pub work_estimate: WorkEstimate,
}

impl RequestPreflight {
    /// Classifies and budgets one request without allocating field storage.
    pub fn admit(
        problem: &InterferenceProblem,
        plane: &SamplingPlane,
        sampling_policy: SamplingPolicy,
        sampling_thresholds: SamplingThresholds,
        work_budget: WorkBudget,
    ) -> Result<Self, InterferenceError> {
        let work_estimate = WorkEstimate::for_request(problem, plane)?;
        work_budget.admit(&work_estimate)?;
        let sampling_certificate =
            SamplingCertificate::measure_with_thresholds(problem, plane, sampling_thresholds)?;
        sampling_policy.admit(&sampling_certificate)?;
        Ok(Self {
            sampling_policy,
            sampling_certificate,
            work_estimate,
        })
    }
}

fn checked_product(left: u64, right: u64, metric: WorkMetric) -> Result<u64, InterferenceError> {
    left.checked_mul(right)
        .ok_or(InterferenceError::WorkEstimateOverflow { metric })
}
