//! Certified composition of independently solved frequency components.
//!
//! A [`ToneStudy`] owns one complete coherent problem, the exact physical
//! sampling plane, its solved phasor field, and the solver evidence that
//! certifies the field. A [`MultiToneStudy`] accepts only those sealed studies
//! and requires exact plane equality before any cellwise observation. Unlike
//! frequencies are therefore never represented as one phasor field.

use std::f64::consts::TAU;

use sim_lib_interference_core::{Hertz, InterferenceProblem, SamplingCertificate, SamplingPlane};

use crate::{
    HostPhasorField, MultiToneError, ReferencePhasorSolver, SolveEvidence, complex::CompensatedSum,
};

/// A valid cross-frequency scalar observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ToneCombination {
    /// `sum(weight_i * |U_i|^2)` for mutually incoherent detection.
    ///
    /// This is a normalized squared-magnitude proxy, not impedance-derived
    /// physical intensity.
    IncoherentMagnitudeSquared,
    /// `sum(weight_i * Re{U_i * exp(-i * omega_i * seconds)})`.
    Instant {
        /// One finite shared time in seconds.
        seconds: f64,
    },
}

/// Spatial and temporal sampling requirements for one frequency set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiToneSamplingRequirements {
    highest_frequency: Hertz,
    spatial_certificate: SamplingCertificate,
    minimum_temporal_samples_per_second: f64,
    maximum_temporal_step_seconds: f64,
}

impl MultiToneSamplingRequirements {
    /// Returns the highest component frequency.
    pub fn highest_frequency(self) -> Hertz {
        self.highest_frequency
    }

    /// Returns the highest-frequency component's spatial certificate.
    ///
    /// Every lower-frequency component certificate remains available
    /// separately through [`ToneCertificate::sampling_certificate`].
    pub fn spatial_certificate(self) -> SamplingCertificate {
        self.spatial_certificate
    }

    /// Returns the Nyquist floor `2 * highest_frequency`.
    pub fn minimum_temporal_samples_per_second(self) -> f64 {
        self.minimum_temporal_samples_per_second
    }

    /// Returns the largest Nyquist time step `1 / (2 * highest_frequency)`.
    pub fn maximum_temporal_step_seconds(self) -> f64 {
        self.maximum_temporal_step_seconds
    }
}

/// Immutable provenance for one component of a multi-tone result.
#[derive(Clone, Debug, PartialEq)]
pub struct ToneCertificate {
    frequency: Hertz,
    weight: f64,
    solve_evidence: SolveEvidence,
}

impl ToneCertificate {
    fn from_study(study: &ToneStudy) -> Self {
        Self {
            frequency: study.frequency(),
            weight: study.weight(),
            solve_evidence: study.evidence().clone(),
        }
    }

    /// Returns the component frequency.
    pub fn frequency(&self) -> Hertz {
        self.frequency
    }

    /// Returns the component's finite positive composition weight.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Borrows the complete evidence from the independent coherent solve.
    pub fn solve_evidence(&self) -> &SolveEvidence {
        &self.solve_evidence
    }

    /// Returns the component's original spatial sampling certificate.
    pub fn sampling_certificate(&self) -> SamplingCertificate {
        self.solve_evidence.preflight().sampling_certificate
    }
}

/// Inseparable provenance for one complete multi-tone scalar result.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiToneCertificate {
    plane: SamplingPlane,
    combination: ToneCombination,
    sampling_requirements: MultiToneSamplingRequirements,
    components: Vec<ToneCertificate>,
}

impl MultiToneCertificate {
    /// Returns the exact shared physical plane.
    pub fn plane(&self) -> SamplingPlane {
        self.plane
    }

    /// Returns the scalar combination rule.
    pub fn combination(&self) -> ToneCombination {
        self.combination
    }

    /// Returns the highest-frequency spatial and temporal requirements.
    pub fn sampling_requirements(&self) -> MultiToneSamplingRequirements {
        self.sampling_requirements
    }

    /// Borrows every component's frequency, weight, and solve evidence.
    pub fn components(&self) -> &[ToneCertificate] {
        &self.components
    }
}

/// A complete scalar observation of one certified multi-tone study.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiToneProjection {
    rows: usize,
    columns: usize,
    samples: Vec<f64>,
    certificate: MultiToneCertificate,
}

impl MultiToneProjection {
    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the number of finite row-major scalar samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns whether the result contains no cells.
    ///
    /// A valid multi-tone projection is never empty because its checked
    /// sampling plane has non-zero dimensions.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Borrows the finite row-major scalar samples.
    pub fn samples(&self) -> &[f64] {
        &self.samples
    }

    /// Returns one scalar sample.
    pub fn cell(&self, row: usize, column: usize) -> Option<f64> {
        let index = row.checked_mul(self.columns)?.checked_add(column)?;
        (row < self.rows && column < self.columns).then(|| self.samples[index])
    }

    /// Returns the exact observation rule used for this result.
    pub fn combination(&self) -> ToneCombination {
        self.certificate.combination()
    }

    /// Borrows the set requirements and every component solve certificate.
    pub fn certificate(&self) -> &MultiToneCertificate {
        &self.certificate
    }
}

/// One positive-weight, independently certified coherent tone.
#[derive(Clone, Debug, PartialEq)]
pub struct ToneStudy {
    problem: InterferenceProblem,
    plane: SamplingPlane,
    weight: f64,
    field: HostPhasorField,
    evidence: SolveEvidence,
}

impl ToneStudy {
    /// Solves and seals one weighted frequency component.
    ///
    /// The weight is checked before propagation work. The returned study owns
    /// the exact problem and plane used by the solver, so its field cannot be
    /// separated from or paired with different sampling evidence.
    pub fn solve(
        problem: InterferenceProblem,
        plane: SamplingPlane,
        weight: f64,
        solver: ReferencePhasorSolver,
    ) -> Result<Self, MultiToneError> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(MultiToneError::InvalidWeight {
                frequency_hz: problem.frequency.get(),
                weight,
            });
        }
        let (field, evidence) =
            solver
                .solve(&problem, &plane)
                .map_err(|cause| MultiToneError::ComponentSolve {
                    frequency_hz: problem.frequency.get(),
                    cause: Box::new(cause),
                })?;
        Ok(Self {
            problem,
            plane,
            weight,
            field,
            evidence,
        })
    }

    /// Returns the exact coherent problem solved for this tone.
    pub fn problem(&self) -> &InterferenceProblem {
        &self.problem
    }

    /// Returns this tone's frequency.
    pub fn frequency(&self) -> Hertz {
        self.problem.frequency
    }

    /// Returns this tone's finite, strictly positive composition weight.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Returns the exact physical plane on which this tone was solved.
    pub fn plane(&self) -> SamplingPlane {
        self.plane
    }

    /// Borrows this tone's complete coherent phasor field.
    pub fn field(&self) -> &HostPhasorField {
        &self.field
    }

    /// Borrows the immutable evidence produced by this tone's solve.
    pub fn evidence(&self) -> &SolveEvidence {
        &self.evidence
    }

    /// Returns this component's sampling certificate.
    pub fn sampling_certificate(&self) -> SamplingCertificate {
        self.evidence.preflight().sampling_certificate
    }
}

/// A non-empty set of certified tones sharing one exact physical plane.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiToneStudy {
    tones: Vec<ToneStudy>,
    plane: SamplingPlane,
    sampling_requirements: MultiToneSamplingRequirements,
}

impl MultiToneStudy {
    /// Validates independently certified tones for cellwise composition.
    ///
    /// Geometry equality is exact: even equal dimensions are insufficient
    /// when origins, axes, extents, or therefore physical sample centres
    /// differ.
    pub fn new(mut tones: Vec<ToneStudy>) -> Result<Self, MultiToneError> {
        let Some(first) = tones.first() else {
            return Err(MultiToneError::EmptyStudy);
        };
        let plane = first.plane();
        tones.sort_by(|left, right| left.frequency().get().total_cmp(&right.frequency().get()));
        for pair in tones.windows(2) {
            if pair[0].frequency() == pair[1].frequency() {
                return Err(MultiToneError::DuplicateFrequency {
                    frequency_hz: pair[0].frequency().get(),
                });
            }
        }
        for tone in &tones {
            if tone.plane() != plane {
                return Err(MultiToneError::MismatchedPlane {
                    frequency_hz: tone.frequency().get(),
                    expected: Box::new(plane),
                    actual: Box::new(tone.plane()),
                });
            }
        }
        let highest = tones.last().expect("non-empty study checked above");
        let minimum_temporal_samples_per_second = 2.0 * highest.frequency().get();
        if !minimum_temporal_samples_per_second.is_finite() {
            return Err(MultiToneError::NonFiniteTemporalSamplingRequirement {
                highest_frequency_hz: highest.frequency().get(),
                samples_per_second: minimum_temporal_samples_per_second,
            });
        }
        let sampling_requirements = MultiToneSamplingRequirements {
            highest_frequency: highest.frequency(),
            spatial_certificate: highest.sampling_certificate(),
            minimum_temporal_samples_per_second,
            maximum_temporal_step_seconds: 1.0 / minimum_temporal_samples_per_second,
        };
        Ok(Self {
            tones,
            plane,
            sampling_requirements,
        })
    }

    /// Borrows the certified component studies.
    pub fn tones(&self) -> &[ToneStudy] {
        &self.tones
    }

    /// Returns the exact physical plane shared by every component.
    pub fn plane(&self) -> SamplingPlane {
        self.plane
    }

    /// Returns requirements derived from the highest component frequency.
    pub fn sampling_requirements(&self) -> MultiToneSamplingRequirements {
        self.sampling_requirements
    }

    /// Combines independently solved tones only after projecting each one to
    /// a real scalar at the requested shared time or detection rule.
    ///
    /// This method never constructs or adds cross-frequency complex values.
    /// Tones are traversed in canonical ascending-frequency order and their
    /// scalar contributions are accumulated with Neumaier compensation.
    pub fn combine(
        &self,
        combination: ToneCombination,
    ) -> Result<MultiToneProjection, MultiToneError> {
        if let ToneCombination::Instant { seconds } = combination
            && !seconds.is_finite()
        {
            return Err(MultiToneError::InvalidSeconds { seconds });
        }

        let cells = self.plane.cell_count();
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(cells)
            .map_err(|_| MultiToneError::AllocationFailed { cells })?;
        for index in 0..cells {
            let row = index / self.plane.columns();
            let column = index % self.plane.columns();
            let mut accumulation = CompensatedSum::default();
            for tone in &self.tones {
                let value = component_scalar(tone, index, combination)?;
                let contribution = tone.weight() * value;
                if !contribution.is_finite() {
                    return Err(MultiToneError::NonFiniteContribution {
                        frequency_hz: tone.frequency().get(),
                        row,
                        column,
                        value: contribution,
                    });
                }
                accumulation.add(contribution);
                if !accumulation.is_finite() {
                    return Err(MultiToneError::NonFiniteAccumulation {
                        row,
                        column,
                        value: accumulation.total(),
                    });
                }
            }
            samples.push(accumulation.total());
        }
        let mut components = Vec::new();
        components
            .try_reserve_exact(self.tones.len())
            .map_err(|_| MultiToneError::CertificateAllocationFailed {
                tones: self.tones.len(),
            })?;
        components.extend(self.tones.iter().map(ToneCertificate::from_study));
        Ok(MultiToneProjection {
            rows: self.plane.rows(),
            columns: self.plane.columns(),
            samples,
            certificate: MultiToneCertificate {
                plane: self.plane,
                combination,
                sampling_requirements: self.sampling_requirements,
                components,
            },
        })
    }
}

fn component_scalar(
    tone: &ToneStudy,
    index: usize,
    combination: ToneCombination,
) -> Result<f64, MultiToneError> {
    let real = tone.field().real()[index];
    let imaginary = tone.field().imaginary()[index];
    match combination {
        ToneCombination::IncoherentMagnitudeSquared => {
            Ok(real.mul_add(real, imaginary * imaginary))
        }
        ToneCombination::Instant { seconds: 0.0 } => Ok(real),
        ToneCombination::Instant { seconds } => {
            let angular_time = TAU * tone.frequency().get() * seconds;
            if !angular_time.is_finite() {
                return Err(MultiToneError::NonFiniteAngularTime {
                    frequency_hz: tone.frequency().get(),
                    seconds,
                    angular_time,
                });
            }
            Ok(real * angular_time.cos() + imaginary * angular_time.sin())
        }
    }
}

#[cfg(test)]
#[path = "multitone_tests.rs"]
mod tests;
