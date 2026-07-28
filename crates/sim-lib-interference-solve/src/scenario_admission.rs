//! Allocation-free count and generated-identity admission for scenarios.

use sim_lib_interference_core::Emitter;

use crate::{ScenarioError, scenario::ScenarioLimits};

#[derive(Clone, Copy)]
pub(crate) struct IdPlan {
    prefix: &'static str,
    pub(crate) count: usize,
    width: usize,
}

impl IdPlan {
    pub(crate) fn admit(
        prefix: &'static str,
        count: usize,
        limits: ScenarioLimits,
    ) -> Result<Self, ScenarioError> {
        if count > limits.max_sources {
            return Err(ScenarioError::SourceLimitExceeded {
                requested: count,
                limit: limits.max_sources,
            });
        }
        let width = decimal_digits(count.saturating_sub(1));
        let longest =
            prefix
                .len()
                .checked_add(width)
                .ok_or(ScenarioError::GeneratedIdLimitExceeded {
                    requested: usize::MAX,
                    limit: limits.max_generated_id_bytes,
                })?;
        if longest > limits.max_generated_id_bytes {
            return Err(ScenarioError::GeneratedIdLimitExceeded {
                requested: longest,
                limit: limits.max_generated_id_bytes,
            });
        }
        let total = longest
            .checked_mul(count)
            .ok_or(ScenarioError::TotalIdLimitExceeded {
                requested: usize::MAX,
                limit: limits.max_total_id_bytes,
            })?;
        if total > limits.max_total_id_bytes {
            return Err(ScenarioError::TotalIdLimitExceeded {
                requested: total,
                limit: limits.max_total_id_bytes,
            });
        }
        Ok(Self {
            prefix,
            count,
            width,
        })
    }

    pub(crate) fn id(self, index: usize) -> String {
        format!("{}{index:0width$}", self.prefix, width = self.width)
    }
}

pub(crate) fn require_limit(
    name: &'static str,
    value: usize,
    absolute_maximum: usize,
) -> Result<(), ScenarioError> {
    if value == 0 || value > absolute_maximum {
        Err(ScenarioError::InvalidLimit {
            name,
            value,
            absolute_maximum,
        })
    } else {
        Ok(())
    }
}

pub(crate) fn require_dimension(name: &'static str, value: usize) -> Result<(), ScenarioError> {
    (value > 0)
        .then_some(())
        .ok_or(ScenarioError::ZeroElementDimension { name })
}

pub(crate) fn allocate_sources(count: usize) -> Result<Vec<Emitter>, ScenarioError> {
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(count)
        .map_err(|_| ScenarioError::AllocationFailed { sources: count })?;
    Ok(sources)
}

pub(crate) fn sources_from(
    plan: IdPlan,
    mut emitter: impl FnMut(usize, String) -> Emitter,
) -> Result<Vec<Emitter>, ScenarioError> {
    let mut sources = allocate_sources(plan.count)?;
    for index in 0..plan.count {
        sources.push(emitter(index, plan.id(index)));
    }
    Ok(sources)
}

pub(crate) fn sources_from_fallible(
    plan: IdPlan,
    mut emitter: impl FnMut(usize, String) -> Result<Emitter, ScenarioError>,
) -> Result<Vec<Emitter>, ScenarioError> {
    let mut sources = allocate_sources(plan.count)?;
    for index in 0..plan.count {
        sources.push(emitter(index, plan.id(index))?);
    }
    Ok(sources)
}

fn decimal_digits(mut value: usize) -> usize {
    let mut digits = 1;
    while value >= 10 {
        value /= 10;
        digits += 1;
    }
    digits
}
