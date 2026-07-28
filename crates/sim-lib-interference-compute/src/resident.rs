//! Resident result assembly over the canonical Tensor lowering.

use sim_kernel::Cx;
use sim_lib_numbers_tensor::{Tensor, TensorLocation};

use crate::{LoweringError, LoweringPlan, PreflightCheck};

pub(crate) struct ResidentExecution {
    pub(crate) real: Tensor,
    pub(crate) imaginary: Tensor,
    pub(crate) uploads: u64,
    pub(crate) submissions: u64,
    pub(crate) segments: u64,
}

pub(crate) fn execute_resident(
    cx: &mut Cx,
    plan: &LoweringPlan,
) -> Result<ResidentExecution, LoweringError> {
    let (mut tiles, uploads) = plan.execute_with_uploaded_inputs(cx)?;
    if tiles.len() != 1 {
        return Err(LoweringError::new(
            PreflightCheck::Execution,
            format!(
                "resident Study requires one admitted output tile, found {}",
                tiles.len()
            ),
        ));
    }
    let lowered = tiles.pop().expect("one resident tile was checked");
    let real = lowered.real().clone();
    let imaginary = lowered.imaginary().clone();
    let real_site = resident_site(&real, "real")?;
    let imaginary_site = resident_site(&imaginary, "imaginary")?;
    if real_site != imaginary_site {
        return Err(LoweringError::new(
            PreflightCheck::Execution,
            format!(
                "resident result components disagree on placement: {real_site} versus {imaginary_site}"
            ),
        ));
    }
    let submissions = lowered
        .submissions()
        .iter()
        .try_fold(0_u64, |total, evidence| {
            total
                .checked_add(u64::try_from(evidence.accepted).map_err(|_| {
                    LoweringError::new(
                        PreflightCheck::Execution,
                        "accepted submission count does not fit u64",
                    )
                })?)
                .ok_or_else(|| {
                    LoweringError::new(
                        PreflightCheck::Execution,
                        "accepted submission count overflowed u64",
                    )
                })
        })?;
    let segments = u64::try_from(lowered.tile().segments_per_tensor())
        .ok()
        .and_then(|segments| segments.checked_mul(2))
        .ok_or_else(|| {
            LoweringError::new(
                PreflightCheck::Execution,
                "final resident segment count overflowed u64",
            )
        })?;
    Ok(ResidentExecution {
        real,
        imaginary,
        uploads: u64::try_from(uploads).map_err(|_| {
            LoweringError::new(
                PreflightCheck::Execution,
                "resident upload count does not fit u64",
            )
        })?,
        submissions,
        segments,
    })
}

fn resident_site(
    tensor: &Tensor,
    component: &'static str,
) -> Result<sim_kernel::Symbol, LoweringError> {
    match tensor.location() {
        TensorLocation::Resident { site, .. } => Ok(site),
        TensorLocation::Host => Err(LoweringError::new(
            PreflightCheck::Execution,
            format!("provider returned host storage for final {component} component"),
        )),
    }
}
