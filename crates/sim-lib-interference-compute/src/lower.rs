//! Complete preflight followed by canonical Tensor request composition.

use std::sync::Arc;

use sim_kernel::{Cx, Symbol, Value};
use sim_lib_interference_core::{InterferenceProblem, SamplingPlane};
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, SubmissionEvidence, Tensor, TensorExecution, TensorExecutor,
    TensorExecutorCard, TensorMeta, TensorOp, TensorRequest, TypedTensorStorage,
    active_tensor_executor, add_op_symbol, cos_op_symbol, div_op_symbol, domains, exp_op_symbol,
    mul_op_symbol, sin_op_symbol, sqrt_op_symbol, sub_op_symbol,
};

use crate::{
    LoweringError, PhaseBudget, PlaneTile, PlaneTileConstants, PointTileConstants, PreflightCheck,
    SourcePhaseEstimate, SourceTileConstants, TilePlan, TileProfile,
    constants::prepare_source_constants,
    coordinates::{prepare_local_coordinates, scalar},
    preflight::{finite_f32, finite_nonzero_f32, validate_executor},
};

struct PreparedTile {
    tile: PlaneTile,
    local: [Tensor; 3],
    sources: Vec<SourceTileConstants>,
}

/// One completed tile with separate canonical real and imaginary f32 Tensors.
#[derive(Clone)]
pub struct LoweredTile {
    tile: PlaneTile,
    real: Tensor,
    imaginary: Tensor,
    submissions: Vec<SubmissionEvidence>,
}

impl LoweredTile {
    /// Returns the global placement and physical center of this tile.
    pub fn tile(&self) -> PlaneTile {
        self.tile
    }

    /// Returns the real component Tensor with shape `[tile.rows, tile.columns]`.
    pub fn real(&self) -> &Tensor {
        &self.real
    }

    /// Returns the imaginary component Tensor with shape `[tile.rows, tile.columns]`.
    pub fn imaginary(&self) -> &Tensor {
        &self.imaginary
    }

    /// Returns executor flush evidence in deterministic submission order.
    pub fn submissions(&self) -> &[SubmissionEvidence] {
        &self.submissions
    }
}

/// A completely admitted lowering bound to one canonical Tensor executor.
///
/// Construction performs all provider, geometry, shape, allocation, and
/// accuracy checks and creates every local-coordinate Tensor. Calling
/// [`execute`](Self::execute) never selects a different executor.
pub struct LoweringPlan {
    executor: Arc<dyn TensorExecutor>,
    executor_card: TensorExecutorCard,
    nil_attributes: Value,
    tile_plan: TilePlan,
    prepared: Vec<PreparedTile>,
    max_phase_estimate: SourcePhaseEstimate,
    phase_budget: PhaseBudget,
    wavenumber: f32,
    attenuation: f32,
}

impl LoweringPlan {
    /// Preflights a complete request against the active environment executor.
    ///
    /// When the environment has no active executor, the canonical CPU executor
    /// is captured. No Tensor request is submitted by this constructor.
    pub fn preflight(
        cx: &mut Cx,
        problem: &InterferenceProblem,
        plane: SamplingPlane,
        budget: PhaseBudget,
        profile: TileProfile,
    ) -> Result<Self, LoweringError> {
        let executor =
            active_tensor_executor(cx).unwrap_or_else(|| Arc::new(CpuTensorExecutor::new()));
        Self::preflight_with_executor(cx, problem, plane, budget, profile, executor)
    }

    pub(crate) fn preflight_with_executor(
        cx: &mut Cx,
        problem: &InterferenceProblem,
        plane: SamplingPlane,
        budget: PhaseBudget,
        profile: TileProfile,
        executor: Arc<dyn TensorExecutor>,
    ) -> Result<Self, LoweringError> {
        budget.validate()?;
        profile.validate()?;
        let executor_card = executor.card();
        validate_executor(&executor_card)?;

        let wave = problem.wavenumber();
        let wavenumber = finite_nonzero_f32("real wavenumber", wave.real_radians_per_metre())?;
        let attenuation = finite_f32("attenuation coefficient", wave.imaginary_nepers_per_metre())?;
        let tile_plan = TilePlan::new(plane, wave.real_radians_per_metre(), budget, profile)?;
        let nil_attributes = cx.factory().nil().map_err(|error| {
            LoweringError::new(
                PreflightCheck::Shape,
                format!("cannot construct Tensor operation attributes: {error}"),
            )
        })?;
        let mut prepared = Vec::with_capacity(tile_plan.tiles().len());
        let mut max_phase_estimate = SourcePhaseEstimate::default();
        for tile in tile_plan.tiles().iter().copied() {
            let local = prepare_local_coordinates(plane, tile)?;
            let mut sources = Vec::with_capacity(problem.sources.len());
            for source in &problem.sources {
                let constants = prepare_source_constants(
                    problem,
                    source,
                    tile.center(),
                    &local.exact,
                    &local.lowered,
                    budget,
                )?;
                merge_estimate(&mut max_phase_estimate, constants.estimate());
                sources.push(constants);
            }
            prepared.push(PreparedTile {
                tile,
                local: local.tensors,
                sources,
            });
        }
        Ok(Self {
            executor,
            executor_card,
            nil_attributes,
            tile_plan,
            prepared,
            max_phase_estimate,
            phase_budget: budget,
            wavenumber,
            attenuation,
        })
    }

    /// Returns the captured executor descriptor.
    pub fn executor_card(&self) -> &TensorExecutorCard {
        &self.executor_card
    }

    /// Returns the complete tile plan.
    pub fn tile_plan(&self) -> &TilePlan {
        &self.tile_plan
    }

    /// Returns the maximum estimate across every source and tile.
    pub fn max_phase_estimate(&self) -> SourcePhaseEstimate {
        self.max_phase_estimate
    }

    /// Returns the admitted residual-phase and predicted-error limits.
    pub fn phase_budget(&self) -> PhaseBudget {
        self.phase_budget
    }

    /// Returns the prepared source constants for one tile.
    pub fn source_constants(&self, tile_index: usize) -> Option<&[SourceTileConstants]> {
        self.prepared
            .get(tile_index)
            .map(|prepared| prepared.sources.as_slice())
    }

    /// Executes every prepared tile through the captured executor.
    ///
    /// Provider errors after accepted work are returned as execution failures;
    /// this method never restarts a tile on another executor.
    pub fn execute(&self, cx: &mut Cx) -> Result<Vec<LoweredTile>, LoweringError> {
        let mut lowered = Vec::with_capacity(self.prepared.len());
        for prepared in &self.prepared {
            let mut source_fields = Vec::with_capacity(prepared.sources.len());
            let mut submissions = Vec::new();
            for source in &prepared.sources {
                source_fields.push(self.lower_source(cx, prepared, *source)?);
                submissions.push(self.flush()?);
            }
            let (real, imaginary) =
                self.pairwise_accumulate(cx, prepared.tile, source_fields, &mut submissions)?;
            lowered.push(LoweredTile {
                tile: prepared.tile,
                real,
                imaginary,
                submissions,
            });
        }
        Ok(lowered)
    }

    pub(crate) fn execute_with_uploaded_inputs(
        &self,
        cx: &mut Cx,
    ) -> Result<(Vec<LoweredTile>, usize), LoweringError> {
        let mut lowered = Vec::with_capacity(self.prepared.len());
        let mut uploads = 0_usize;
        for prepared in &self.prepared {
            let zero = Tensor::from_storage(
                prepared.tile.shape().to_vec(),
                domains::f32(),
                Arc::new(TypedTensorStorage::<f32>::new(vec![
                    0.0;
                    prepared.tile.rows()
                        * prepared
                            .tile
                            .columns()
                ])),
            )
            .map_err(|error| {
                LoweringError::new(
                    PreflightCheck::Execution,
                    format!("cannot construct resident-upload zero Tensor: {error}"),
                )
            })?;
            let local = [
                self.binary(
                    cx,
                    add_op_symbol(),
                    &prepared.local[0],
                    &zero,
                    &prepared.tile.shape(),
                )?,
                self.binary(
                    cx,
                    add_op_symbol(),
                    &prepared.local[1],
                    &zero,
                    &prepared.tile.shape(),
                )?,
                self.binary(
                    cx,
                    add_op_symbol(),
                    &prepared.local[2],
                    &zero,
                    &prepared.tile.shape(),
                )?,
            ];
            uploads = uploads.checked_add(local.len()).ok_or_else(|| {
                LoweringError::new(
                    PreflightCheck::Execution,
                    "resident input upload count overflowed usize",
                )
            })?;
            let uploaded = PreparedTile {
                tile: prepared.tile,
                local,
                sources: prepared.sources.clone(),
            };
            let mut source_fields = Vec::with_capacity(uploaded.sources.len());
            let mut submissions = vec![self.flush()?];
            for source in &uploaded.sources {
                source_fields.push(self.lower_source(cx, &uploaded, *source)?);
                submissions.push(self.flush()?);
            }
            let (real, imaginary) =
                self.pairwise_accumulate(cx, uploaded.tile, source_fields, &mut submissions)?;
            lowered.push(LoweredTile {
                tile: uploaded.tile,
                real,
                imaginary,
                submissions,
            });
        }
        Ok((lowered, uploads))
    }

    fn lower_source(
        &self,
        cx: &mut Cx,
        tile: &PreparedTile,
        source: SourceTileConstants,
    ) -> Result<(Tensor, Tensor), LoweringError> {
        let delta = match source {
            SourceTileConstants::Point { constants, .. } => {
                self.point_delta(cx, tile, constants)?
            }
            SourceTileConstants::ForwardPlane { constants, .. } => {
                self.plane_delta(cx, tile, constants)?
            }
        };
        let shape = tile.tile.shape().to_vec();
        let psi = self.binary(
            cx,
            mul_op_symbol(),
            &delta,
            &scalar(self.wavenumber)?,
            &shape,
        )?;
        let phase_sin = self.unary(cx, sin_op_symbol(), &psi, &shape)?;
        let phase_cos = self.unary(cx, cos_op_symbol(), &psi, &shape)?;
        let (anchor_cos, anchor_sin, gain0) = match source {
            SourceTileConstants::Point { constants, .. } => {
                (constants.phase_cos, constants.phase_sin, constants.gain0)
            }
            SourceTileConstants::ForwardPlane { constants, .. } => {
                (constants.phase_cos, constants.phase_sin, constants.gain0)
            }
        };
        let cos_cos = self.binary(
            cx,
            mul_op_symbol(),
            &phase_cos,
            &scalar(anchor_cos)?,
            &shape,
        )?;
        let sin_sin = self.binary(
            cx,
            mul_op_symbol(),
            &phase_sin,
            &scalar(anchor_sin)?,
            &shape,
        )?;
        let real_phase = self.binary(cx, sub_op_symbol(), &cos_cos, &sin_sin, &shape)?;
        let sin_cos = self.binary(
            cx,
            mul_op_symbol(),
            &phase_sin,
            &scalar(anchor_cos)?,
            &shape,
        )?;
        let cos_sin = self.binary(
            cx,
            mul_op_symbol(),
            &phase_cos,
            &scalar(anchor_sin)?,
            &shape,
        )?;
        let imaginary_phase = self.binary(cx, add_op_symbol(), &sin_cos, &cos_sin, &shape)?;
        let attenuation_argument = self.binary(
            cx,
            mul_op_symbol(),
            &delta,
            &scalar(-self.attenuation)?,
            &shape,
        )?;
        let attenuation = self.unary(cx, exp_op_symbol(), &attenuation_argument, &shape)?;
        let gain = match source {
            SourceTileConstants::Point { constants, .. } => {
                let ratio_offset =
                    self.binary(cx, mul_op_symbol(), &delta, &scalar(constants.rho)?, &shape)?;
                let ratio_denominator =
                    self.binary(cx, add_op_symbol(), &scalar(1.0)?, &ratio_offset, &shape)?;
                let center_scaled =
                    self.binary(cx, mul_op_symbol(), &attenuation, &scalar(gain0)?, &shape)?;
                self.binary(
                    cx,
                    div_op_symbol(),
                    &center_scaled,
                    &ratio_denominator,
                    &shape,
                )?
            }
            SourceTileConstants::ForwardPlane { .. } => {
                self.binary(cx, mul_op_symbol(), &attenuation, &scalar(gain0)?, &shape)?
            }
        };
        Ok((
            self.binary(cx, mul_op_symbol(), &gain, &real_phase, &shape)?,
            self.binary(cx, mul_op_symbol(), &gain, &imaginary_phase, &shape)?,
        ))
    }

    fn point_delta(
        &self,
        cx: &mut Cx,
        tile: &PreparedTile,
        constants: PointTileConstants,
    ) -> Result<Tensor, LoweringError> {
        let shape = tile.tile.shape().to_vec();
        let a = self.dot_local(cx, tile, constants.n0)?;
        let b = self.dot_local_tensors(cx, tile)?;
        let two_a = self.binary(cx, mul_op_symbol(), &a, &scalar(2.0)?, &shape)?;
        let two_a_rho =
            self.binary(cx, mul_op_symbol(), &two_a, &scalar(constants.rho)?, &shape)?;
        let b_rho_squared = self.binary(
            cx,
            mul_op_symbol(),
            &b,
            &scalar(constants.rho * constants.rho)?,
            &shape,
        )?;
        let z = self.binary(cx, add_op_symbol(), &two_a_rho, &b_rho_squared, &shape)?;
        let b_rho = self.binary(cx, mul_op_symbol(), &b, &scalar(constants.rho)?, &shape)?;
        let numerator = self.binary(cx, add_op_symbol(), &two_a, &b_rho, &shape)?;
        let sqrt_argument = self.binary(cx, add_op_symbol(), &scalar(1.0)?, &z, &shape)?;
        let root = self.unary(cx, sqrt_op_symbol(), &sqrt_argument, &shape)?;
        let denominator = self.binary(cx, add_op_symbol(), &root, &scalar(1.0)?, &shape)?;
        self.binary(cx, div_op_symbol(), &numerator, &denominator, &shape)
    }

    fn plane_delta(
        &self,
        cx: &mut Cx,
        tile: &PreparedTile,
        constants: PlaneTileConstants,
    ) -> Result<Tensor, LoweringError> {
        self.dot_local(cx, tile, constants.direction)
    }

    fn dot_local(
        &self,
        cx: &mut Cx,
        tile: &PreparedTile,
        vector: [f32; 3],
    ) -> Result<Tensor, LoweringError> {
        let shape = tile.tile.shape().to_vec();
        let x = self.binary(
            cx,
            mul_op_symbol(),
            &tile.local[0],
            &scalar(vector[0])?,
            &shape,
        )?;
        let y = self.binary(
            cx,
            mul_op_symbol(),
            &tile.local[1],
            &scalar(vector[1])?,
            &shape,
        )?;
        let z = self.binary(
            cx,
            mul_op_symbol(),
            &tile.local[2],
            &scalar(vector[2])?,
            &shape,
        )?;
        let xy = self.binary(cx, add_op_symbol(), &x, &y, &shape)?;
        self.binary(cx, add_op_symbol(), &xy, &z, &shape)
    }

    fn dot_local_tensors(&self, cx: &mut Cx, tile: &PreparedTile) -> Result<Tensor, LoweringError> {
        let shape = tile.tile.shape().to_vec();
        let x = self.binary(cx, mul_op_symbol(), &tile.local[0], &tile.local[0], &shape)?;
        let y = self.binary(cx, mul_op_symbol(), &tile.local[1], &tile.local[1], &shape)?;
        let z = self.binary(cx, mul_op_symbol(), &tile.local[2], &tile.local[2], &shape)?;
        let xy = self.binary(cx, add_op_symbol(), &x, &y, &shape)?;
        self.binary(cx, add_op_symbol(), &xy, &z, &shape)
    }

    fn pairwise_accumulate(
        &self,
        cx: &mut Cx,
        tile: PlaneTile,
        mut fields: Vec<(Tensor, Tensor)>,
        submissions: &mut Vec<SubmissionEvidence>,
    ) -> Result<(Tensor, Tensor), LoweringError> {
        let shape = tile.shape().to_vec();
        while fields.len() > 1 {
            let mut next = Vec::with_capacity(fields.len().div_ceil(2));
            let mut pairs = fields.into_iter();
            while let Some(left) = pairs.next() {
                if let Some(right) = pairs.next() {
                    next.push((
                        self.binary(cx, add_op_symbol(), &left.0, &right.0, &shape)?,
                        self.binary(cx, add_op_symbol(), &left.1, &right.1, &shape)?,
                    ));
                } else {
                    next.push(left);
                }
            }
            submissions.push(self.flush()?);
            fields = next;
        }
        fields.into_iter().next().ok_or_else(|| {
            LoweringError::new(
                PreflightCheck::Execution,
                "source accumulation received no source fields",
            )
        })
    }

    fn binary(
        &self,
        cx: &mut Cx,
        symbol: Symbol,
        left: &Tensor,
        right: &Tensor,
        shape: &[usize],
    ) -> Result<Tensor, LoweringError> {
        self.submit(cx, symbol, vec![left.clone(), right.clone()], shape)
    }

    fn unary(
        &self,
        cx: &mut Cx,
        symbol: Symbol,
        input: &Tensor,
        shape: &[usize],
    ) -> Result<Tensor, LoweringError> {
        self.submit(cx, symbol, vec![input.clone()], shape)
    }

    fn submit(
        &self,
        cx: &mut Cx,
        symbol: Symbol,
        inputs: Vec<Tensor>,
        shape: &[usize],
    ) -> Result<Tensor, LoweringError> {
        let request = TensorRequest::new(
            TensorOp::new(symbol.clone(), self.nil_attributes.clone()),
            inputs,
            TensorMeta::new(shape.to_vec(), domains::f32()),
        );
        let tensor = match self.executor.execute(cx, request).map_err(|error| {
            LoweringError::new(
                PreflightCheck::Execution,
                format!(
                    "executor {} failed {symbol}: {error}",
                    self.executor_card.symbol
                ),
            )
        })? {
            TensorExecution::Complete(tensor) => tensor,
            TensorExecution::Unsupported { reason } => {
                return Err(LoweringError::new(
                    PreflightCheck::Execution,
                    format!(
                        "executor {} advertised then declined {symbol}: {reason}",
                        self.executor_card.symbol
                    ),
                ));
            }
        };
        if tensor.shape() != shape || tensor.dtype() != &domains::f32() {
            return Err(LoweringError::new(
                PreflightCheck::Execution,
                format!(
                    "executor {} returned shape {:?} dtype {} for {symbol}, expected {shape:?} numbers/f32",
                    self.executor_card.symbol,
                    tensor.shape(),
                    tensor.dtype()
                ),
            ));
        }
        Ok(tensor)
    }

    fn flush(&self) -> Result<SubmissionEvidence, LoweringError> {
        self.executor.flush().map_err(|error| {
            LoweringError::new(
                PreflightCheck::Execution,
                format!(
                    "executor {} flush failed: {error}",
                    self.executor_card.symbol
                ),
            )
        })
    }
}

fn merge_estimate(total: &mut SourcePhaseEstimate, next: SourcePhaseEstimate) {
    total.max_abs_residual_phase_rad = total
        .max_abs_residual_phase_rad
        .max(next.max_abs_residual_phase_rad);
    total.max_predicted_geometry_error_rad = total
        .max_predicted_geometry_error_rad
        .max(next.max_predicted_geometry_error_rad);
    total.max_predicted_roundoff_error_rad = total
        .max_predicted_roundoff_error_rad
        .max(next.max_predicted_roundoff_error_rad);
}
