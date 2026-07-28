use std::{
    collections::BTreeSet,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use sim_kernel::{Cx, Symbol};
use sim_lib_interference_core::{
    Emitter, FieldAmplitude, Hertz, InterferenceProblem, MetresPerSecond, NepersPerMetre, Point3M,
    PositiveMetres, Radians, SamplingPlane, ScalarMedium, SourceSet, UnitVector3,
};
use sim_lib_numbers_tensor::{
    SubmissionEvidence, Tensor, TensorExecError, TensorExecution, TensorExecutor,
    TensorExecutorCard, TensorRequest, TypedTensorStorage, add_op_symbol, cos_op_symbol,
    div_op_symbol, domains, exp_op_symbol, mul_op_symbol, parse_f32_literal_cell, sin_op_symbol,
    sqrt_op_symbol, sub_op_symbol,
};

use crate::{
    LoweringPlan, PhaseBudget, PreflightCheck, SourceTileConstants, TilePlan, TileProfile,
    preflight::required_operation_symbols,
};

// conformance: normalized tile-local f32 tensor lowering

fn test_context() -> Cx {
    sim_kernel::testing::eager_cx()
}

fn plane_at_x(x: f64, rows: usize, columns: usize) -> SamplingPlane {
    SamplingPlane::new(
        Point3M::from_metres(x, -0.5, -0.5).unwrap(),
        UnitVector3::new(0.0, 1.0, 0.0).unwrap(),
        UnitVector3::new(0.0, 0.0, 1.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        PositiveMetres::new(1.0).unwrap(),
        rows,
        columns,
    )
    .unwrap()
}

fn point(id: &str, position: [f64; 3], phase: f64) -> Emitter {
    Emitter::Point {
        id: id.to_owned(),
        position: Point3M::from_metres(position[0], position[1], position[2]).unwrap(),
        amplitude_at_reference: FieldAmplitude::new(1.0).unwrap(),
        phase: Radians::new(phase).unwrap(),
    }
}

fn forward_plane(id: &str, through_x: f64) -> Emitter {
    Emitter::ForwardPlane {
        id: id.to_owned(),
        through: Point3M::from_metres(through_x, 0.0, 0.0).unwrap(),
        direction: UnitVector3::new(1.0, 0.0, 0.0).unwrap(),
        amplitude: FieldAmplitude::new(1.0).unwrap(),
        phase: Radians::new(0.25).unwrap(),
    }
}

fn problem(sources: Vec<Emitter>, singularity_radius: f64) -> InterferenceProblem {
    InterferenceProblem::new(
        Hertz::new(343.0).unwrap(),
        ScalarMedium::new(
            MetresPerSecond::new(343.0).unwrap(),
            NepersPerMetre::new(0.01).unwrap(),
        ),
        SourceSet::new(sources).unwrap(),
        PositiveMetres::new(singularity_radius).unwrap(),
    )
}

#[test]
fn phase_budget_never_admits_arguments_above_pi() {
    assert!(PhaseBudget::default().max_abs_residual_phase_rad <= std::f64::consts::PI);
    let error = PhaseBudget::new(std::f64::consts::PI.next_up(), 1.0e-4, 1.0e-4).unwrap_err();
    assert_eq!(error.check(), PreflightCheck::Configuration);
}

#[test]
fn tiles_obey_phase_element_segment_and_profile_limits() {
    let plane = plane_at_x(1_000.0, 4, 4);
    let profile = TileProfile {
        max_elements_per_tile: 4,
        max_segment_bytes: 8,
        max_segments_per_tensor: 2,
        max_tensor_bytes: 16,
        max_result_bytes: 128,
        ..TileProfile::default()
    };
    let plan = TilePlan::new(
        plane,
        2.0 * std::f64::consts::PI,
        PhaseBudget::default(),
        profile,
    )
    .unwrap();
    assert!(plan.tiles().len() > 1);
    assert_eq!(
        plan.tiles()
            .iter()
            .map(|tile| tile.rows() * tile.columns())
            .sum::<usize>(),
        16
    );
    for tile in plan.tiles() {
        assert!(tile.rows() * tile.columns() <= 4);
        assert!(tile.segments_per_tensor() <= 2);
        assert!(tile.tensor_bytes() <= 16);
        assert!(2.0 * std::f64::consts::PI * tile.max_offset_metres() <= std::f64::consts::PI);
    }
}

#[test]
fn point_and_plane_constants_are_host_anchored_and_local() {
    let problem = problem(
        vec![
            point("point", [0.0, 0.0, 0.0], 0.0),
            forward_plane("plane", 0.0),
        ],
        0.1,
    );
    let mut cx = test_context();
    let plan = LoweringPlan::preflight(
        &mut cx,
        &problem,
        plane_at_x(1_000.0, 2, 2),
        PhaseBudget::default(),
        TileProfile::default(),
    )
    .unwrap();
    let constants = plan.source_constants(0).unwrap();
    let point = constants
        .iter()
        .find_map(|constants| match constants {
            SourceTileConstants::Point { constants, .. } => Some(constants),
            SourceTileConstants::ForwardPlane { .. } => None,
        })
        .unwrap();
    assert_eq!(point.n0, [1.0, 0.0, 0.0]);
    assert!((point.rho - 0.001).abs() < 1.0e-9);
    assert!(point.phase_cos.is_finite() && point.phase_sin.is_finite());
    assert!(point.gain0 > 0.0);
    let plane = constants
        .iter()
        .find_map(|constants| match constants {
            SourceTileConstants::Point { .. } => None,
            SourceTileConstants::ForwardPlane { constants, .. } => Some(constants),
        })
        .unwrap();
    assert_eq!(plane.direction, [1.0, 0.0, 0.0]);
    assert_eq!(plane.signed_distance0_m, 1_000.0);
    assert!(plane.phase_cos.is_finite() && plane.phase_sin.is_finite());
}

#[test]
fn every_preflight_refusal_happens_before_the_first_execute() {
    let problem = problem(vec![point("point", [0.0, 0.0, 0.0], 0.0)], 0.1);
    let plane = plane_at_x(1_000.0, 2, 2);
    for missing in required_operation_symbols() {
        let executor = Arc::new(RecordingExecutor::missing(missing));
        let mut cx = test_context();
        let error = LoweringPlan::preflight_with_executor(
            &mut cx,
            &problem,
            plane,
            PhaseBudget::default(),
            TileProfile::default(),
            executor.clone(),
        )
        .err()
        .expect("missing operation must fail");
        assert_eq!(error.check(), PreflightCheck::Operator);
        assert_eq!(executor.executes.load(Ordering::SeqCst), 0);
    }

    let mut cx = test_context();
    let error = LoweringPlan::preflight(
        &mut cx,
        &problem,
        plane,
        PhaseBudget::default(),
        TileProfile {
            supports_f32: false,
            ..TileProfile::default()
        },
    )
    .err()
    .expect("f32 refusal");
    assert_eq!(error.check(), PreflightCheck::DType);

    let error = LoweringPlan::preflight(
        &mut cx,
        &problem,
        plane,
        PhaseBudget::default(),
        TileProfile {
            max_result_bytes: 31,
            ..TileProfile::default()
        },
    )
    .err()
    .expect("result allocation refusal");
    assert_eq!(error.check(), PreflightCheck::ResultAllocation);

    let error = LoweringPlan::preflight(
        &mut cx,
        &problem,
        plane,
        PhaseBudget::new(std::f64::consts::PI, 1.0e-12, 1.0e-12).unwrap(),
        TileProfile::default(),
    )
    .err()
    .expect("roundoff budget refusal");
    assert!(matches!(
        error.check(),
        PreflightCheck::GeometryBudget | PreflightCheck::RoundoffBudget
    ));
}

#[test]
fn singular_and_backward_samples_fail_before_execution() {
    let mut cx = test_context();
    let singular = problem(vec![point("singular", [0.0, 0.0, 0.0], 0.0)], 0.1);
    let error = LoweringPlan::preflight(
        &mut cx,
        &singular,
        plane_at_x(0.0, 1, 1),
        PhaseBudget::default(),
        TileProfile::default(),
    )
    .err()
    .expect("singular sample refusal");
    assert_eq!(error.check(), PreflightCheck::Singularity);

    let backward = problem(vec![forward_plane("plane", 1.0)], 0.1);
    let error = LoweringPlan::preflight(
        &mut cx,
        &backward,
        plane_at_x(0.0, 2, 2),
        PhaseBudget::default(),
        TileProfile::default(),
    )
    .err()
    .expect("backward sample refusal");
    assert_eq!(error.check(), PreflightCheck::ForwardPlane);
}

#[test]
fn lowering_uses_only_open_operations_and_pairwise_levels() {
    let problem = problem(
        vec![
            point("a", [0.0, -0.25, 0.0], 0.0),
            point("b", [0.0, 0.0, 0.0], 0.5),
            point("c", [0.0, 0.25, 0.0], -0.5),
        ],
        0.1,
    );
    let executor = Arc::new(RecordingExecutor::complete());
    let mut cx = test_context();
    let plan = LoweringPlan::preflight_with_executor(
        &mut cx,
        &problem,
        plane_at_x(1_000.0, 2, 2),
        PhaseBudget::default(),
        TileProfile::default(),
        executor.clone(),
    )
    .unwrap();
    assert!(plan.max_phase_estimate().max_abs_residual_phase_rad <= std::f64::consts::PI);
    let lowered = plan.execute(&mut cx).unwrap();
    assert_eq!(lowered.len(), 1);
    assert_eq!(lowered[0].real().shape(), &[2, 2]);
    assert_eq!(lowered[0].imaginary().shape(), &[2, 2]);
    assert_eq!(lowered[0].submissions().len(), 5);
    for tensor in [lowered[0].real(), lowered[0].imaginary()] {
        assert!(
            tensor
                .cells()
                .unwrap()
                .iter()
                .all(|cell| parse_f32_literal_cell(cell).unwrap().is_finite())
        );
    }
    let observed: BTreeSet<_> = executor
        .operations
        .lock()
        .unwrap()
        .iter()
        .map(ToString::to_string)
        .collect();
    let expected: BTreeSet<_> = required_operation_symbols()
        .into_iter()
        .map(|symbol| symbol.to_string())
        .collect();
    assert_eq!(observed, expected);
}

struct RecordingExecutor {
    missing: Option<Symbol>,
    executes: AtomicUsize,
    flushes: AtomicUsize,
    operations: Mutex<Vec<Symbol>>,
}

impl RecordingExecutor {
    fn complete() -> Self {
        Self {
            missing: None,
            executes: AtomicUsize::new(0),
            flushes: AtomicUsize::new(0),
            operations: Mutex::new(Vec::new()),
        }
    }

    fn missing(symbol: Symbol) -> Self {
        Self {
            missing: Some(symbol),
            ..Self::complete()
        }
    }
}

impl TensorExecutor for RecordingExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            Symbol::qualified("test", "recording-executor"),
            "recording",
            Symbol::qualified("test", "local"),
            required_operation_symbols()
                .into_iter()
                .filter(|symbol| self.missing.as_ref() != Some(symbol))
                .collect(),
            None,
        )
    }

    fn execute(
        &self,
        _cx: &mut Cx,
        request: TensorRequest,
    ) -> Result<TensorExecution, TensorExecError> {
        self.executes.fetch_add(1, Ordering::SeqCst);
        self.operations
            .lock()
            .unwrap()
            .push(request.operation.symbol.clone());
        let cells = if request.inputs.len() == 1 {
            unary_cells(&request)?
        } else {
            binary_cells(&request)?
        };
        let tensor = Tensor::from_storage(
            request.output.shape().to_vec(),
            domains::f32(),
            Arc::new(TypedTensorStorage::<f32>::new(cells)),
        )
        .map_err(exec_error)?;
        Ok(TensorExecution::Complete(tensor))
    }

    fn flush(&self) -> Result<SubmissionEvidence, TensorExecError> {
        let accepted = self.flushes.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(SubmissionEvidence::new(
            Symbol::qualified("test", "recording-executor"),
            accepted,
        ))
    }
}

fn unary_cells(request: &TensorRequest) -> Result<Vec<f32>, TensorExecError> {
    let [input] = request.inputs.as_ref() else {
        return Err(exec_error("unary test operation expects one input"));
    };
    tensor_cells(input)?
        .into_iter()
        .map(|value| {
            if request.operation.symbol == sqrt_op_symbol() {
                Ok(value.sqrt())
            } else if request.operation.symbol == exp_op_symbol() {
                Ok(value.exp())
            } else if request.operation.symbol == sin_op_symbol() {
                Ok(value.sin())
            } else if request.operation.symbol == cos_op_symbol() {
                Ok(value.cos())
            } else {
                Err(exec_error("unexpected unary test operation"))
            }
        })
        .collect()
}

fn binary_cells(request: &TensorRequest) -> Result<Vec<f32>, TensorExecError> {
    let [left, right] = request.inputs.as_ref() else {
        return Err(exec_error("binary test operation expects two inputs"));
    };
    let output_len = request.output.shape().iter().copied().product::<usize>();
    let left = broadcast_cells(left, output_len)?;
    let right = broadcast_cells(right, output_len)?;
    left.into_iter()
        .zip(right)
        .map(|(left, right)| {
            if request.operation.symbol == add_op_symbol() {
                Ok(left + right)
            } else if request.operation.symbol == sub_op_symbol() {
                Ok(left - right)
            } else if request.operation.symbol == mul_op_symbol() {
                Ok(left * right)
            } else if request.operation.symbol == div_op_symbol() {
                Ok(left / right)
            } else {
                Err(exec_error("unexpected binary test operation"))
            }
        })
        .collect()
}

fn broadcast_cells(tensor: &Tensor, output_len: usize) -> Result<Vec<f32>, TensorExecError> {
    let cells = tensor_cells(tensor)?;
    match cells.as_slice() {
        [scalar] if tensor.shape().is_empty() => Ok(vec![*scalar; output_len]),
        _ if cells.len() == output_len => Ok(cells),
        _ => Err(exec_error("unexpected test broadcast shape")),
    }
}

fn tensor_cells(tensor: &Tensor) -> Result<Vec<f32>, TensorExecError> {
    tensor
        .cells()
        .map_err(exec_error)?
        .iter()
        .map(|cell| {
            parse_f32_literal_cell(cell)
                .ok_or_else(|| exec_error("test Tensor contains a non-f32 cell"))
        })
        .collect()
}

fn exec_error(error: impl std::fmt::Display) -> TensorExecError {
    TensorExecError::Eval {
        message: Arc::from(error.to_string()),
    }
}
