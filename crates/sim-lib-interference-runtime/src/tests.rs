//! Focused record, Tensor, Shape, and codec conformance.

use std::{
    any::Any,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use sim_citizen::{CitizenRuntime, values_citizen_eq};
use sim_codec::{Input, Output, decode_with_codec, encode_with_codec};
use sim_kernel::{
    CapabilitySet, Cx, DefaultFactory, EagerPolicy, EncodeOptions, Env, Error, Expr, NumberLiteral,
    ObjectEncode, ObjectEncoding, ReadPolicy, Result, Symbol, TrustLevel, Value,
    read_construct_capability,
};
use sim_lib_interference_core::{SamplingPolicy, SamplingThresholds, WorkBudget};
use sim_lib_interference_solve::{Observable, ReductionRule, ReferencePhasorSolver, project};
use sim_lib_numbers_tensor::{Tensor, TensorLocation, TensorStorage, TypedTensorStorage, domains};

use crate::{
    EmitterDescriptor, InterferenceLib, InterferenceRecordsLib, MediumDescriptor,
    PhasorFieldDescriptor, PlaneDescriptor, ProblemDescriptor, ProjectionCertificateDescriptor,
    ProjectionRequestDescriptor, SamplingCertificateDescriptor, ScalarProjectionDescriptor,
    SolveRequest, SolverProvider, StudyDescriptor, StudyEvidenceDescriptor, StudySolver,
    WorkEstimateDescriptor,
    citizen::{RecordCitizenSpec, interference_citizen_registry, read_construct_parts},
    projection_records::sample_projection,
    records::{sample_plane, sample_problem},
    resolve_study_solver,
    shapes::{
        plane_shape_symbol, problem_shape_symbol, projection_request_shape_symbol,
        projection_shape_symbol, study_shape_symbol,
    },
    study_solver_symbol,
};

const CITIZEN_SYMBOLS: [&str; 12] = [
    "interference/Medium",
    "interference/Emitter",
    "interference/Problem",
    "interference/Plane",
    "interference/SamplingCertificate",
    "interference/WorkEstimate",
    "interference/PhasorField",
    "interference/StudyEvidence",
    "interference/Study",
    "interference/ProjectionCertificate",
    "interference/ProjectionRequest",
    "interference/Projection",
];

#[test]
fn interference_lib_installs_the_reference_solver_as_registry_default() {
    let mut cx = bare_cx();
    cx.load_lib(&InterferenceRecordsLib).unwrap();
    cx.load_lib(&InterferenceLib).unwrap();

    let solver = resolve_study_solver(&cx).unwrap();
    let problem = sample_problem().to_problem().unwrap();
    let plane = sample_plane().to_plane().unwrap();
    let request = SolveRequest::new(
        &problem,
        &plane,
        SamplingPolicy::Annotate,
        SamplingThresholds::default(),
        WorkBudget::default(),
    );
    let study = solver.solve(&mut cx, &request).unwrap();

    assert_eq!(study.problem, sample_problem());
    assert_eq!(study.plane, sample_plane());
    assert!(
        cx.registry()
            .value_by_symbol(&study_solver_symbol())
            .unwrap()
            .object()
            .downcast_ref::<SolverProvider>()
            .is_some()
    );
}

#[test]
fn child_environment_solver_precedes_the_registry_default() {
    struct UnusableSolver;

    impl StudySolver for UnusableSolver {
        fn solve(&self, _cx: &mut Cx, _request: &SolveRequest<'_>) -> Result<StudyDescriptor> {
            Err(Error::Eval("child solver selected".to_owned()))
        }
    }

    let mut cx = bare_cx();
    cx.load_lib(&InterferenceRecordsLib).unwrap();
    cx.load_lib(&InterferenceLib).unwrap();
    let mut child = Env::child(Arc::new(cx.env().clone()));
    child.define(
        study_solver_symbol(),
        SolverProvider::new(Arc::new(UnusableSolver))
            .into_value()
            .unwrap(),
    );

    cx.with_env(child, |cx| {
        let problem = sample_problem().to_problem().unwrap();
        let plane = sample_plane().to_plane().unwrap();
        let request = SolveRequest::new(
            &problem,
            &plane,
            SamplingPolicy::Annotate,
            SamplingThresholds::default(),
            WorkBudget::default(),
        );
        let error = resolve_study_solver(cx)
            .unwrap()
            .solve(cx, &request)
            .unwrap_err();
        assert!(error.to_string().contains("child solver selected"));
        Ok(())
    })
    .unwrap();
}

#[test]
fn explicit_citizen_registry_is_complete_and_conformant() {
    let registry = interference_citizen_registry().unwrap();
    registry.ensure_contains_symbols(&CITIZEN_SYMBOLS).unwrap();
    let mut cx = bare_cx();
    sim_citizen::run_registry_conformance_expecting(&mut cx, &registry, &CITIZEN_SYMBOLS).unwrap();
}

#[test]
fn installed_lisp_json_and_binary_round_trip_every_record() {
    let mut cx = codec_cx();
    let values = example_record_values(&mut cx);
    for value in values {
        let expr = value.object().as_expr(&mut cx).unwrap();
        for codec in [
            Symbol::qualified("codec", "lisp"),
            Symbol::qualified("codec", "json"),
            Symbol::qualified("codec", "binary"),
        ] {
            let encoded =
                encode_with_codec(&mut cx, &codec, &expr, EncodeOptions::default()).unwrap();
            let input = match encoded {
                Output::Text(text) => Input::Text(text),
                Output::Bytes(bytes) => Input::Bytes(bytes),
            };
            let decoded =
                decode_with_codec(&mut cx, &codec, input, read_policy_with_construct()).unwrap();
            assert!(
                decoded.canonical_eq(&expr),
                "codec {codec} changed {expr:?} into {decoded:?}"
            );
            let reconstructed = reconstruct_record_expr(&mut cx, &decoded).unwrap();
            assert!(
                values_citizen_eq(&mut cx, &value, &reconstructed).unwrap(),
                "codec {codec} failed semantic reconstruction"
            );
        }
    }
}

#[test]
fn host_field_moves_into_typed_f64_tensors_and_materializes_back() {
    let field = sim_lib_interference_solve::HostPhasorField::from_component_planes(
        2,
        3,
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    )
    .unwrap();
    let descriptor = PhasorFieldDescriptor::from_host(field).unwrap();

    assert_eq!(descriptor.real.shape(), &[2, 3]);
    assert_eq!(descriptor.real.dtype(), &domains::f64());
    assert_eq!(descriptor.real.location(), TensorLocation::Host);
    assert!(
        descriptor
            .real
            .storage()
            .as_any()
            .downcast_ref::<TypedTensorStorage<f64>>()
            .is_some()
    );

    let materialized = descriptor.materialize_host(&mut bare_cx()).unwrap();
    assert_eq!(materialized.real(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(
        materialized.imaginary(),
        &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    );
}

#[test]
fn resident_field_materializes_each_component_exactly_once() {
    let real_count = Arc::new(AtomicUsize::new(0));
    let imag_count = Arc::new(AtomicUsize::new(0));
    let real = resident_tensor(
        "real-buffer",
        vec![1.0, 2.0, 3.0, 4.0],
        real_count.clone(),
        "compute/site/model",
    );
    let imag = resident_tensor(
        "imag-buffer",
        vec![5.0, 6.0, 7.0, 8.0],
        imag_count.clone(),
        "compute/site/model",
    );
    let descriptor = PhasorFieldDescriptor::new(2, 2, real, imag).unwrap();

    assert_eq!(real_count.load(Ordering::SeqCst), 0);
    assert_eq!(imag_count.load(Ordering::SeqCst), 0);
    let host = descriptor.materialize_host(&mut bare_cx()).unwrap();
    assert_eq!(real_count.load(Ordering::SeqCst), 1);
    assert_eq!(imag_count.load(Ordering::SeqCst), 1);
    assert_eq!(host.real(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(host.imaginary(), &[5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn field_refuses_shape_dtype_and_site_mismatches() {
    let host_f64 = typed_tensor::<f64>(vec![2, 2], vec![0.0; 4]);
    let host_f32 = typed_tensor::<f32>(vec![2, 2], vec![0.0; 4]);
    assert!(
        PhasorFieldDescriptor::new(2, 2, host_f64.clone(), host_f32)
            .unwrap_err()
            .to_string()
            .contains("dtypes")
    );

    let wrong_shape = typed_tensor::<f64>(vec![1, 4], vec![0.0; 4]);
    assert!(
        PhasorFieldDescriptor::new(2, 2, host_f64.clone(), wrong_shape)
            .unwrap_err()
            .to_string()
            .contains("shapes")
    );

    let left = resident_tensor(
        "left",
        vec![0.0; 4],
        Arc::new(AtomicUsize::new(0)),
        "compute/site/model",
    );
    let right = resident_tensor(
        "right",
        vec![0.0; 4],
        Arc::new(AtomicUsize::new(0)),
        "compute/site/other",
    );
    assert!(
        PhasorFieldDescriptor::new(2, 2, left, right)
            .unwrap_err()
            .to_string()
            .contains("locations")
    );
}

#[test]
fn actual_reference_study_preserves_problem_plane_field_and_evidence() {
    let problem_descriptor = sample_problem();
    let plane_descriptor = sample_plane();
    let problem = problem_descriptor.to_problem().unwrap();
    let plane = plane_descriptor.to_plane().unwrap();
    let solver = ReferencePhasorSolver::new(
        SamplingPolicy::Annotate,
        SamplingThresholds::default(),
        WorkBudget::default(),
    );
    let (host, evidence) = solver.solve(&problem, &plane).unwrap();
    let expected_real = host.real().to_vec();
    let expected_imag = host.imaginary().to_vec();
    let study = StudyDescriptor::from_reference(&problem, plane, host, &evidence).unwrap();

    let rematerialized = study.field.materialize_host(&mut bare_cx()).unwrap();
    assert_eq!(rematerialized.real(), expected_real);
    assert_eq!(rematerialized.imaginary(), expected_imag);
    assert_eq!(study.problem, problem_descriptor);
    assert_eq!(study.plane, plane_descriptor);
    assert_eq!(study.evidence.work.cells, 4);
    assert_eq!(study.evidence.completed_cells, 4);
    assert_eq!(study.evidence.intermediate_materializations, 0);
}

#[test]
fn registered_shapes_accept_exact_records_and_reject_malformed_values() {
    let mut cx = bare_cx();
    cx.load_lib(&InterferenceRecordsLib).unwrap();

    for (symbol, value) in [
        (
            problem_shape_symbol(),
            opaque(&cx, <ProblemDescriptor as RecordCitizenSpec>::example()),
        ),
        (
            plane_shape_symbol(),
            opaque(&cx, <PlaneDescriptor as RecordCitizenSpec>::example()),
        ),
        (
            study_shape_symbol(),
            opaque(&cx, <StudyDescriptor as RecordCitizenSpec>::example()),
        ),
        (
            projection_request_shape_symbol(),
            opaque(
                &cx,
                <ProjectionRequestDescriptor as RecordCitizenSpec>::example(),
            ),
        ),
        (
            projection_shape_symbol(),
            opaque(
                &cx,
                <ScalarProjectionDescriptor as RecordCitizenSpec>::example(),
            ),
        ),
    ] {
        let shape = cx.registry().shape_by_symbol(&symbol).unwrap().clone();
        assert!(
            shape
                .object()
                .as_shape()
                .unwrap()
                .check_value(&mut cx, value)
                .unwrap()
                .accepted,
            "Shape {symbol} rejected its exact record"
        );
    }

    let mut malformed = sample_problem();
    malformed.frequency_hz = -1.0;
    let shape = cx
        .registry()
        .shape_by_symbol(&problem_shape_symbol())
        .unwrap()
        .clone();
    let malformed = opaque(&cx, malformed);
    assert!(
        !shape
            .object()
            .as_shape()
            .unwrap()
            .check_value(&mut cx, malformed)
            .unwrap()
            .accepted
    );
}

#[test]
fn malformed_tensor_dimensions_units_masks_and_evidence_fail_closed() {
    let mut cx = codec_cx();

    let field = <PhasorFieldDescriptor as RecordCitizenSpec>::example();
    let mut field_args = constructor_args(&mut cx, &field);
    let Expr::Extension { payload, .. } = &mut field_args[3] else {
        panic!("real component must be a read construct");
    };
    let Expr::Vector(tensor_args) = payload.as_mut() else {
        panic!("Tensor payload must be a vector");
    };
    tensor_args[2] = Expr::List(vec![int_expr(3), int_expr(2)]);
    assert_construct_rejected::<PhasorFieldDescriptor>(&mut cx, field_args, "tensor shape");

    let mut medium_args =
        constructor_args(&mut cx, &<MediumDescriptor as RecordCitizenSpec>::example());
    medium_args[1] = f64_expr(-343.0);
    assert_construct_rejected::<MediumDescriptor>(&mut cx, medium_args, "Medium");

    let projection = sample_projection();
    let mut projection_args = constructor_args(&mut cx, &projection);
    projection_args[4] = Expr::List(vec![Expr::Bool(false)]);
    assert_construct_rejected::<ScalarProjectionDescriptor>(&mut cx, projection_args, "mask");

    let evidence = <StudyEvidenceDescriptor as RecordCitizenSpec>::example();
    let mut evidence_args = constructor_args(&mut cx, &evidence);
    evidence_args[8] = int_expr(evidence.completed_cells + 1);
    assert_construct_rejected::<StudyEvidenceDescriptor>(&mut cx, evidence_args, "completion");
}

#[test]
fn scalar_projection_has_one_tensor_and_exact_mask_provenance() {
    let problem = sample_problem().to_problem().unwrap();
    let plane = sample_plane().to_plane().unwrap();
    let sampling =
        sim_lib_interference_core::SamplingCertificate::measure(&problem, &plane).unwrap();
    let field = sim_lib_interference_solve::HostPhasorField::from_component_planes(
        2,
        2,
        vec![0.0, 1.0, 0.0, -1.0],
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let projected = project(&field, sampling, Observable::Phase, 0.0).unwrap();
    let record = ScalarProjectionDescriptor::from_projection(&projected).unwrap();

    assert_eq!(record.values.shape(), &[2, 2]);
    assert_eq!(record.values.dtype(), &domains::f64());
    assert_eq!(record.mask, vec![true, false, true, false]);
    assert_eq!(record.certificate.mask_count, 2);
    record.validate().unwrap();

    ProjectionRequestDescriptor::new(
        Observable::Phase,
        0.0,
        2,
        2,
        ReductionRule::DetectorComplexMean,
    )
    .unwrap();
}

fn example_record_values(cx: &mut Cx) -> Vec<Value> {
    vec![
        opaque(cx, <MediumDescriptor as RecordCitizenSpec>::example()),
        opaque(cx, <EmitterDescriptor as RecordCitizenSpec>::example()),
        opaque(cx, <ProblemDescriptor as RecordCitizenSpec>::example()),
        opaque(cx, <PlaneDescriptor as RecordCitizenSpec>::example()),
        opaque(
            cx,
            <SamplingCertificateDescriptor as RecordCitizenSpec>::example(),
        ),
        opaque(cx, <WorkEstimateDescriptor as RecordCitizenSpec>::example()),
        opaque(cx, <PhasorFieldDescriptor as RecordCitizenSpec>::example()),
        opaque(
            cx,
            <StudyEvidenceDescriptor as RecordCitizenSpec>::example(),
        ),
        opaque(cx, <StudyDescriptor as RecordCitizenSpec>::example()),
        opaque(
            cx,
            <ProjectionCertificateDescriptor as RecordCitizenSpec>::example(),
        ),
        opaque(
            cx,
            <ProjectionRequestDescriptor as RecordCitizenSpec>::example(),
        ),
        opaque(
            cx,
            <ScalarProjectionDescriptor as RecordCitizenSpec>::example(),
        ),
    ]
}

fn codec_cx() -> Cx {
    let mut cx = bare_cx();
    cx.grant(read_construct_capability());
    cx.load_lib(&InterferenceRecordsLib).unwrap();
    let lisp = sim_codec_lisp::LispCodecLib::new(cx.registry_mut().fresh_codec_id()).unwrap();
    cx.load_lib(&lisp).unwrap();
    let json = sim_codec_json::JsonCodecLib::new(cx.registry_mut().fresh_codec_id());
    cx.load_lib(&json).unwrap();
    let binary = sim_codec_binary::BinaryCodecLib::new(cx.registry_mut().fresh_codec_id());
    cx.load_lib(&binary).unwrap();
    cx
}

fn reconstruct_record_expr(cx: &mut Cx, expr: &Expr) -> Result<Value> {
    let (class, args) = read_construct_parts(expr, "test record")?;
    let values = args
        .iter()
        .map(|arg| sim_citizen::value_from_expr(cx, arg))
        .collect::<Result<Vec<_>>>()?;
    cx.read_construct(&class, values)
}

fn constructor_args<T>(cx: &mut Cx, value: &T) -> Vec<Expr>
where
    T: ObjectEncode,
{
    let ObjectEncoding::Constructor { args, .. } = value.object_encoding(cx).unwrap() else {
        panic!("record must encode as a constructor");
    };
    args
}

fn assert_construct_rejected<T>(cx: &mut Cx, args: Vec<Expr>, expected: &str)
where
    T: CitizenRuntime,
{
    let values = args
        .iter()
        .map(|arg| sim_citizen::value_from_expr(cx, arg))
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let error = T::construct_from_values(cx, values).unwrap_err();
    assert!(
        error.to_string().contains(expected),
        "expected {expected:?} in {error}"
    );
}

fn opaque<T>(cx: &Cx, value: T) -> Value
where
    T: sim_kernel::Object + sim_kernel::ObjectCompat + Send + Sync + 'static,
{
    cx.factory().opaque(Arc::new(value)).unwrap()
}

fn read_policy_with_construct() -> ReadPolicy {
    ReadPolicy {
        trust: TrustLevel::TrustedSource,
        capabilities: CapabilitySet::new().grant(read_construct_capability()),
    }
}

fn bare_cx() -> Cx {
    Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
}

fn int_expr(value: impl ToString) -> Expr {
    Expr::Number(NumberLiteral {
        domain: Symbol::qualified("citizen", "int"),
        canonical: value.to_string(),
    })
}

fn f64_expr(value: f64) -> Expr {
    Expr::Number(NumberLiteral {
        domain: domains::f64(),
        canonical: value.to_string(),
    })
}

fn typed_tensor<T>(shape: Vec<usize>, cells: Vec<T>) -> Tensor
where
    T: sim_lib_numbers_tensor::TensorCell,
{
    Tensor::from_storage(
        shape,
        T::dtype(),
        Arc::new(TypedTensorStorage::<T>::new(cells)),
    )
    .unwrap()
}

fn resident_tensor(
    allocation: &str,
    cells: Vec<f64>,
    count: Arc<AtomicUsize>,
    site: &str,
) -> Tensor {
    Tensor::from_storage(
        vec![2, 2],
        domains::f64(),
        Arc::new(CountingResidentStorage {
            cells: cells.into(),
            count,
            site: Symbol::new(site),
            allocation: Symbol::new(allocation),
        }),
    )
    .unwrap()
}

struct CountingResidentStorage {
    cells: Arc<[f64]>,
    count: Arc<AtomicUsize>,
    site: Symbol,
    allocation: Symbol,
}

impl TensorStorage for CountingResidentStorage {
    fn dtype(&self) -> &Symbol {
        static DTYPE: std::sync::OnceLock<Symbol> = std::sync::OnceLock::new();
        DTYPE.get_or_init(domains::f64)
    }

    fn len(&self) -> usize {
        self.cells.len()
    }

    fn location(&self) -> TensorLocation {
        TensorLocation::Resident {
            site: self.site.clone(),
            allocation: self.allocation.clone(),
        }
    }

    fn cell(&self, _index: usize) -> Result<Value> {
        Err(Error::Eval(
            "resident cells require explicit materialization".to_owned(),
        ))
    }

    fn materialize(&self) -> Result<Arc<dyn TensorStorage>> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(TypedTensorStorage::<f64>::from_shared(
            self.cells.clone(),
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
