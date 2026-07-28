//! Citizen plumbing shared by the validated interference records.

use sim_citizen::{CitizenField, CitizenRegistry, CitizenRuntime};
use sim_kernel::{
    AbiVersion, Cx, Error, Export, Expr, Lib, LibManifest, LibTarget, Linker, Result, Symbol,
    Value, Version,
};

use crate::{
    EmitterDescriptor, MediumDescriptor, PhasorFieldDescriptor, PlaneDescriptor, ProblemDescriptor,
    ProjectionCertificateDescriptor, ProjectionRequestDescriptor, SamplingCertificateDescriptor,
    ScalarProjectionDescriptor, StudyDescriptor, StudyEvidenceDescriptor, WorkEstimateDescriptor,
};

pub(crate) trait RecordCitizenSpec:
    Clone + core::fmt::Debug + PartialEq + Send + Sync + 'static
{
    const FIELDS: &'static [&'static str];

    fn encode_fields(&self, cx: &mut Cx) -> Result<Vec<Expr>>;
    fn decode_fields(cx: &mut Cx, fields: Vec<Value>) -> Result<Self>;
    fn example() -> Self;
    fn validate(&self) -> Result<()>;
}

macro_rules! impl_record_citizen {
    ($ty:ty, $symbol:literal, $arity:literal) => {
        impl ::sim_citizen::Citizen for $ty {
            fn citizen_symbol() -> ::sim_kernel::Symbol {
                ::sim_citizen::parse_symbol($symbol)
            }

            fn citizen_version() -> u32 {
                1
            }

            fn citizen_arity() -> usize {
                <Self as $crate::citizen::RecordCitizenSpec>::FIELDS.len()
            }

            fn citizen_fields() -> &'static [&'static str] {
                <Self as $crate::citizen::RecordCitizenSpec>::FIELDS
            }
        }

        impl ::sim_kernel::Object for $ty {
            fn display(&self, cx: &mut ::sim_kernel::Cx) -> ::sim_kernel::Result<String> {
                let fields = <Self as $crate::citizen::RecordCitizenSpec>::encode_fields(self, cx)?;
                Ok(format!("#<citizen {}:{}>", $symbol, fields.len()))
            }

            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }
        }

        impl ::sim_kernel::ObjectCompat for $ty {
            fn class(
                &self,
                cx: &mut ::sim_kernel::Cx,
            ) -> ::sim_kernel::Result<::sim_kernel::ClassRef> {
                let symbol = <Self as ::sim_citizen::Citizen>::citizen_symbol();
                if let Some(value) = cx.registry().class_by_symbol(&symbol) {
                    return Ok(value.clone());
                }
                ::sim_kernel::Factory::class_stub(cx.factory(), ::sim_kernel::ClassId(0), symbol)
            }

            fn as_expr(
                &self,
                cx: &mut ::sim_kernel::Cx,
            ) -> ::sim_kernel::Result<::sim_kernel::Expr> {
                ::sim_citizen::constructor_expr(cx, self)
            }

            fn as_object_encoder(&self) -> Option<&dyn ::sim_kernel::ObjectEncode> {
                Some(self)
            }
        }

        impl ::sim_kernel::ObjectEncode for $ty {
            fn object_encoding(
                &self,
                cx: &mut ::sim_kernel::Cx,
            ) -> ::sim_kernel::Result<::sim_kernel::ObjectEncoding> {
                <Self as $crate::citizen::RecordCitizenSpec>::validate(self)?;
                let mut args = vec![::sim_kernel::Expr::Symbol(::sim_kernel::Symbol::new("v1"))];
                args.extend(<Self as $crate::citizen::RecordCitizenSpec>::encode_fields(
                    self, cx,
                )?);
                Ok(::sim_kernel::ObjectEncoding::Constructor {
                    class: <Self as ::sim_citizen::Citizen>::citizen_symbol(),
                    args,
                })
            }
        }

        impl ::sim_citizen::CitizenRuntime for $ty {
            fn citizen_info() -> ::sim_citizen::CitizenInfo {
                ::sim_citizen::CitizenInfo {
                    symbol: $symbol,
                    version: 1,
                    crate_name: env!("CARGO_PKG_NAME"),
                    arity: <Self as ::sim_citizen::Citizen>::citizen_arity(),
                    install: <Self as ::sim_citizen::CitizenRuntime>::install,
                    conformance: <Self as ::sim_citizen::CitizenRuntime>::conformance,
                }
            }

            fn conformance(cx: &mut ::sim_kernel::Cx) -> ::sim_kernel::Result<()> {
                ::sim_citizen::check_fixture(
                    cx,
                    <Self as $crate::citizen::RecordCitizenSpec>::example(),
                )
            }

            fn construct_from_values(
                cx: &mut ::sim_kernel::Cx,
                args: Vec<::sim_kernel::Value>,
            ) -> ::sim_kernel::Result<Self> {
                let expected = <Self as ::sim_citizen::Citizen>::citizen_arity() + 1;
                if args.len() != expected {
                    return Err(::sim_citizen::arity_error(
                        <Self as ::sim_citizen::Citizen>::citizen_symbol(),
                        expected,
                        args.len(),
                    ));
                }
                let mut args = args.into_iter();
                let version = args.next().expect("arity checked");
                ::sim_citizen::decode_version(
                    cx,
                    version,
                    1,
                    <Self as ::sim_citizen::Citizen>::citizen_symbol(),
                )?;
                let value = <Self as $crate::citizen::RecordCitizenSpec>::decode_fields(
                    cx,
                    args.collect(),
                )?;
                <Self as $crate::citizen::RecordCitizenSpec>::validate(&value)?;
                Ok(value)
            }

            fn example() -> Self {
                <Self as $crate::citizen::RecordCitizenSpec>::example()
            }
        }

        const _: () = {
            ::sim_citizen::inventory::submit! {
                ::sim_citizen::CitizenInfo {
                    symbol: $symbol,
                    version: 1,
                    crate_name: env!("CARGO_PKG_NAME"),
                    arity: $arity,
                    install: <$ty as ::sim_citizen::CitizenRuntime>::install,
                    conformance: <$ty as ::sim_citizen::CitizenRuntime>::conformance,
                }
            }
        };
    };
}

pub(crate) fn invalid(record: &str, message: impl Into<String>) -> Error {
    Error::Eval(format!("interference {record}: {}", message.into()))
}

pub(crate) fn next_field<T>(
    cx: &mut Cx,
    fields: &mut impl Iterator<Item = Value>,
    name: &'static str,
) -> Result<T>
where
    T: CitizenField,
{
    T::decode_field_value(
        cx,
        fields
            .next()
            .ok_or_else(|| invalid("record", format!("missing field {name}")))?,
        name,
    )
}

pub(crate) fn encode_field<T: CitizenField>(value: &T) -> Expr {
    value.encode_field()
}

pub(crate) fn encode_record<T>(cx: &mut Cx, value: &T) -> Result<Expr>
where
    T: CitizenRuntime,
{
    sim_citizen::constructor_expr(cx, value)
}

pub(crate) fn decode_record<T>(cx: &mut Cx, value: Value, field: &'static str) -> Result<T>
where
    T: CitizenRuntime,
{
    let expr = sim_citizen::value_to_expr(cx, value, field)?;
    decode_record_expr(cx, &expr, field)
}

pub(crate) fn decode_record_expr<T>(cx: &mut Cx, expr: &Expr, field: &'static str) -> Result<T>
where
    T: CitizenRuntime,
{
    let (class, args) = read_construct_parts(expr, field)?;
    if class != T::citizen_symbol() {
        return Err(invalid(
            field,
            format!("expected nested {}, found {class}", T::citizen_symbol()),
        ));
    }
    let values = args
        .iter()
        .map(|arg| sim_citizen::value_from_expr(cx, arg))
        .collect::<Result<Vec<_>>>()?;
    T::construct_from_values(cx, values)
}

pub(crate) fn encode_records<T>(cx: &mut Cx, values: &[T]) -> Result<Expr>
where
    T: CitizenRuntime,
{
    values
        .iter()
        .map(|value| encode_record(cx, value))
        .collect::<Result<Vec<_>>>()
        .map(Expr::List)
}

pub(crate) fn decode_records<T>(cx: &mut Cx, value: Value, field: &'static str) -> Result<Vec<T>>
where
    T: CitizenRuntime,
{
    let expr = sim_citizen::value_to_expr(cx, value, field)?;
    let Expr::List(items) = expr else {
        return Err(invalid(field, "expected a list of nested citizens"));
    };
    items
        .iter()
        .map(|item| decode_record_expr(cx, item, field))
        .collect()
}

pub(crate) fn read_construct_parts<'a>(
    expr: &'a Expr,
    field: &'static str,
) -> Result<(Symbol, &'a [Expr])> {
    let Expr::Extension { tag, payload } = expr else {
        return Err(invalid(field, "expected a read-construct extension"));
    };
    if *tag != Symbol::qualified("citizen", "read-construct") {
        return Err(invalid(
            field,
            format!("expected citizen/read-construct, found {tag}"),
        ));
    }
    let Expr::Vector(items) = payload.as_ref() else {
        return Err(invalid(field, "read-construct payload must be a vector"));
    };
    let Some((Expr::Symbol(class), args)) = items.split_first() else {
        return Err(invalid(
            field,
            "read-construct payload must start with a class symbol",
        ));
    };
    Ok((class.clone(), args))
}

/// Builds the DCE-safe registry containing every interference record Citizen.
pub fn interference_citizen_registry() -> Result<CitizenRegistry> {
    let mut registry = CitizenRegistry::new();
    registry
        .register::<MediumDescriptor>()?
        .register::<EmitterDescriptor>()?
        .register::<ProblemDescriptor>()?
        .register::<PlaneDescriptor>()?
        .register::<SamplingCertificateDescriptor>()?
        .register::<WorkEstimateDescriptor>()?
        .register::<PhasorFieldDescriptor>()?
        .register::<StudyEvidenceDescriptor>()?
        .register::<StudyDescriptor>()?
        .register::<ProjectionCertificateDescriptor>()?
        .register::<ProjectionRequestDescriptor>()?
        .register::<ScalarProjectionDescriptor>()?;
    Ok(registry)
}

/// Loadable class and Shape registrations for interference records.
pub struct InterferenceRecordsLib;

impl Lib for InterferenceRecordsLib {
    fn manifest(&self) -> LibManifest {
        let mut exports = RECORD_SYMBOLS
            .iter()
            .map(|symbol| Export::Class {
                symbol: sim_citizen::parse_symbol(symbol),
                class_id: None,
            })
            .collect::<Vec<_>>();
        exports.extend(
            crate::shapes::interference_shape_symbols()
                .into_iter()
                .map(|symbol| Export::Shape {
                    symbol,
                    shape_id: None,
                }),
        );
        LibManifest {
            id: records_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports,
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        interference_citizen_registry()?.install_all(linker)?;
        crate::shapes::register_interference_shapes(linker)
    }
}

/// Installs the record classes and Shapes once in `cx`.
pub fn install_interference_records(cx: &mut Cx) -> Result<()> {
    if cx.registry().lib(&records_lib_symbol()).is_none() {
        cx.load_lib(&InterferenceRecordsLib)?;
    }
    Ok(())
}

fn records_lib_symbol() -> Symbol {
    Symbol::qualified("sim", "interference-records")
}

const RECORD_SYMBOLS: [&str; 12] = [
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
