//! Loadable interference runtime library.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, Dependency, Export, Lib, LibManifest, LibTarget, Linker, Result, Symbol, Version,
};

use crate::{
    ReferenceStudySolver, SolverProvider,
    ops::{function_symbols, runtime_functions},
    study_solver_symbol,
};

/// Stable library symbol for the runtime interference operations.
pub fn interference_lib_symbol() -> Symbol {
    Symbol::qualified("sim", "interference")
}

/// Loadable interference runtime operations and default CPU solver.
#[derive(Clone, Copy, Debug, Default)]
pub struct InterferenceLib;

impl Lib for InterferenceLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: interference_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![Dependency {
                id: Symbol::qualified("sim", "interference-records"),
                minimum_version: None,
            }],
            capabilities: Vec::new(),
            exports: std::iter::once(Export::Value {
                symbol: study_solver_symbol(),
            })
            .chain(
                function_symbols()
                    .into_iter()
                    .map(|symbol| Export::Function {
                        symbol,
                        function_id: None,
                    }),
            )
            .collect(),
        }
    }

    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.value(
            study_solver_symbol(),
            SolverProvider::new(Arc::new(ReferenceStudySolver)).into_value()?,
        )?;
        for (symbol, function) in runtime_functions(cx) {
            linker.function_value(symbol, cx.factory().opaque(Arc::new(function))?)?;
        }
        Ok(())
    }
}
