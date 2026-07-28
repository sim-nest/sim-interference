//! Thin index-check launcher: defers to the shared sim-tooling checker.

use std::env;
use std::process::Command;

pub fn run(args: Vec<String>) -> Result<(), String> {
    let program = args.first().map(String::as_str).unwrap_or("xtask");
    if args.get(1).map(String::as_str) != Some("index-check") {
        return Err(format!("usage: {program} index-check [--strict SPEC]"));
    }

    let root = env::current_dir().map_err(|err| format!("current dir: {err}"))?;
    let manifest = crate::simdoc::locate_sim_tooling_manifest(&root)?;
    let mut command = Command::new("cargo");
    command.args(["run", "--manifest-path"]);
    command.arg(manifest);
    command.args(["--quiet", "--", "index-check", "--repo"]);
    command.arg(&root);
    for arg in args.iter().skip(2) {
        command.arg(arg);
    }

    let status = command
        .status()
        .map_err(|err| format!("run shared index checker: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("shared index checker failed with status {status}"))
    }
}
