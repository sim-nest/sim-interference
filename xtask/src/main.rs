#![forbid(unsafe_code)]
//! Repository automation wrapper for validation and generated documentation.

mod file_sizes;
mod host_blind;
mod index_check;
mod package_floors;
mod recipes;
mod simdoc;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let program = args.first().map(String::as_str).unwrap_or("xtask");
    let result = match args.get(1).map(String::as_str) {
        Some("simdoc") => simdoc::run(args),
        Some("index-check") => index_check::run(args),
        Some("check-file-sizes") => file_sizes::run(),
        Some("check-host-blind") => host_blind::run(),
        Some("check-recipes") => recipes::run(),
        Some("check-package-floors") => package_floors::run(),
        _ => Err(format!(
            "usage: {program} simdoc [--check] | index-check [--strict SPEC] | check-file-sizes | check-recipes | check-package-floors"
        )),
    };

    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
