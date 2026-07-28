# Contributing

SIM is human-directed and AI-executed. Keep crate entrypoints thin, preserve
validated boundaries, and put behavior in the crate that owns it.

Run the repository validation block from `README.md` before proposing changes.
Generated documentation must be refreshed with `cargo run -p xtask -- simdoc`
and committed whenever its inputs change.
