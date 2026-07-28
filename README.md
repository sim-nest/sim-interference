# sim-interference

`sim-interference` is the public SIM repository for certified coherent scalar
wave-field studies. The current foundation is deliberately narrow:
`sim-lib-interference-core` owns checked physical quantity boundaries and
nothing yet claims to solve or render a field.

The quantity vocabulary rejects non-finite values at construction, distinguishes
positive from non-negative domains, and normalizes phase into one documented
interval. Later crates will build the model, reference solver, runtime adapter,
compute lowering, and view surface on these boundaries without introducing a
second tensor or a kernel dependency.

Repository validation:

```bash
cargo fmt --all --check
cargo run -p xtask -- check-file-sizes
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features
cargo run -p xtask -- simdoc --check
```

Refresh generated documentation with:

```bash
cargo run -p xtask -- simdoc
```
