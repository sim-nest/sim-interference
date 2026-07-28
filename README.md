# sim-interference

`sim-interference` is the public SIM repository for certified coherent scalar
wave-field studies. `sim-lib-interference-core` owns checked physical
quantities, the exact homogeneous scalar propagation model, finite physical
sample planes, sampling certificates, and allocation-free work preflight.
`sim-lib-interference-solve` owns the deterministic CPU `f64` reference. The
repository does not claim a runtime tensor adapter, accelerated provider, or
renderer.

Every coherent problem has one frequency and a canonical source set. The
outgoing `exp(-i omega t)` convention, point-source `1/r` spreading,
attenuation, singularity radius, and forward-plane half-space are explicit.
Sampling certificates measure carrier wavelength, worst-case
half-wavelength squared-magnitude fringes, and point-source envelope change.
Strict requests require a resolved certificate; annotated requests preserve
the warning. Work budgets independently bound cells, emitter evaluations,
host and result bytes, and certificate stencil work before field allocation.
The reference solver then validates every cell/source geometry before
allocation, traverses cells row-major and sources in stable-id order, and uses
independent Neumaier compensation for real and imaginary components. A
successful result carries the complete preflight evidence; any failure returns
no field.

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
