# sim-interference

`sim-interference` is the public SIM repository for certified coherent scalar
wave-field studies. `sim-lib-interference-core` owns checked physical
quantities, the exact homogeneous scalar propagation model, finite physical
sample planes, sampling certificates, and allocation-free work preflight.
`sim-lib-interference-solve` owns the deterministic CPU `f64` reference, its
analytic, metamorphic, time-sign, and Helmholtz convergence verification, and
certified scalar observation and detector reduction.
`sim-lib-interference-compute` lowers the same admitted model into normalized
tile-local `f32` operations over the canonical `TensorExecutor`, provides the
portable dense CPU baseline, and reports fixed-tolerance differential evidence
against the `f64` oracle without defining a provider or device API.
`sim-lib-interference-runtime` projects those checked values into Tensor-backed
Citizen records and registers the public Problem, Plane, Study,
ProjectionRequest, and Projection Shapes. It reuses the canonical Tensor and
general-purpose codecs; the repository does not claim an accelerated provider
or renderer.

Every coherent problem has one frequency and a canonical source set. The
time convention `u=Re{U exp(-i omega t)}`, outgoing spatial sign
`U~exp(+i k r)`, point-source `1/r` spreading, attenuation, singularity
radius, and forward-plane half-space are explicit.
Sampling certificates measure carrier wavelength, worst-case
half-wavelength squared-magnitude fringes, and point-source envelope change.
Strict requests require a resolved certificate; annotated requests preserve
the warning. Work budgets independently bound cells, emitter evaluations,
host and result bytes, and certificate stencil work before field allocation.
The reference solver then validates every cell/source geometry before
allocation, traverses cells row-major and sources in stable-id order, and uses
independent Neumaier compensation for real and imaginary components. A
successful result carries the complete preflight evidence; any failure returns
no field. The verification suite checks closed-form identities, true
metamorphic laws, crest direction, and second-order convergence of the complex
Helmholtz residual on common physical points at two stencil spacings.
The normalized `f32` lowering keeps executor trigonometry in `[-pi, pi]`,
partitions awkward and multi-segment fields before submission, and compares
real, imaginary, amplitude, above-floor wrapped phase, and squared magnitude
under one fixed absolute-plus-relative contract. Its checked distance sweep
keeps normalized phase error bounded while naive absolute-f32 error grows
across 1, 10, 100, and 1000 metres.
Projection derives real, imaginary, amplitude, wrapped phase, normalized
squared magnitude, and instantaneous fields without rerunning propagation.
Undefined phase is structurally masked. Detail mode refuses loss, while named
complex, scalar-area, and squared-magnitude-area detectors integrate an exact
partition of every source cell and return their own projection certificate.
Field analysis adds deterministic statistics, strict local node/antinode
candidates, and optional Michelson contrast while retaining both sampling
truth and projection identity. Bounded scenario builders cover two point
sources, inward counter-propagating planes, progressive-phase line arrays, and
rectangular discrete apertures. Array/aperture amplitude is normalized across
elements, active spacing is reported in wavelengths, and strict construction
refuses spacing above `lambda/2`.

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
