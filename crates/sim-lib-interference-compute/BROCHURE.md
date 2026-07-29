# sim-lib-interference-compute

In one line: accurate coherent-wave geometry is reduced on the host so ordinary
portable `f32` Tensor operations can evaluate it without large-world phase loss.

Scope limit: this lowers the homogeneous, isotropic, three-dimensional scalar
free-field model unchanged. It does not add polarization, impedance,
interfaces, obstacles, or diffraction. Amplitude-squared metrics are
normalized observables, not physical intensity, power, or energy.

## What it gives you

- Complete fail-closed preflight before the first executor submission.
- Bounded physical tiles selected from phase, element, segment, and byte limits.
- Host `f64` point and forward-plane constants with a local `f32` residual.
- A recorded geometry/roundoff prediction for every source and tile.
- Only canonical open Tensor arithmetic and transcendental requests.
- Stable deterministic pairwise accumulation of source contributions.
- A canonical dense CPU baseline with executor, tile, segment, flush, phase,
  and predicted-error evidence.
- One fixed differential report for components, amplitude, above-floor wrapped
  phase, and squared magnitude against the deterministic `f64` oracle.
- The same differential report for dense, modeled, and materialized wgpu
  results.
- Sanitized adapter/profile evidence with an explicit measured/not-measured
  boundary and a 100-repeat same-profile determinism gate.
- Scoped compute-site routing that leaves the CPU registry default untouched.
- Pre-submission CPU selection for absent or ineligible providers.
- Resident real/imaginary Studies with upload, submission, segment, readback,
  adapter, profile, and tolerance evidence.
- Fail-closed OOM, deadline, device-loss, execution, and readback behavior with
  no CPU restart after provider selection.
- Explicit unavailable-wgpu refusal and automatic pre-submit CPU-choice
  evidence.

## Why you will be glad

Absolute world coordinates and absolute propagation phase never enter an `f32`
Tensor. Missing operations, singular geometry, forward-plane violations, unsafe
denominators, and impossible allocations are discovered before accepted device
work can make fallback ambiguous. The checked 1-to-1000-metre sweep keeps
normalized phase error below `3.3e-8` while naive absolute-f32 error grows past
`2.8e-4`, making the reduction's accuracy contribution measurable.

## Where it fits

This is the domain-specific lowering leaf between checked interference records
and any canonical `TensorExecutor`. The CPU, modeled, wgpu, and vendor providers
remain unchanged and interchangeable.

`InterferenceComputeLib` is the thin provider adapter above that leaf. It
reuses `TensorSite`, the active environment executor, canonical resident Tensor
storage, and the runtime `StudySolver`; it adds no device enum, shader API,
storage type, or kernel surface.

The physical evidence recipe remains headless by default. An opted-in run
measures a probe-backed adapter; absence is printed as `not-measured` and never
counted as hardware acceptance.
