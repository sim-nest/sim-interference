# sim-lib-interference-compute

In one line: accurate coherent-wave geometry is reduced on the host so ordinary
portable `f32` Tensor operations can evaluate it without large-world phase loss.

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
