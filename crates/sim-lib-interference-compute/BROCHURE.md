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

## Why you will be glad

Absolute world coordinates and absolute propagation phase never enter an `f32`
Tensor. Missing operations, singular geometry, forward-plane violations, unsafe
denominators, and impossible allocations are discovered before accepted device
work can make fallback ambiguous.

## Where it fits

This is the domain-specific lowering leaf between checked interference records
and any canonical `TensorExecutor`. The CPU, modeled, wgpu, and vendor providers
remain unchanged and interchangeable.
