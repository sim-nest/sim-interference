# sim-interference

In one line: `sim-interference` produces deterministic coherent scalar-wave
fields without separating the answer from its sampling and work evidence.

Scope limit: the model is homogeneous, isotropic, three-dimensional, scalar,
and free-field. It does not claim polarization, impedance, interfaces,
obstacles, or diffraction. Amplitude-squared results are normalized
observables, not physical intensity, power, or energy.

## What it gives you

The dependency-free crates provide checked physical quantities, canonical
point and forward-plane sources, an exact outgoing propagation convention,
finite physical pixel-centre planes, truth-carrying sampling certificates,
checked work budgets, and a deterministic CPU `f64` reference field. That
field can be observed as real, imaginary, amplitude, honest masked phase,
normalized squared magnitude, or instantaneous time, then reduced through a
named detector with complete projection provenance. The runtime crate turns
those values into fail-closed Citizen records and Shapes while storing phasors
in the canonical Tensor implementation. The compute crate lowers the same
model into bounded tile-local `f32` Tensor operations, supplies a dense CPU
baseline, and reports fixed-tolerance conformance against the `f64` oracle.

## Why you will be glad

- Reject NaN and infinity before they can contaminate a study.
- Refuse aliased carrier/fringe sampling and under-resolved point envelopes by
  default.
- Preserve explicit thresholds and all measured numbers when annotation is
  selected.
- Bound cells, source evaluations, bytes, and certificate work before output
  allocation.
- Reproduce every real and imaginary result bit through canonical source order,
  row-major traversal, and independent compensated accumulation.
- Receive a complete certified component-plane field or no field.
- Refuse lossy detail decimation and wrapped-angle averaging.
- Integrate every source cell exactly once under an explicit detector rule.
- Carry dimensions, detector footprint, loss class, source sampling evidence,
  and phase mask count with every scalar projection.
- Keep executor phase bounded as world distance grows, with a checked sweep
  that falsifies naive absolute-f32 phase.

## Where it fits

This repository owns the dependency-free physical and CPU-reference boundary,
the Tensor-backed runtime records, and the provider-neutral normalized compute
leaf. It owns no compute provider, device API, shader, fallback router, or view
surface.
