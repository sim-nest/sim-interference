# sim-lib-interference-runtime

In one line: this crate makes certified interference studies ordinary, Shape-checked SIM values without giving up tensor residency or provenance.

Scope limit: the carried model is homogeneous, isotropic, three-dimensional,
scalar, and free-field. It does not add polarization, impedance, interfaces,
obstacles, or diffraction. Amplitude-squared values are normalized
observables, not physical intensity, power, or energy.

## What it gives you

- Citizen records for problems, planes, fields, studies, evidence, and
  projections.
- A loadable `InterferenceLib` with Shape-checked constructors, provider-routed
  solve, propagation-free project and analysis, bounded scenarios, and
  multi-tone composition.
- A narrow `StudySolver` provider seam resolved from the child environment
  before the registry, with the deterministic reference CPU solver as default.
- Two canonical Tensors for phasor components and one canonical Tensor for a
  scalar projection.
- Fail-closed Shape and codec boundaries for dimensions, physical units,
  evidence counts, and phase masks.
- An exact checked Lisp recipe for a 1024-by-1024 two-source cancellation study
  through the local evaluation fabric.

## Why you will be glad

Host reference buffers cross into Tensor storage once. Resident results remain
resident until an explicit host materialization asks each component for one
readback. `interference/project` operates only on that completed field, so
views cannot accidentally trigger propagation. Lisp, JSON, and binary transport
the same read-construct expression graph; there is no interference-specific
parser or serializer.

## Where it fits

The dependency-free interference model and reference solver sit below this
crate. The library is the standard runtime boundary; accelerator and remote
providers extend the `StudySolver` seam without changing its functions or
Citizen records.
