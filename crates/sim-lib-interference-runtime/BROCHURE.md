# sim-lib-interference-runtime

In one line: this crate makes certified interference studies ordinary,
Shape-checked SIM values without giving up tensor residency or provenance.

## What it gives you

- Citizen records for problems, planes, fields, studies, evidence, and
  projections.
- Two canonical Tensors for phasor components and one canonical Tensor for a
  scalar projection.
- Fail-closed Shape and codec boundaries for dimensions, physical units,
  evidence counts, and phase masks.

## Why you will be glad

Host reference buffers cross into Tensor storage once. Resident results remain
resident until an explicit host materialization asks each component for one
readback. Lisp, JSON, and binary transport the same read-construct expression
graph; there is no interference-specific parser or serializer.

## Where it fits

The dependency-free interference model and reference solver sit below this
crate. Loadable interference operations and compute lowering build above it.
