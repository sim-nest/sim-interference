# sim-lib-interference-runtime

Loadable, Shape-checked runtime operations for coherent interference studies.

`InterferenceLib` exports:

- `interference/problem` and `interference/sampling-plane` for checked physical
  inputs;
- `interference/solve`, resolved through a child-Env-first `StudySolver`
  provider with the deterministic reference CPU solver installed by default;
- `interference/project` and `interference/analyze` for propagation-free
  projections and certified fringe reports; and
- `interference/scenarios` and `interference/multitone` for bounded source
  construction and independently certified tone composition.

The crate projects problems, planes, evidence, Tensor-backed phasor fields, and
scalar projections into Citizen read-constructs. Registered Shapes admit every
public argument and result. Decoding re-runs domain, tensor, evidence, and mask
invariants; it does not add a domain parser or a second grid or tensor
representation.

The embedded `two-source-cancellation` Lisp recipe constructs a strict
1024-by-1024 study and realizes the `interference/Study` through
`core/local-fabric`. Its cargo-example harness runs the same CPU contract and
checks stable evidence counts.
