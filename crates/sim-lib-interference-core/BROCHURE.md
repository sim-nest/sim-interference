# sim-lib-interference-core

In one line: this crate validates scalar-wave physics, sampling truth, and
bounded work before solving begins.

Scope limit: the model is homogeneous, isotropic, three-dimensional, scalar,
and free-field. It does not claim polarization, impedance, interfaces,
obstacles, or diffraction. Downstream amplitude-squared values are normalized
observables, not physical intensity, power, or energy.

## What it gives you

Focused `f64` wrappers feed one coherent source model and exact Green-function
convention. `SamplingPlane` defines finite pixel centres;
`SamplingCertificate` records carrier, power-fringe, and point-envelope
adequacy; `RequestPreflight` applies sampling policy and `WorkBudget`.

## Why you will be glad

- Sampling thresholds are data carried by the certificate, not hidden UI
  policy.
- Strict requests fail closed with the exact measurements that caused refusal.
- Every count and byte product uses checked arithmetic.
- No general units, grid, tensor, solver, or presentation dependency is pulled
  into the base.

## Where it fits

Interference solver and runtime layers consume this crate's admitted records.
It has no SIM dependency and owns no tensor, compute, runtime, codec, or view
behavior.
