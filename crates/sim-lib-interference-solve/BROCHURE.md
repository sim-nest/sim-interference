# sim-lib-interference-solve

In one line: this crate turns an admitted coherent wave problem into a
deterministic host phasor field with inseparable solve evidence.

Scope limit: the solved model is homogeneous, isotropic, three-dimensional,
scalar, and free-field. It does not claim polarization, impedance, interfaces,
obstacles, or diffraction. Amplitude-squared outputs are normalized
observables, not physical intensity, power, or energy.

## What it gives you

`ReferencePhasorSolver` produces separate row-major real and imaginary `f64`
component planes. `SolveEvidence` retains the sampling policy, certificate,
and complete work estimate used to admit the request. `project` and
`reduce_for_view` derive certified scalar fields without rerunning propagation.
`analyze_fringes` adds deterministic statistics, strict local node/antinode
candidates, and Michelson contrast while retaining sampling and projection
identity. `ScenarioBuilder` provides bounded canonical point, plane, phased
array, and discrete-aperture fixtures.

## Why you will be glad

- The host field has one explicit two-dimensional shape.
- Result components remain separate until a caller deliberately projects them.
- Undefined phase is a masked cell, never numeric zero.
- Detector rules distinguish coherent complex means, amplitude area means, and
  squared-magnitude area means.
- Odd and non-divisible grids cover every source cell exactly once.
- Every scalar result retains its reduction footprint, loss class, source
  sampling certificate, and mask count.
- Every analysis report states both what was sampled and which scalar
  projection was analyzed.
- Array and aperture amplitudes are normalized, wavelength-relative spacing is
  explicit, and strict policy rejects spacing above `lambda/2`.
- Source count and generated identity limits are checked before allocation.
- Solver configuration is immutable and explicit.
- No SIM runtime, tensor, compute provider, or surface dependency is pulled in.

## Where it fits

This is the executable CPU reference between the physical core and later
runtime or accelerated adapters.
