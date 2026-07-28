# sim-lib-interference-solve

In one line: this crate turns an admitted coherent wave problem into a
deterministic host phasor field with inseparable solve evidence.

## What it gives you

`ReferencePhasorSolver` produces separate row-major real and imaginary `f64`
component planes. `SolveEvidence` retains the sampling policy, certificate,
and complete work estimate used to admit the request. `project` and
`reduce_for_view` derive certified scalar fields without rerunning propagation.

## Why you will be glad

- The host field has one explicit two-dimensional shape.
- Result components remain separate until a caller deliberately projects them.
- Undefined phase is a masked cell, never numeric zero.
- Detector rules distinguish coherent complex means, amplitude area means, and
  squared-magnitude area means.
- Odd and non-divisible grids cover every source cell exactly once.
- Every scalar result retains its reduction footprint, loss class, source
  sampling certificate, and mask count.
- Solver configuration is immutable and explicit.
- No SIM runtime, tensor, compute provider, or surface dependency is pulled in.

## Where it fits

This is the executable CPU reference between the physical core and later
runtime or accelerated adapters.
