# sim-lib-interference-solve

In one line: this crate turns an admitted coherent wave problem into a
deterministic host phasor field with inseparable solve evidence.

## What it gives you

`ReferencePhasorSolver` produces separate row-major real and imaginary `f64`
component planes. `SolveEvidence` retains the sampling policy, certificate,
and complete work estimate used to admit the request.

## Why you will be glad

- The host field has one explicit two-dimensional shape.
- Result components remain separate until a caller deliberately projects them.
- Solver configuration is immutable and explicit.
- No SIM runtime, tensor, compute provider, or surface dependency is pulled in.

## Where it fits

This is the executable CPU reference between the physical core and later
runtime or accelerated adapters.
