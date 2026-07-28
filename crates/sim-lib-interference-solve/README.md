# sim-lib-interference-solve

Deterministic CPU `f64` reference solving for coherent scalar wave fields.

The crate consumes the checked, dependency-free model and preflight records
from `sim-lib-interference-core`. It stores host results as separate row-major
real and imaginary component planes and carries the exact sampling and work
evidence that admitted each complete field. It has no SIM runtime, tensor,
compute-provider, or presentation dependency.
