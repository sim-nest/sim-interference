# sim-lib-interference-solve

Deterministic CPU `f64` reference solving for coherent scalar wave fields.

The crate consumes the checked, dependency-free model and preflight records
from `sim-lib-interference-core`. It stores host results as separate row-major
real and imaginary component planes and carries the exact sampling and work
evidence that admitted each complete field. Its fail-closed verification suite
checks analytic propagation identities, reciprocity, linearity, rigid-motion
covariance, global phase, canonical source permutations, the outgoing time
sign, and second-order seven-point Helmholtz residual convergence across point,
plane, mixed, attenuating, and multi-source fields.

Scalar projection provides real, imaginary, amplitude, phase, normalized
squared magnitude, and instantaneous observations. Phase at or below its
declared amplitude floor is represented as a mask, never a number. Reduction
uses explicit complex-mean, scalar-area, or squared-magnitude-area detector
rules over every source cell exactly once; detail mode refuses a smaller
target. Every scalar result carries source and target dimensions, footprint,
loss class, source sampling evidence, and mask count. The crate has no SIM
runtime, tensor, compute-provider, or presentation dependency.
