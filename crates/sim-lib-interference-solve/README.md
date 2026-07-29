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

The solver executes a homogeneous, isotropic, three-dimensional scalar
free-field model. It does not model polarization, impedance, interfaces,
obstacles, or diffraction. Its amplitude-squared projections and detector
reductions are normalized observables, not physical intensity, power, or
energy.

Scalar projection provides real, imaginary, amplitude, phase, normalized
squared magnitude, and instantaneous observations. Phase at or below its
declared amplitude floor is represented as a mask, never a number. Reduction
uses explicit complex-mean, scalar-area, or squared-magnitude-area detector
rules over every source cell exactly once; detail mode refuses a smaller
target. Every scalar result carries source and target dimensions, footprint,
loss class, source sampling evidence, and mask count.

`analyze_fringes` accepts only amplitude-like projections and returns
deterministic field statistics, strict eight-neighbour node/antinode
candidates, and optional Michelson contrast. Every report keeps both its
unchanged sampling certificate and its exact projection identity. Contrast is
absent when the whole field lies at or below the caller's amplitude floor.

`ScenarioBuilder` creates bounded two-point, inward counter-propagating plane,
progressive-phase line-array, and rectangular discrete-aperture problems.
Counts and generated identity storage are checked before allocation.
Array/aperture total amplitude is divided equally among elements; certificates
report active neighbour spacing in wavelengths. Strict aperture policy refuses
spacing above `lambda/2`, while annotate policy preserves the approximation.
The crate has no SIM runtime, tensor, compute-provider, or presentation
dependency.
