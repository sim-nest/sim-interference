# sim-lib-interference-core

Validated physical boundaries for coherent scalar wave-field studies.

The crate exports checked quantities, one-frequency point and forward-plane
emitters, the exact homogeneous propagation convention, orthonormal finite
sampling planes, carrier/fringe/envelope certificates, and checked work
preflight. `RequestPreflight` classifies sampling and admits every explicit
work dimension before a solver allocates output. The crate owns no solver,
tensor, runtime binding, compute provider, or surface.
