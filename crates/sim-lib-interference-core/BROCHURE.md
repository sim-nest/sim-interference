# sim-lib-interference-core

In one line: this crate prevents invalid scalar-wave inputs from crossing the
interference domain boundary.

## What it gives you

Seven focused `f64` wrappers cover signed coordinates, positive lengths,
positive frequencies and speeds, non-negative attenuation and amplitude, and
canonical phase.

## Why you will be glad

- Non-finite values fail immediately with stable quantity names.
- Physical zero rules are visible in the type constructors.
- Phase comparisons do not depend on how many turns a caller supplied.
- No general units framework or solver dependency is pulled into the base.

## Where it fits

Future interference model and solver crates depend on this crate. It has no SIM
dependency and owns no tensor, compute, runtime, codec, or view behavior.
