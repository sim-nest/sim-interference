# sim-interference

In one line: `sim-interference` makes the physical inputs to coherent
wave-field studies explicit and impossible to construct with invalid numeric
boundaries.

## What it gives you

The first public layer provides finite metres, positive distances, frequencies,
propagation speeds, non-negative attenuation, normalized phase, and
non-negative field amplitudes. Each type owns its physical admission rule and
keeps the raw `f64` private.

## Why you will be glad

- Reject NaN and infinity before they can contaminate a study.
- Keep zero legal only where the physical contract permits it.
- Compare and display phase through one canonical interval.
- Receive stable quantity names in diagnostics at every boundary.

## Where it fits

This is the dependency-free base of the interference family. It does not yet
claim a wave model, solver, tensor runtime adapter, compute provider, or view
surface; those layers arrive only with their own checked behavior and evidence.
