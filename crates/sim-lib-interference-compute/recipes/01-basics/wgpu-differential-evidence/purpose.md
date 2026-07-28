# Bounded wgpu differential evidence

Use the unchanged `interference/solve` expression at a probe-backed wgpu
`TensorSite`. The checked harness compares the materialized result with the
same deterministic f64 oracle and fixed report used by dense f32 and modeled
execution. It covers attenuation, planned multi-segment fields, long world
distance, exact cancellation, and measured crossover edges.

Physical execution is deliberately opt-in with
`SIM_INTERFERENCE_WGPU_PHYSICAL=1`. A qualifying adapter repeats every case 100
times with one unchanged adapter/profile and requires bit-identical component
planes, the published differential tolerances, zero intermediate
materializations, and exactly two final component materializations.

Without that opt-in, or without a qualifying adapter, the bounded evidence is
`not-measured`. This is useful headless contract evidence, but it never
satisfies a hardware acceptance gate and is never reported as a pass.
