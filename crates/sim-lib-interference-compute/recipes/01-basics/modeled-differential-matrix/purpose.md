# Compare modeled resident compute with the CPU oracle

Run the complete differential matrix through the headless modeled executor. The
same deterministic `f64` oracle and fixed absolute-plus-relative tolerances
check dense and modeled `f32` results for attenuation, segmentation, long-world
coordinates, exact cancellation, and crossover edges.

The checked harness also proves that final component observation is cached and
that the modeled path produces the same report shape as the portable dense
baseline.

Both paths execute the homogeneous, isotropic, three-dimensional scalar
free-field model. They do not add polarization, impedance, interfaces,
obstacles, or diffraction. Amplitude-squared metrics are normalized
observables, not physical intensity, power, or energy.
