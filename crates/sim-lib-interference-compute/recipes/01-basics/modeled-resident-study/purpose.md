# Modeled resident interference study

Run the ordinary Shape-checked `interference/solve` expression at
`site/compute/model`. `TensorSite` binds its executor only in the realization
environment, where `InterferenceComputeLib` selects `TensorStudySolver`.

The adapter admits the whole request and chooses CPU before submission when
there is no executor, the card lacks an operation, f32 is unavailable, or work
is below the configured crossover. Once selected, provider failure is returned
without a CPU restart. Immutable coordinate planes upload once, arithmetic
stays resident, and only the final real and imaginary components cross to a
host observation. Repeated observation uses the resident storage cache.
