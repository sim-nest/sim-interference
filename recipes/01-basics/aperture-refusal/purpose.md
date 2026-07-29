# Refuse a sparse aperture approximation

Ask the scenario builder for a two-element array spaced at `0.55 lambda`.
Strict aperture policy refuses the discrete approximation before source
construction because active spacing may not exceed `lambda / 2`.

`AperturePolicy::Annotate` is the explicit alternative when preserving sparse
approximation evidence is more useful than refusal.
