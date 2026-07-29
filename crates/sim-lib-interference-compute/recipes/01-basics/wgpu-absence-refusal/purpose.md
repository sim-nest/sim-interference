# Refuse an unavailable wgpu site before submission

Resolve an explicitly requested wgpu site from an empty discovery result. The
request fails before the interference solver selects a provider or submits
Tensor work. By contrast, the separate automatic compute provider may choose
its declared CPU route before submission.

This headless recipe distinguishes explicit placement failure from a provider's
pre-submission fallback policy without probing hardware.

Placement does not broaden the homogeneous, isotropic, three-dimensional
scalar free-field model. Polarization, impedance, interfaces, obstacles, and
diffraction remain outside scope, and amplitude-squared metrics remain
normalized rather than physical intensity, power, or energy.
