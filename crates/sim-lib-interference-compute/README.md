# sim-lib-interference-compute

Normalized tile-local `f32` Tensor lowering for coherent scalar-wave fields.

The lowering preserves the core's homogeneous, isotropic, three-dimensional
scalar free-field scope. It does not add polarization, impedance, interfaces,
obstacles, or diffraction. Amplitude-squared differential metrics are
normalized observables, not physical intensity, power, or energy.

The crate preflights a complete interference problem and physical sampling
plane before submitting any work. It partitions the plane under explicit
phase, element, segment, tensor-byte, and result-byte limits. Host `f64`
geometry supplies a phase anchor and gain at each tile center; device-visible
Tensors contain only bounded offsets from that center.

Point sources use the cancellation-safe normalized distance residual

```text
delta = (2*a + b*rho) / (sqrt(1 + 2*a*rho + b*rho*rho) + 1)
```

and every source uses host angle addition so executor `sin` and `cos` receive
only residual phase in `[-pi, pi]`. Contributions are accumulated with a
deterministic pairwise tree. The lowering submits only the canonical open
Tensor operations `add`, `sub`, `mul`, `div`, `sqrt`, `exp`, `sin`, and `cos`;
it owns no executor, device API, shader, queue, or storage type.

`LoweringPlan::preflight` rejects unsupported operators and dtype, invalid
shapes, singular point samples, samples behind forward planes, unstable
denominators, allocation excess, and phase/error-budget excess before the
captured executor receives its first request.

`solve_dense_f32_cpu` executes that exact lowering through the canonical CPU
`TensorExecutor` and assembles its tiles into a finite row-major
`DenseF32Field`. `DenseExecutionEvidence` retains the executor, tile and
segment counts, flushes, admitted phase limit, predicted tile-radius bound,
observed maximum residual phase, and predicted geometry and arithmetic errors.

`compare_dense_to_reference` applies one published fixed comparison contract:

| quantity | absolute | relative |
| --- | ---: | ---: |
| real and imaginary | `2e-5` | `2e-4` |
| amplitude | `2e-5` | `2e-4` |
| wrapped phase above amplitude `1e-5` | `3e-4` | `1e-4` |
| squared magnitude | `4e-5` | `4e-4` |

`compare_materialized_to_reference` applies that identical report to modeled
and physical-provider results after the two explicit final component
materializations. `HardwareEvidenceReport` binds the result to a sanitized
adapter/profile id, phase bound, tile and segment counts, lifecycle counters,
and exactly 100 same-profile repeats. A measured pass requires bit-identical
component planes across all repeats in addition to the differential
tolerances. `not-measured` is an explicit result and can never satisfy the
hardware gate.

The checked tilted-plane sweep records why normalized reduction is required:

| distance (m) | observed max abs psi | normalized phase error | naive absolute-f32 phase error | predicted bound | tiles | worst cell |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `6.234665588e-2` | `2.623882128e-8` | `3.152816520e-7` | `6.543099468e-1` | 1 | `(0,1)` |
| 10 | `6.234665588e-2` | `9.706112192e-9` | `5.229437373e-6` | `6.543099468e-1` | 1 | `(0,1)` |
| 100 | `6.234665588e-2` | `1.830839347e-8` | `6.098604535e-5` | `6.543099468e-1` | 1 | `(0,4)` |
| 1000 | `6.234665588e-2` | `3.277787286e-8` | `2.844246718e-4` | `6.543099468e-1` | 1 | `(0,1)` |

Both phase bounds remain below pi. Normalized error stays bounded while the
falsified absolute-f32 formulation grows every decade and is more than 8,000
times worse at 1000 m.

`InterferenceComputeLib` composes this lowering with the runtime's narrow
`StudySolver` seam. A `TensorSite` binds its executor in the active child
environment; only then does `TensorStudySolver` supersede the reference CPU
solver. Missing executors, operations, f32 support, crossover work, or a
single-resident-output layout select CPU before submission. Once a provider is
selected, a failure is returned without a CPU restart.

The resident adapter uploads the three immutable coordinate planes once,
chains every intermediate at the selected site, and returns real and imaginary
as the final resident Tensors. `StudyEvidence` carries the provider,
adapter/profile, upload/submission/segment counts, zero intermediate
materializations, two final component materializations, and fixed f32
tolerances. Resident storage owns synchronized observation caching, so repeated
host projection does not repeat readback.

The checked provider matrix exercises the same f64 oracle, dense f32 baseline,
modeled site, and opt-in wgpu site over attenuated multi-segment fields,
long-world normalized phase, exact cancellation, and measured crossover edges.
Explicit wgpu absence exports no site. The automatic provider's `auto/cpu`
decision is recognized before the first Tensor submission and retained as
`ProviderCpuChoice`; accepted provider failures still never restart on CPU.

Set `SIM_INTERFERENCE_WGPU_PHYSICAL=1` only when a probe-backed adapter should be
measured. Headless default tests emit the bounded `not-measured` recipe evidence
without probing hardware or turning absence into a passing hardware claim.
