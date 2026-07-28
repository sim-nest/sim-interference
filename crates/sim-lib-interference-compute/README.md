# sim-lib-interference-compute

Normalized tile-local `f32` Tensor lowering for coherent scalar-wave fields.

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
