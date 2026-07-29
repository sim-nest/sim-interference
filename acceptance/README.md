# Physical interference acceptance

This directory holds the public, sanitized proof that the resident interference
feature runs through physical wgpu compute on every GPU_MATH_5 target profile.
The committed reports are:

- `5080-laptop-v1.sx`
- `5090-v1.sx`
- `ryzen-ai-max-395-v1.sx`

Each report names the exact measured source commit and sanitized GPU_MATH_5
profile provenance. Per-case records retain cell, source, tile, and segment
counts; maximum normalized phase; fixed component and phase tolerances; observed
differential maxima; and the zero-intermediate/two-final materialization
lifecycle. They also retain the driver/backend and the power and thermal context
available from the originating GPU_MATH_5 profile.

`run-physical.sh capture` is intended for the registered control-plane target
runner. It selects the requested physical adapter, executes the fixed
100-repeat wgpu matrix, checks projection and fail-closed routing, emits only
the sanitized report, and verifies it before returning success. The two larger
targets additionally run a three-repeat 16,641-cell capacity probe, above the
RTX 5080 Laptop profile's measured 16,384-cell crossover. Keeping that probe
separate prevents capacity scaling from weakening or silently lengthening the
100-repeat determinism matrix. Capacity-probe reports preserve the observed
determinism boolean but do not turn it into a requirement; the probe's claim is
successful above-crossover resident execution within the fixed numerical
tolerances. The Ryzen report describes segmented resident execution; it does
not treat the machine's total 128 GiB memory as one GPU buffer.

Verify a committed report against its measured source:

```text
sh acceptance/run-physical.sh verify \
  --source d0efde42c9a56fac1a6bc23e94a4a6ad835b2780 \
  acceptance/5080-laptop-v1.sx
```

The verifier rejects missing required cases, non-physical or non-passing
evidence, absent larger-target crossover proof, and common private identity or
path fields.
