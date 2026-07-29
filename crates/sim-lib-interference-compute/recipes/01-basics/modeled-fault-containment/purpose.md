# Contain selected-provider faults without CPU restart

Inject out-of-memory, device-loss, deadline, execution, and readback failures at
the modeled `TensorExecutor` seam. Once a provider owns the request, every
failure remains an error with explicit evidence; none silently restarts the
study on the CPU.

The ordinary domain expression below is unchanged. Fault injection belongs to
the checked provider harness, not to the public interference problem or solve
protocol.
