# CUDA lifecycle contract

The runtime owns CUDA buffer lifetime. Every CUDA release path must first
restore or verify the device primary context before calling an asynchronous
free operation. This applies to allocation-pool eviction, retired buffers,
staging buffers, explicit buffer destruction, and final engine teardown.

Applications must not destroy or reset the CUDA primary context while an
`AOTEngine` instance or one of its buffers is still alive. If an external
component invalidates the context, the runtime must fail closed and recreate
the backend before accepting new work; it must not issue `cuMemFreeAsync` on
an invalid context.

Validation records for lifecycle changes must include the backend, device,
shape, dtype, exact command, explicit teardown result, and any allocator or
driver error. A successful kernel run alone is not sufficient evidence for
safe cleanup.
