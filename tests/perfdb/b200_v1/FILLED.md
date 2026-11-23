# Entries filled from another GPU

The profiler recorded these kernels as 0 ns on b200_v1 in two separate timing runs (it missed the kernel's activity record), although they take real time on every other GPU. They were filled with the closest sibling GPU's measured time.

| key | nanos | taken from |
| --- | --- | --- |
| `{"ZerosLike":["f32",[32768]]}` | 831 | h200_v1 |
