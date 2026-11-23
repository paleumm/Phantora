# Entries filled from another GPU

The profiler recorded these kernels as 0 ns on b300_v2 in two separate timing runs (it missed the kernel's activity record), although they take real time on every other GPU. They were filled with the closest sibling GPU's measured time.

| key | nanos | taken from |
| --- | --- | --- |
| `{"Add":[["bf16",[1,64,1024,1024]],["bf16",[1,1,1024,1024]]]}` | 160528 | b200_v2 |
| `{"Generic":{"args":[{"T":["b8",[1,1024]]},{"T":["b8",[1,1024]]}],"name":"aten::__or__"}}` | 1376 | b200_v2 |
