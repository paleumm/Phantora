# Entries filled from another GPU

The profiler recorded these kernels as 0 ns on b300_v2 in two separate timing runs (it missed the kernel's activity record), although they take real time on every other GPU. They were filled with the closest sibling GPU's measured time.

| key | nanos | taken from |
| --- | --- | --- |
| `{"Add":[["bf16",[1,64,1024,1024]],["bf16",[1,1,1024,1024]]]}` | 160528 | b200_v2 |
| `{"Generic":{"args":[{"T":["b8",[1,1024]]},{"T":["b8",[1,1024]]}],"name":"aten::__or__"}}` | 1376 | b200_v2 |

# Entries timed with a newer PyTorch

PyTorch 2.9.1 (the version every other entry was timed with) runs `aten::_fused_adamw_`
about 7x slower on B300 than on B200: its multi-tensor kernel has no sm_103 build and
launches 14 times per call. Measured on a Modal B300 for W7's 99-tensor optimizer step:
13.3 ms with 2.9.1, 1.2 ms with 2.14.1, versus 1.8 ms on B200 with 2.9.1. Every other op
in this DB is within a few percent of b200_v2, and plain copies reach 6.8 TB/s on both
versions. Anyone training on B300 would use a PyTorch that supports it, so these two keys
were re-timed with PyTorch 2.14.1 (cu130), same bench.py, `PERFDB_ALIAS=0`
(`PERFDB_TORCH=2.14.1` in modal/record_perfdb.py).

| key | 2.9.1 nanos | 2.14.1 nanos |
| --- | --- | --- |
| `aten::_fused_adamw_` over 99 tensors (315.8M params) | 15406103 | 1679854 |
| `aten::_fused_adamw_` over 57 tensors (6.2M params) | 666872 | 74661 |
