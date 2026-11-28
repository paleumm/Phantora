# Phantora performance database

- **GPU:** NVIDIA H100 80GB HBM3
- **Schema version:** 1
- Recorded by tests/perfdb/bench.py (stock PyTorch 2.9.1, no Phantora build) on Modal, from the cluster-sim v1 key set (3,173 compute + 207 memcpy shapes from W1-W7 at 5 layouts; see github.com/paleumm/cluster-sim).
- **v2:** distinct operand buffers (`PERFDB_ALIAS=0`), wrapped Python scalars on the CPU, generic floats replayed as 2.0. Kernels the profiler dropped (0 ns here, non-zero on other GPUs) were re-timed once.
- Later shapes were added with `bench.py --merge` on the same GPU type.
- 2026-11-26 (MSH-355 F22, F6): 261 compute + 3 memcpy keys added with `bench.py` on a Modal `H100!` (stock PyTorch 2.9.1, `PERFDB_ALIAS=0`), appended without changing any existing entry: gpt-oss balanced expert routing (both `experts=` options, incl. `Cuda(1)` variants for 1x2) and autocast casts (`ConvDType` f32->bf16 [14336,4096], fp32 `Linear` [1,1024,4096]x[14336,4096]). The 21 zero entries are views/no-ops (`aten::reshape`, same-dtype casts, same-device `aten::to`), zero in every v2 DB. Nothing filled by hand.
