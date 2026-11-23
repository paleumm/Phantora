# Phantora performance database

- **GPU:** NVIDIA H100 80GB HBM3
- **Schema version:** 1
- Recorded by tests/perfdb/bench.py (stock PyTorch 2.9.1, no Phantora build) on Modal, from the cluster-sim v1 key set (3,173 compute + 207 memcpy shapes from W1-W7 at 5 layouts; see github.com/paleumm/cluster-sim).
- **v2:** distinct operand buffers (`PERFDB_ALIAS=0`), wrapped Python scalars on the CPU, generic floats replayed as 2.0. Kernels the profiler dropped (0 ns here, non-zero on other GPUs) were re-timed once.
