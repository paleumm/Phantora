# Phantora performance database

- **GPU:** NVIDIA B200
- **Schema version:** 1
- Recorded by tests/perfdb/bench.py (stock PyTorch, no Phantora build) from the cluster-sim v1 key set (3,173 compute + 207 memcpy shapes from W1-W7 at 5 layouts; see github.com/paleumm/cluster-sim).
- Later shapes were added with `bench.py --merge` on the same GPU type.
