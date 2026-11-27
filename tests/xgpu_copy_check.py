# xgpu_copy_check: test of the CUDA stand-in's refusal of GPU-to-GPU copies inside one process
# (patch 0007). Such a copy (torch `t.to("cuda:1")` from cuda:0) would otherwise be timed as a copy
# inside one GPU, with no link between the GPUs. Run on one server with 2 GPUs:
#
#   --case nccl        every rank all-reduces on its own GPU. Must run to the end.
#   --case same_gpu    as nccl, plus rank 0 copies a tensor to another tensor on its own GPU
#                      (a device-to-device copy that is legitimately inside one GPU). Must run to the end.
#   --case to_other    as nccl, plus rank 0 copies a tensor from its GPU to the next GPU with
#                      `t.to(...)`. Must stop with "PHANTORA: copy of ... from GPU 0 to GPU 1".
#   --case copy_other  the same with `dst.copy_(src)` between preallocated tensors on two GPUs.
#
# Follows scripts/README.md (tracer, mark_step, mock data, no control flow on values).
import sys
sys.path.insert(0, "/phantora/tests")
from phantora_utils import mark_step, time_pair, enable_function_tracer, disable_function_tracer

import argparse
import os

import torch
import torch.distributed as dist

CASES = ["nccl", "same_gpu", "to_other", "copy_other"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=CASES)
    ap.add_argument("--mb", type=float, default=64.0, help="message / copy size, MB of bf16")
    ap.add_argument("--iterations", type=int, default=10)
    a = ap.parse_args()

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    other = torch.device("cuda", (local + 1) % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", device_id=dev)

    n = int(a.mb * 1e6 / 2)
    bf16 = torch.bfloat16
    ar = torch.empty(n, device=dev, dtype=bf16)
    src = torch.empty(n, device=dev, dtype=bf16)
    same_dst = torch.empty(n, device=dev, dtype=bf16)
    other_dst = torch.empty(n, device=other, dtype=bf16) if a.case == "copy_other" and rank == 0 else None
    if rank == 0:
        print(f"xgpu case {a.case}: {n * 2 / 1e6:.1f} MB, world {world}, this rank on {dev}, other {other}, "
              f"devices visible {torch.cuda.device_count()}")

    enable_function_tracer()
    duras = []
    for i in range(a.iterations):
        mark_step(i)
        start, _ = time_pair()
        dist.all_reduce(ar)
        if rank == 0:
            if a.case == "same_gpu":
                same_dst.copy_(src)
            elif a.case == "to_other":
                src.to(other)
            elif a.case == "copy_other":
                other_dst.copy_(src)
        torch.cuda.synchronize()
        end, _ = time_pair()
        duras.append(end - start)
        print(f"rank {rank} iter {i} time: {end - start:.6f}\n", end="")
    mark_step(a.iterations)
    disable_function_tracer()
    steady = duras[2:] or duras
    print(f"Rank {rank} Time: {duras} Avg Time: {sum(steady) / len(steady):.6f}\n", end="")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
