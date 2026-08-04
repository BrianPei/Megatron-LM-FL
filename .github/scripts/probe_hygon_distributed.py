#!/usr/bin/env python3

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError("BW1000 CUDA-compatible API is unavailable")

    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"Local rank {local_rank} cannot use any of the {device_count} visible devices"
        )

    torch.cuda.set_device(local_rank)
    value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
    torch.cuda.synchronize(local_rank)

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    try:
        dist.all_reduce(value)
        torch.cuda.synchronize(local_rank)

        expected = world_size * (world_size + 1) / 2
        actual = value.item()
        if actual != expected:
            raise RuntimeError(
                f"RCCL all-reduce mismatch on rank {rank}: expected {expected}, got {actual}"
            )

        dist.barrier()
        if rank == 0:
            print(
                f"BW1000 distributed preflight passed on {world_size} devices",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
