"""The Python engine against the C++ engine on ONE GPU (e.g. Colab's T4),
both ranks sharing it through CUDA IPC. NCCL cannot run two ranks on one GPU,
so this compares the two tandem engines only; bench/run.py engine compares
with NCCL on two GPUs.

usage: python bench/engine_one_gpu.py
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from tandem import launch

SIZES_KB = [64, 256, 1024, 4096, 16384, 65536]


def rank(g, reps):
    dev = g.device
    out = {}
    for kb in SIZES_KB:
        x = torch.ones(kb * 1024 // 4, device=dev)
        g.all_reduce(x)
        torch.cuda.synchronize(dev)
        ts = []
        for _ in range(reps):
            g.barrier()
            torch.cuda.synchronize(dev)
            t = time.perf_counter()
            g.all_reduce(x)
            torch.cuda.synchronize(dev)
            ts.append(time.perf_counter() - t)
        out[kb] = statistics.median(ts)
    return out


def main():
    if not torch.cuda.is_available():
        sys.exit("needs a GPU (Colab: Runtime -> Change runtime type -> T4 GPU)")
    res = {}
    for backend in ("python", "cpp"):
        res[backend] = launch(rank, 2, 20, device="cuda", backend=backend, gpus=[0, 0])[0]
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}))
    for kb in SIZES_KB:
        py, cpp = res["python"][kb], res["cpp"][kb]
        print(json.dumps({"size_KB": kb, "python_ms": round(py * 1e3, 3), "cpp_ms": round(cpp * 1e3, 3),
                          "speedup": round(py / cpp, 2)}))


if __name__ == "__main__":
    main()
