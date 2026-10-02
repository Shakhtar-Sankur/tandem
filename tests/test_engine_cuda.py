"""The C++ engine on GPUs (milestone M3). Skipped without a GPU. With one GPU
both ranks share it (CUDA IPC within one device); with two or more, each rank
has its own (IPC across devices, peer-to-peer when the GPUs allow it).

    PYTHONPATH=.:tests python -m pytest -q tests/test_engine_cuda.py
"""

import os

import pytest
import torch

from tandem import launch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def gpus(world):
    n = torch.cuda.device_count()
    return [r % n for r in range(world)]


def values(rank, n, dtype, device):
    g = torch.Generator().manual_seed(1234 + rank)
    if dtype.is_floating_point:
        t = torch.randn(n, generator=g)
    else:
        t = torch.randint(-1000, 1000, (n,), generator=g)
    return t.to(dtype).to(device)


def collectives(g, dtype):
    dev = g.device
    n, w, r = 10_007, g.size, g.rank  # not a multiple of anything: uneven chunks and a short last piece
    mine = [values(q, n, dtype, dev) for q in range(w)]
    out = {}

    # The reference sums in rank order, as the ring does for 2 ranks; for more
    # ranks the ring's order differs, so floating point compares with a tolerance.
    expect = mine[0].clone()
    for q in range(1, w):
        expect = expect + mine[q]

    t = mine[r].clone()
    g.all_reduce(t)
    out["all_reduce"] = (t.cpu(), expect.cpu())

    if dtype.is_floating_point:  # integer tensors cannot be divided in place
        t = mine[r].clone()
        work = g.all_reduce(t, op="avg", async_op=True)
        busy = torch.randn(512, 512, device=dev)
        for _ in range(4):
            busy = busy @ busy.T / 512  # the GPU works on something else meanwhile
        work.wait()
        avg = mine[0] / w
        for q in range(1, w):
            avg = avg + mine[q] / w
        out["all_reduce_avg"] = (t.cpu(), avg.cpu())

    sizes = [n // w + (1 if q < n % w else 0) for q in range(w)]
    t = mine[r].clone()
    chunk = g.reduce_scatter(t, sizes=sizes)
    lo = sum(sizes[:r])
    out["reduce_scatter"] = (chunk.cpu(), expect[lo:lo + sizes[r]].cpu())

    t = torch.zeros(n, dtype=dtype, device=dev)
    t[lo:lo + sizes[r]] = mine[r][lo:lo + sizes[r]]
    g.all_gather(t, sizes=sizes)
    gathered = torch.cat([mine[q][sum(sizes[:q]):sum(sizes[:q]) + sizes[q]] for q in range(w)])
    out["all_gather"] = (t.cpu(), gathered.cpu())

    t = mine[r].clone()
    g.broadcast(t, root=w - 1)
    out["broadcast"] = (t.cpu(), mine[w - 1].cpu())

    if w == 2:
        got = torch.empty(n, dtype=dtype, device=dev)
        g.sendrecv(mine[r], 1 - r, got, 1 - r)
        out["sendrecv"] = (got.cpu(), mine[1 - r].cpu())
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("world", [2, 3])
def test_collectives_on_gpu(dtype, world):
    # 4 KB slots: every collective runs through many pieces and reuses each slot.
    res = launch(collectives, world, dtype, device="cuda", backend="cpp", slot_bytes=4096, gpus=gpus(world))
    for rank, out in enumerate(res):
        for name, (got, want) in out.items():
            exact = world == 2 or not dtype.is_floating_point or name in ("all_gather", "broadcast", "sendrecv")
            if exact:
                assert torch.equal(got, want), f"rank {rank} {name} {dtype}"
            else:
                tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
                torch.testing.assert_close(got, want, rtol=tol, atol=tol, msg=f"rank {rank} {name} {dtype}")


def ddp_steps(g, backend_name):
    from test_ddp import CFG

    from tandem.data import synthetic
    from tandem.ddp import DDP
    from tandem.model import GPT

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.manual_seed(0)
    text = synthetic(20_000)
    model = GPT(CFG).to(g.device)
    ddp = DDP(model, g)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, foreach=False)
    for step in range(4):
        x, y = text.batch(step, 8, CFG.seq, rank=g.rank, world=g.size, device=g.device)
        ddp(x, y).backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


def test_ddp_on_gpu_matches_the_python_engine():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # read by cuBLAS in each rank
    py = launch(ddp_steps, 2, "python", device="cuda", backend="python", gpus=gpus(2))
    cpp = launch(ddp_steps, 2, "cpp", device="cuda", backend="cpp", gpus=gpus(2))
    assert torch.equal(py[0], cpp[0]) and torch.equal(py[1], cpp[1])
