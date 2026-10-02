"""tandem's DDP against torch's DistributedDataParallel (gloo backend) and
against one process training on the whole batch.

With 2 ranks, averaging gradients is (a / 2) + (b / 2) whatever the
algorithm, which floating point computes exactly the same way in any order,
so tandem must match torch DDP bit for bit: every loss and every weight.
Against a single process, the batch is summed in a different order inside
the kernels, so the match is to rounding."""

import contextlib
import os
import tempfile

import torch
import torch.multiprocessing as mp

from tandem import launch
from tandem.data import synthetic
from tandem.ddp import DDP
from tandem.model import GPT, Config

CFG = Config(seq=32, layers=2, heads=2, dim=32)
STEPS, BATCH = 6, 8


def train(model_fn, rank, world, accumulate=1):
    torch.manual_seed(0)
    text = synthetic(20_000)
    model, step_fn = model_fn()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1, foreach=False)
    losses = []
    for step in range(STEPS):
        for micro in range(accumulate):
            x, y = text.batch(step * accumulate + micro, BATCH, CFG.seq, rank=rank, world=world)
            loss = step_fn(x, y, last=micro == accumulate - 1) / accumulate
            losses.append(loss.item())
        opt.step()
        opt.zero_grad()
    return losses, torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def tandem_rank(g, bucket_bytes, accumulate):
    def make():
        model = GPT(CFG)
        ddp = DDP(model, g, bucket_bytes=bucket_bytes)

        def step(x, y, last):
            with contextlib.nullcontext() if last else ddp.no_sync():
                loss = ddp(x, y)
                (loss / accumulate).backward()
            return loss

        return model, step

    return train(make, g.rank, g.size, accumulate)


def torch_rank(rank, world, path, out, accumulate):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=world)

    def make():
        model = GPT(CFG)
        ddp = DistributedDataParallel(model)

        def step(x, y, last):
            ctx = ddp.no_sync() if not last else torch.enable_grad()
            with ctx:
                loss = ddp(x, y)
                (loss / accumulate).backward()
            return loss

        return model, step

    res = train(make, rank, world, accumulate)
    torch.save(res, f"{out}.{rank}")
    dist.destroy_process_group()


def run_torch(world, accumulate=1):
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(torch_rank, args=(world, os.path.join(d, "init"), os.path.join(d, "out"), accumulate), nprocs=world)
        return [torch.load(os.path.join(d, f"out.{r}")) for r in range(world)]


def single(accumulate=1):
    torch.set_num_threads(1)

    def make():
        model = GPT(CFG)

        def step(x, y, last):
            loss = model(x, y)
            (loss / accumulate).backward()
            return loss

        return model, step

    return train(make, 0, 1, accumulate)


def test_bit_identical_to_torch_ddp():
    # Small buckets: several all-reduces in flight during each backward.
    ours = launch(tandem_rank, 2, 64 << 10, 1)
    ref = run_torch(2)
    for r in range(2):
        assert ours[r][0] == ref[r][0], f"rank {r} losses differ"
        assert torch.equal(ours[r][1], ref[r][1]), f"rank {r} weights differ"
    assert torch.equal(ours[0][1], ours[1][1])  # ranks stay in step


def test_gradient_accumulation_bit_identical():
    ours = launch(tandem_rank, 2, 1 << 20, 2)
    ref = run_torch(2, accumulate=2)
    assert ours[0][0] == ref[0][0]
    assert torch.equal(ours[0][1], ref[0][1])


def test_matches_single_process():
    ours = launch(tandem_rank, 2, 1 << 20, 1)
    _, w = single()
    assert torch.allclose(ours[0][1], w, rtol=0, atol=1e-5), (ours[0][1] - w).abs().max()
