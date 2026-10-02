"""Pipeline parallelism against one process accumulating gradients over the
same micro-batches. Each stage computes, per micro-batch, exactly what that
process computes for its units, and runs the backwards in the same order,
so gradients, losses and weights after AdamW must match bit for bit, for
both schedules and any number of stages (including stages that hold one
unit and stages whose warm-up runs ahead by several micro-batches)."""

import pytest
import torch

from tandem import launch
from tandem.data import synthetic
from tandem.model import GPT, loss_fn
from tandem.pipeline import Pipeline, split_units
from tandem.profile import Timeline

from test_ddp import CFG

STEPS, BATCH, MICRO = 3, 12, 6


def stage_rank(g, schedule):
    torch.manual_seed(0)
    text = synthetic(20_000)
    model = GPT(CFG)
    tl = Timeline()
    pipe = Pipeline(model, g, MICRO, schedule, timeline=tl)
    opt = torch.optim.AdamW(pipe.parameters(), lr=3e-3, weight_decay=0.1, foreach=False)
    losses = []
    for step in range(STEPS):
        x, y = text.batch(step, BATCH, CFG.seq)
        loss = pipe.train_step(x, y)
        losses.append(None if loss is None else loss.item())
        opt.step()
        opt.zero_grad()
    names = [n for n, _, _ in tl.spans[: 2 * MICRO]]
    return losses, torch.cat([p.detach().reshape(-1) for p in pipe.parameters()]), names


def single():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    text = synthetic(20_000)
    model = GPT(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1, foreach=False)
    losses = []
    for step in range(STEPS):
        x, y = text.batch(step, BATCH, CFG.seq)
        total = 0
        for xb, yb in zip(x.chunk(MICRO), y.chunk(MICRO)):
            loss = loss_fn(model(xb), yb) / MICRO
            loss.backward()
            total = total + loss.detach()
        losses.append(total.item())
        opt.step()
        opt.zero_grad()
    units = model.units()
    return losses, [torch.cat([p.detach().reshape(-1) for p in u.parameters()]) for u in units]


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
@pytest.mark.parametrize("stages", [2, 3, 4])
def test_bit_identical_to_accumulation(schedule, stages):
    ours = launch(stage_rank, stages, schedule, slot_bytes=1024)
    ref_losses, ref_units = single()
    n_units = len(ref_units)
    for r, (lo, hi) in enumerate(split_units(n_units, stages)):
        assert torch.equal(ours[r][1], torch.cat(ref_units[lo:hi])), f"stage {r} weights differ"
    assert ours[-1][0] == ref_losses
    # The schedules really differ: 1F1B interleaves backwards early on stage 0.
    first = ours[0][2]
    if schedule == "gpipe":
        assert first == [f"F{i}" for i in range(MICRO)] + [f"B{i}" for i in range(MICRO)]
    else:
        warm = stages - 1
        assert first[: warm + 2] == [f"F{i}" for i in range(warm + 1)] + ["B0"]
