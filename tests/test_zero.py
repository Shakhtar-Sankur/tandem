"""ZeRO stages 1, 2 and 3 against torch DDP with torch's AdamW (gloo, 2
ranks). Each stage changes only where gradients, optimizer state and
parameters live, never the arithmetic: the all-gathers copy exactly, the
averaging is (a / 2) + (b / 2), and the sharded AdamW does torch's
operations element by element. So all three must match torch DDP bit for
bit, and each rank holds only its share of the optimizer state."""

import torch

from tandem import launch
from tandem.data import synthetic
from tandem.model import GPT
from tandem.optim import ShardedAdamW
from tandem.zero import FSDP, ZeRO

from test_ddp import BATCH, CFG, STEPS, run_torch


def zero_rank(g, stage, bucket_bytes):
    torch.manual_seed(0)
    text = synthetic(20_000)
    model = GPT(CFG)
    opt = ShardedAdamW(lr=3e-3, weight_decay=0.1)
    z = ZeRO(model, g, stage, opt, bucket_bytes) if stage < 3 else FSDP(model, g, opt)
    losses = []
    for step in range(STEPS):
        x, y = text.batch(step, BATCH, CFG.seq, rank=g.rank, world=g.size)
        loss = z(x, y)
        loss.backward()
        losses.append(loss.item())
        z.step()
        z.zero_grad()
    if stage < 3:
        w = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        if stage == 2:
            assert all(p.grad is None for p in model.parameters())  # only shard gradients kept
    else:
        # At most the running unit, the one being prefetched and the head
        # (kept from forward to backward) are ever whole at once.
        assert z.max_live <= 3, z.max_live
        assert z.live == 0
        w = z.full_parameters()
    return losses, w, opt.state_bytes()


def check(stage, bucket_bytes=4 << 10):
    ours = launch(zero_rank, 2, stage, bucket_bytes)
    ref = run_torch(2)
    nparams = ref[0][1].numel()
    for r in range(2):
        assert ours[r][0] == ref[r][0], f"stage {stage}, rank {r}: losses differ"
        assert torch.equal(ours[r][1], ref[r][1]), f"stage {stage}, rank {r}: weights differ"
        # Adam's two moments for half the parameters, not all of them.
        assert ours[r][2] <= 2 * 4 * (nparams // 2 + 1)


def test_stage1_bit_identical():
    check(1)


def test_stage2_bit_identical():
    check(2)


def test_stage3_bit_identical():
    check(3)
