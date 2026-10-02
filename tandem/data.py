"""Byte-level language-model batches from a text file (vocabulary of 256, no
tokenizer). Every rank draws the same global batch from a seeded generator
and keeps its own rows, so a run on N ranks sees exactly the data a single
process sees with the whole batch."""

import torch


class Text:
    def __init__(self, path=None, data=None):
        if data is None:
            with open(path, "rb") as f:
                data = f.read()
        self.data = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()

    def batch(self, step, batch, seq, seed=0, rank=0, world=1, device="cpu"):
        """Global batch `step`: `batch` rows of `seq + 1` bytes; this rank's
        rows are [rank * batch / world, (rank + 1) * batch / world)."""
        if batch % world:
            raise ValueError(f"global batch {batch} does not divide over {world} ranks")
        g = torch.Generator().manual_seed(seed * 1_000_003 + step)
        starts = torch.randint(0, self.data.numel() - seq - 1, (batch,), generator=g)
        per = batch // world
        starts = starts[rank * per:(rank + 1) * per]
        rows = torch.stack([self.data[s:s + seq + 1] for s in starts.tolist()])
        x, y = rows[:, :-1], rows[:, 1:]
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def synthetic(n=200_000, seed=0):
    """Random-but-structured bytes for tests: a seeded Markov chain, so a
    model can learn something without a corpus on disk."""
    g = torch.Generator().manual_seed(seed)
    trans = torch.randint(0, 256, (256, 4), generator=g)
    choice = torch.randint(0, 4, (n,), generator=g)
    out = bytearray(n)
    s = 0
    for i in range(n):
        s = int(trans[s, choice[i]])
        out[i] = s
    return Text(data=bytes(out))
