"""The collectives against values computed locally, on 2, 3 and 4 CPU
processes. Inputs are integer-valued, so sums are exact and every comparison
is bit-for-bit; slots are 64 bytes, so every message travels in many pieces."""

import pytest
import torch

from tandem import launch

SIZES = [0, 1, 5, 64, 1000, 4099]


def data(rank, n, dtype, seed=0):
    g = torch.Generator().manual_seed(seed * 1000 + rank)
    hi = 30 if dtype == torch.bfloat16 else 100  # partial sums stay exact in bf16
    return torch.randint(-hi, hi, (n,), generator=g).to(dtype)


def collectives(g):
    r, w = g.rank, g.size
    for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.int64):
        for n in SIZES:
            want = sum(data(q, n, dtype).double() for q in range(w))
            x = data(r, n, dtype)
            g.all_reduce(x)
            assert torch.equal(x.double(), want), ("all_reduce", dtype, n)

            # Uneven chunks, one of them empty.
            sizes = [n // w] * w
            sizes[-1] += n - sum(sizes)
            sizes = sizes[1:] + sizes[:1]
            x = data(r, n, dtype)
            mine = g.reduce_scatter(x, sizes)
            lo = sum(sizes[:r])
            assert torch.equal(mine.double(), want[lo:lo + sizes[r]]), ("reduce_scatter", dtype, n)

            full = want.to(dtype)
            x = torch.full((n,), -7, dtype=dtype)
            x[lo:lo + sizes[r]] = full[lo:lo + sizes[r]]
            g.all_gather(x, sizes)
            assert torch.equal(x, full), ("all_gather", dtype, n)

            x = data(r, n, dtype, seed=1)
            g.broadcast(x, root=w - 1)
            assert torch.equal(x, data(w - 1, n, dtype, seed=1)), ("broadcast", dtype, n)

    x = data(r, 37, torch.float32)
    g.all_reduce(x, op="avg")
    want = sum(data(q, 37, torch.float32).double() for q in range(w)) / w
    assert torch.allclose(x.double(), want, rtol=0, atol=1e-5)
    return r


@pytest.mark.parametrize("world", [2, 3, 4])
def test_collectives(world):
    assert launch(collectives, world, slot_bytes=64) == list(range(world))


def async_order(g):
    # Several collectives in flight at once complete in submission order with
    # the right values, while this thread keeps working.
    xs = [data(g.rank, 3000 + k, torch.float32, seed=k) for k in range(5)]
    works = [g.all_reduce(x, async_op=True) for x in xs]
    busy = torch.randn(200, 200) @ torch.randn(200, 200)
    for k, (x, wk) in enumerate(zip(xs, works)):
        wk.wait()
        want = sum(data(q, 3000 + k, torch.float32, seed=k) for q in range(g.size))
        assert torch.equal(x, want)
    return float(busy.sum())


def test_async():
    assert len(launch(async_order, 3, slot_bytes=256)) == 3


def ping_pong(g):
    other = 1 - g.rank
    x = torch.arange(10_000, dtype=torch.float32) * (g.rank + 1)
    y = torch.empty_like(x)
    for _ in range(3):
        if g.rank == 0:
            g.send(x, other)
            g.recv(y, other)
        else:
            g.recv(y, other)
            g.send(x, other)
        assert torch.equal(y, torch.arange(10_000, dtype=torch.float32) * (other + 1))
    return True


def test_send_recv():
    assert launch(ping_pong, 2, slot_bytes=512) == [True, True]


def failing(g):
    if g.rank == 1:
        raise ValueError("boom")
    return True


def test_error_propagates():
    with pytest.raises(Exception, match="boom"):
        launch(failing, 2)


def test_exposed_counts_only_running_communication():
    from tandem.profile import summarize

    # Waits 0-10 and 20-30; collectives run 5-12 and 25-40.
    s = summarize([], [("a", 5, 12, 0), ("b", 25, 40, 0)], [("a", 0, 10), ("b", 20, 30)], 40)
    assert s["comm_s"] == 22 and s["exposed_comm_s"] == 10
    assert abs(s["overlap"] - (1 - 10 / 22)) < 1e-12
