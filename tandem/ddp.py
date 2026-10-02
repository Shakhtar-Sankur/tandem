"""Data parallelism: every rank holds the whole model, trains on its share of
the batch, and averages gradients with the others before each step.

Gradients are reduced in buckets of about `bucket_bytes`, in the order they
are produced, and each bucket's all-reduce starts as soon as its last
gradient is ready, on the communication thread, while backward goes on
computing earlier layers. At the end of backward the reducer waits for the
buckets still in flight. After the first step, buckets are rebuilt in the
order gradients actually arrived, so the first bucket to fill is the first
to be sent. This is the design of torch's DistributedDataParallel; the
reduction itself is tandem's own ring all-reduce.

Each parameter's gradient is a view into its bucket's flat buffer, so the
all-reduce happens in place, with no copy back."""

import contextlib

import torch
from torch.autograd import Variable


class Bucket:
    def __init__(self, params, device, dtype):
        self.params = params
        self.flat = torch.zeros(sum(p.numel() for p in params), device=device, dtype=dtype)
        self.views = {}
        off = 0
        for p in params:
            self.views[p] = self.flat[off:off + p.numel()].view_as(p)
            off += p.numel()
        self.ready = 0
        self.work = None


class DDP(torch.nn.Module):
    def __init__(self, module, group, bucket_bytes=25 << 20, overlap=True, rebuild=True):
        super().__init__()
        self.module = module
        self.g = group
        self.bucket_bytes = bucket_bytes
        self.overlap = overlap
        self._rebuild = rebuild
        self.params = [p for p in module.parameters() if p.requires_grad]
        with torch.no_grad():  # every rank starts from rank 0's weights
            for p in self.params:
                self.g.broadcast(p.data, root=0)
        # Until the first backward shows the real order, assume gradients
        # arrive in reverse registration order, as they mostly do.
        self._make_buckets(list(reversed(self.params)))
        self._sync = True
        self._order = []
        self._queued = False
        for p in self.params:
            p.register_post_accumulate_grad_hook(self._hook)

    def _make_buckets(self, order):
        self.buckets, self.bucket_of = [], {}
        cur, size = [], 0
        for p in order:
            nb = p.numel() * p.element_size()
            if cur and (size + nb > self.bucket_bytes or p.dtype != cur[0].dtype):
                self._add_bucket(cur)
                cur, size = [], 0
            cur.append(p)
            size += nb
        if cur:
            self._add_bucket(cur)
        self._next = 0

    def _add_bucket(self, params):
        b = Bucket(params, params[0].device, params[0].dtype)
        for p in params:
            self.bucket_of[p] = b
        self.buckets.append(b)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextlib.contextmanager
    def no_sync(self):
        """Gradient accumulation: backward passes inside only accumulate
        locally; the next one outside reduces the sum."""
        old, self._sync = self._sync, False
        try:
            yield
        finally:
            self._sync = old

    def _hook(self, p):
        b = self.bucket_of[p]
        view = b.views[p]
        if p.grad is not view:  # first gradient since zero_grad: move it into the bucket
            view.copy_(p.grad)
            p.grad = view
        if not self._sync:
            return
        if not self._queued:
            self._queued = True
            Variable._execution_engine.queue_callback(self._finish_backward)
        if self._rebuild:
            self._order.append(p)
        b.ready += 1
        # Launch in bucket order, so every rank submits the same sequence.
        while self._next < len(self.buckets) and self.buckets[self._next].ready == len(self.buckets[self._next].params):
            nb = self.buckets[self._next]
            nb.work = self.g.all_reduce(nb.flat, op="avg", async_op=True)
            if not self.overlap:
                nb.work.wait()
            self._next += 1

    def _finish_backward(self):
        self._queued = False
        missing = [b for b in self.buckets if b.ready != len(b.params)]
        if missing:
            unused = [p.shape for b in missing for p in b.params if p.grad is None]
            raise RuntimeError(f"tandem DDP: some parameters received no gradient ({unused[:3]}...); every parameter must be used")
        for b in self.buckets:
            b.work.wait()
            b.ready, b.work = 0, None
        self._next = 0
        if self._rebuild and len(self._order) == len(self.params):
            order, self._order, self._rebuild = self._order, [], False
            grads = {p: p.grad.detach().clone() for p in self.params}
            self._make_buckets(order)
            for p in self.params:
                self.bucket_of[p].views[p].copy_(grads[p])
                p.grad = self.bucket_of[p].views[p]
        else:
            self._order = []
