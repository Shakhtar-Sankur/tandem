"""ZeRO: data parallelism that shards what every rank would otherwise hold a
full copy of (Rajbhandari et al., 2020).

  stage 1  optimizer state sharded: gradients are all-reduced as in DDP,
           each rank updates only its shard of the parameters, then the
           ranks all-gather the updated parameters.
  stage 2  gradients sharded too: each bucket of gradients is
           reduce-scattered, so a rank keeps only the sums for its shard,
           and the full-size gradients are freed as soon as they are sent.
  stage 3  parameters sharded too (FSDP): the model is a sequence of units
           (its blocks); a unit's full parameters exist only while it runs.
           They are all-gathered before its forward, freed after, gathered
           again before its backward, and its gradients are
           reduce-scattered as soon as they are complete. The next unit's
           gather is started early, so it travels while this one computes.

Stages 1 and 2 keep all parameters as views into one flat buffer, divided
into one contiguous shard per rank; stage 3 does the same per unit. The
optimizer (ShardedAdamW) only ever sees a rank's shards."""

import warnings

import torch
from torch.autograd import Variable

from .optim import ShardedAdamW


def _split_sizes(n, world):
    return [len(c) for c in torch.tensor_split(torch.empty(n, device="meta"), world)]


class _Flat:
    """Parameters as views into one flat buffer, divided into shards."""

    def __init__(self, params, world, rank):
        self.params = params
        self.numel = sum(p.numel() for p in params)
        self.sizes = _split_sizes(self.numel, world)
        self.lo = sum(self.sizes[:rank])
        self.hi = self.lo + self.sizes[rank]
        self.offset = {}
        off = 0
        for p in params:
            self.offset[p] = off
            off += p.numel()

    def bucket_sizes(self, a, b):
        """How much of the flat range [a, b) falls in each rank's shard."""
        out, lo = [], 0
        for s in self.sizes:
            out.append(max(0, min(b, lo + s) - max(a, lo)))
            lo += s
        return out


class _Bucket:
    def __init__(self, params, a, b):
        self.params, self.a, self.b = params, a, b
        self.ready = 0
        self.work = None
        self.staging = None


class ZeRO(torch.nn.Module):
    """Stages 1 and 2. Train with: loss = z(x, y); loss.backward(); z.step();
    z.zero_grad()."""

    def __init__(self, module, group, stage=1, optimizer=None, bucket_bytes=25 << 20):
        super().__init__()
        if stage not in (1, 2):
            raise ValueError("ZeRO handles stages 1 and 2; stage 3 is FSDP")
        self.module, self.g, self.stage = module, group, stage
        self.opt = optimizer or ShardedAdamW()
        params = [p for p in module.parameters() if p.requires_grad]
        dev, dtype = params[0].device, params[0].dtype
        self.flat = _Flat(params, group.size, group.rank)
        buf = torch.empty(self.flat.numel, device=dev, dtype=dtype)
        with torch.no_grad():
            for p in params:
                buf[self.flat.offset[p]:self.flat.offset[p] + p.numel()].copy_(p.reshape(-1))
            self.g.broadcast(buf, root=0)
            for p in params:
                p.data = buf[self.flat.offset[p]:self.flat.offset[p] + p.numel()].view_as(p)
        self.buf = buf
        if stage == 1:
            self.grad = torch.zeros_like(buf)
        else:
            self.grad_shard = torch.zeros(self.flat.hi - self.flat.lo, device=dev, dtype=dtype)
        # Buckets: contiguous ranges of the flat buffer, from its end, the
        # order backward produces gradients in.
        self.buckets, self.bucket_of = [], {}
        cur, size = [], 0
        for p in reversed(params):
            nb = p.numel() * p.element_size()
            if cur and size + nb > bucket_bytes:
                self._add(cur)
                cur, size = [], 0
            cur.append(p)
            size += nb
        if cur:
            self._add(cur)
        self._next = 0
        self._queued = False
        for p in params:
            p.register_post_accumulate_grad_hook(self._hook)

    def _add(self, params):
        a = min(self.flat.offset[p] for p in params)
        b = max(self.flat.offset[p] + p.numel() for p in params)
        bk = _Bucket(params, a, b)
        for p in params:
            self.bucket_of[p] = bk
        self.buckets.append(bk)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _hook(self, p):
        bk = self.bucket_of[p]
        off = self.flat.offset[p]
        if self.stage == 1:
            view = self.grad[off:off + p.numel()].view_as(p)
            if p.grad is not view:
                view.copy_(p.grad)
                p.grad = view
        else:
            if bk.staging is None:
                bk.staging = torch.empty(bk.b - bk.a, device=p.device, dtype=p.dtype)
            bk.staging[off - bk.a:off - bk.a + p.numel()].copy_(p.grad.reshape(-1))
            p.grad = None  # the full-size gradient is not kept
        if not self._queued:
            self._queued = True
            Variable._execution_engine.queue_callback(self._finish_backward)
        bk.ready += 1
        while self._next < len(self.buckets) and self.buckets[self._next].ready == len(self.buckets[self._next].params):
            nb = self.buckets[self._next]
            if self.stage == 1:
                nb.work = self.g.all_reduce(self.grad[nb.a:nb.b], op="avg", async_op=True)
            else:
                nb.work = self.g.reduce_scatter(nb.staging, self.flat.bucket_sizes(nb.a, nb.b), op="avg", async_op=True)
            self._next += 1

    def _finish_backward(self):
        self._queued = False
        if any(bk.ready != len(bk.params) for bk in self.buckets):
            raise RuntimeError("tandem ZeRO: every parameter must receive a gradient")
        for bk in self.buckets:
            mine = bk.work.wait()
            if self.stage == 2:
                lo = max(bk.a, self.flat.lo)
                if mine.numel():
                    self.grad_shard[lo - self.flat.lo:lo - self.flat.lo + mine.numel()].add_(mine)
                bk.staging = None
            bk.ready, bk.work = 0, None
        self._next = 0

    @torch.no_grad()
    def step(self):
        lo, hi = self.flat.lo, self.flat.hi
        grad = self.grad[lo:hi] if self.stage == 1 else self.grad_shard
        self.opt.update("flat", self.buf[lo:hi], grad)
        self.g.all_gather(self.buf, self.flat.sizes)

    def zero_grad(self):
        if self.stage == 1:
            for p in self.flat.params:
                p.grad = None
        else:
            self.grad_shard.zero_()


class _Unit:
    def __init__(self, index, module, params, world, rank, device, dtype):
        self.index, self.module = index, module
        self.flat = _Flat(params, world, rank)
        self.full = torch.empty(self.flat.numel, device=device, dtype=dtype)
        self.nbytes = self.full.untyped_storage().nbytes()
        self.shard = None
        self.grad_shard = torch.zeros(self.flat.hi - self.flat.lo, device=device, dtype=dtype)
        self.gathered = True
        self.pending = None  # all-gather in flight
        self.ready = 0
        self.rs = None  # reduce-scatter in flight
        self.staging = None


class FSDP(torch.nn.Module):
    """ZeRO stage 3 over model.units() (for tandem's GPT: the embedding, each
    block, the head). Train as with ZeRO: loss = f(x, y); loss.backward();
    f.step(); f.zero_grad()."""

    def __init__(self, model, group, optimizer=None, prefetch=True):
        super().__init__()
        self.module, self.g = model, group
        self.opt = optimizer or ShardedAdamW()
        self.prefetch = prefetch
        # The embedding's input (token ids) needs no gradient; torch warns
        # that its backward hook then fires on the outputs, which is the
        # moment we want.
        warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed with respect to module outputs")
        self.live = 0  # units whose full parameters exist right now
        self.max_live = 0
        mods = model.units()
        seen = set()
        self.units, self.unit_of = [], {}
        for i, m in enumerate(mods):
            ps = [p for p in m.parameters() if p.requires_grad and id(p) not in seen]
            seen.update(id(p) for p in ps)
            u = _Unit(i, m, ps, group.size, group.rank, ps[0].device, ps[0].dtype)
            with torch.no_grad():
                for p in ps:
                    o = u.flat.offset[p]
                    u.full[o:o + p.numel()].copy_(p.reshape(-1))
                self.g.broadcast(u.full, root=0)
                for p in ps:
                    o = u.flat.offset[p]
                    p.data = u.full[o:o + p.numel()].view_as(p)
                u.shard = u.full[u.flat.lo:u.flat.hi].clone()
            for p in ps:
                self.unit_of[p] = u
                p.register_post_accumulate_grad_hook(self._grad_hook)
            m.register_forward_pre_hook(self._pre_forward(u))
            m.register_forward_hook(self._post_forward(u))
            m.register_full_backward_pre_hook(self._pre_backward(u))
            self.live += 1  # allocated whole by _Unit
            self._free(u)
            self.units.append(u)
        if len(seen) != sum(1 for p in model.parameters() if p.requires_grad):
            raise ValueError("every parameter must belong to one of model.units()")
        self._queued = False
        self.max_live = 0

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # ---- full parameters of a unit: gather, wait, free
    def _start_gather(self, u):
        if u.gathered or u.pending is not None:
            return
        u.full.untyped_storage().resize_(u.nbytes)
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        u.full[u.flat.lo:u.flat.hi].copy_(u.shard)
        u.pending = self.g.all_gather(u.full, u.flat.sizes, async_op=True)

    def _gather(self, u):
        self._start_gather(u)
        if u.pending is not None:
            u.pending.wait()
            u.pending = None
            u.gathered = True

    def _free(self, u):
        if u.pending is not None:
            u.pending.wait()
            u.pending = None
        if u.full.untyped_storage().nbytes():
            self.live -= 1
        u.full.untyped_storage().resize_(0)
        u.gathered = False

    # ---- hooks
    def _pre_forward(self, u):
        def hook(mod, args):
            self._gather(u)
            if self.prefetch and u.index + 1 < len(self.units):
                self._start_gather(self.units[u.index + 1])

        return hook

    def _post_forward(self, u):
        def hook(mod, args, out):
            # The last unit's backward comes first: keep it.
            if torch.is_grad_enabled() and u.index == len(self.units) - 1:
                return
            self._free(u)

        return hook

    def _pre_backward(self, u):
        def hook(mod, grad_out):
            if not self._queued:
                self._queued = True
                Variable._execution_engine.queue_callback(self._finish_backward)
            self._gather(u)
            if self.prefetch and u.index > 0:
                self._start_gather(self.units[u.index - 1])

        return hook

    def _grad_hook(self, p):
        u = self.unit_of[p]
        if u.staging is None:
            u.staging = torch.empty(u.flat.numel, device=p.device, dtype=p.dtype)
        o = u.flat.offset[p]
        u.staging[o:o + p.numel()].copy_(p.grad.reshape(-1))
        p.grad = None
        u.ready += 1
        if u.ready == len(u.flat.params):
            # The unit's backward is done: drop its full parameters and send
            # its gradients on their way.
            self._free(u)
            u.rs = self.g.reduce_scatter(u.staging, u.flat.sizes, op="avg", async_op=True)

    def _finish_backward(self):
        self._queued = False
        for u in self.units:
            if u.ready != len(u.flat.params):
                raise RuntimeError("tandem FSDP: every parameter must receive a gradient")
            mine = u.rs.wait()
            if mine.numel():
                u.grad_shard.add_(mine)
            u.rs, u.staging, u.ready = None, None, 0
            if u.gathered:
                self._free(u)

    @torch.no_grad()
    def step(self):
        for u in self.units:
            self.opt.update(u.index, u.shard, u.grad_shard)

    def zero_grad(self):
        for u in self.units:
            u.grad_shard.zero_()

    @torch.no_grad()
    def full_parameters(self):
        """All parameters, gathered (for checkpoints and tests)."""
        out = []
        for u in self.units:
            full = torch.empty(u.flat.numel, device=u.shard.device, dtype=u.shard.dtype)
            full[u.flat.lo:u.flat.hi].copy_(u.shard)
            self.g.all_gather(full, u.flat.sizes)
            out.append(full)
        return torch.cat(out)
