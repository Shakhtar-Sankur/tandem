"""Pipeline parallelism: the model's units are split into one contiguous
stage per rank, and each batch is cut into micro-batches that flow through
the stages, activations forward and their gradients back.

  gpipe  every micro-batch forward, then every micro-batch backward
         (Huang et al., 2019). Simple; holds all M micro-batches'
         activations at once.
  1f1b   after a warm-up of (stages - rank - 1) forwards, each stage
         alternates one forward with one backward, then drains (as in
         PipeDream-Flush and Megatron-LM). The same idle "bubble", but a
         stage holds at most (stages - rank) micro-batches' activations.

Neighbouring stages swap tensors with paired transfers (send one while
receiving the other), in the order Megatron-LM pairs them, so no two
stages ever wait on each other. Every stage runs each micro-batch's
backward in micro-batch order, so its gradients accumulate exactly as one
process accumulating over the same micro-batches would."""

import torch

from .model import loss_fn
from .profile import Timeline


def split_units(n_units, stages):
    """Contiguous ranges of units, one per stage, as even as possible."""
    bounds = [len(c) for c in torch.tensor_split(torch.arange(n_units), stages)]
    out, lo = [], 0
    for b in bounds:
        out.append((lo, lo + b))
        lo += b
    return out


class Pipeline:
    def __init__(self, model, group, micro_batches, schedule="1f1b", timeline=None):
        if schedule not in ("gpipe", "1f1b"):
            raise ValueError(f"unknown schedule {schedule}")
        self.g, self.m, self.schedule = group, micro_batches, schedule
        units = model.units()
        lo, hi = split_units(len(units), group.size)[group.rank]
        self.units = torch.nn.ModuleList(units[lo:hi])
        self.first, self.last = group.rank == 0, group.rank == group.size - 1
        self.dim = model.config.dim
        self.timeline = timeline or Timeline(enabled=False)

    def parameters(self):
        return self.units.parameters()

    def _forward(self, inp):
        x = inp
        for u in self.units:
            x = u(x)
        return x

    def train_step(self, x, y):
        """One step's forward and backward over the global batch (x, y),
        which every rank passes (only stage 0 reads x, only the last stage
        reads y). Gradients accumulate in this stage's parameters; returns
        the mean loss on the last stage, None elsewhere."""
        mbs = list(zip(x.chunk(self.m), y.chunk(self.m)))
        if any(xb.shape[0] != mbs[0][0].shape[0] for xb, _ in mbs):
            raise ValueError("the batch must divide evenly into micro-batches")
        b, t = mbs[0][0].shape
        dev = next(self.parameters()).device
        self._act_shape = (b, t, self.dim)
        self._dev = dev
        self._saved = {}
        self._losses = []
        if self.schedule == "gpipe":
            self._gpipe(mbs)
        else:
            self._one_f_one_b(mbs)
        if self.last:
            return sum(self._losses)
        return None

    # ---- one micro-batch, one direction
    def _fwd(self, i, mbs, inp):
        with self.timeline.span(f"F{i}"):
            if self.first:
                inp = mbs[i][0].to(self._dev)
            else:
                inp.requires_grad_()
            out = self._forward(inp)
            if self.last:
                loss = loss_fn(out, mbs[i][1].to(self._dev)) / self.m
                self._losses.append(loss.detach())
                out = loss
            self._saved[i] = (inp, out)
        return out

    def _bwd(self, i, grad_out):
        with self.timeline.span(f"B{i}"):
            inp, out = self._saved.pop(i)
            if self.last:
                out.backward()
            else:
                out.backward(grad_out)
            return None if self.first else inp.grad

    # ---- transfers with the neighbours
    def _empty(self):
        return torch.empty(self._act_shape, device=self._dev)

    def _recv_forward(self):
        if self.first:
            return None
        t = self._empty()
        self.g.recv(t, self.g.rank - 1)
        return t

    def _send_forward(self, out):
        if not self.last:
            self.g.send(out.detach().contiguous(), self.g.rank + 1)

    def _recv_backward(self):
        if self.last:
            return None
        t = self._empty()
        self.g.recv(t, self.g.rank + 1)
        return t

    def _send_backward(self, grad):
        if not self.first:
            self.g.send(grad.contiguous(), self.g.rank - 1)

    def _send_forward_recv_backward(self, out):
        if self.last:
            return None
        t = self._empty()
        self.g.sendrecv(out.detach().contiguous(), self.g.rank + 1, t, self.g.rank + 1)
        return t

    def _send_backward_recv_forward(self, grad):
        if self.first:
            return None
        t = self._empty()
        self.g.sendrecv(grad.contiguous(), self.g.rank - 1, t, self.g.rank - 1)
        return t

    # ---- schedules
    def _gpipe(self, mbs):
        for i in range(self.m):
            out = self._fwd(i, mbs, self._recv_forward())
            self._send_forward(out)
        for i in range(self.m):
            self._send_backward(self._bwd(i, self._recv_backward()))

    def _one_f_one_b(self, mbs):
        warmup = min(self.g.size - self.g.rank - 1, self.m)
        steady = self.m - warmup
        for i in range(warmup):
            out = self._fwd(i, mbs, self._recv_forward())
            self._send_forward(out)
        inp = self._recv_forward() if steady > 0 else None
        for k in range(steady):
            out = self._fwd(warmup + k, mbs, inp)
            grad_out = self._send_forward_recv_backward(out)
            grad_in = self._bwd(k, grad_out)
            if k == steady - 1:
                self._send_backward(grad_in)
            else:
                inp = self._send_backward_recv_forward(grad_in)
        for k in range(steady, self.m):
            self._send_backward(self._bwd(k, self._recv_backward()))
