"""Collectives between processes on one machine, without torch.distributed or NCCL.

Each ordered pair of ranks (src, dst) has a mailbox: `slots` buffers of
`slot_bytes` each, allocated by src (in shared memory for CPU ranks, or on
src's GPU and shared through CUDA IPC), and two counters in a shared control
block: messages src has published and messages dst has consumed. A message
larger than a slot travels as a sequence of pieces.

  put (src):  wait for a free slot (published - consumed < slots), copy the
              piece in, finish the copy, then publish.
  take (dst): wait until a piece is published, copy (or add) it out of the
              slot into the destination, finish the copy, then consume.

A counter has one writer, and a writer completes its data copy before it
writes the counter, so a reader that sees the counter sees the data. The
collectives are built from these two operations: ring reduce-scatter and
all-gather (so all-reduce), a pipelined broadcast, and point-to-point
send/recv for pipeline parallelism.

Every collective runs on one background thread per rank (and, on GPUs, on
its own CUDA stream), in submission order, so ranks agree on the order of
their collectives and the training thread can keep computing while
gradients travel: `all_reduce(..., async_op=True)` returns a Work to wait on.
"""

import math
import os
import queue
import threading
import time

import torch
import torch.multiprocessing as mp

DEFAULT_SLOT_BYTES = 4 << 20
DEFAULT_SLOTS = 2
DEFAULT_TIMEOUT = float(os.environ.get("TANDEM_TIMEOUT", "600"))


class CommError(RuntimeError):
    pass


def _spin(cond, timeout, what):
    """Waits until cond() holds: busy at first, then yielding the CPU (and the
    GIL) between checks so the training thread keeps running."""
    if cond():
        return
    t0 = time.monotonic()
    n = 0
    while not cond():
        n += 1
        if n > 100:
            time.sleep(2e-5)
        if n % 2048 == 0 and time.monotonic() - t0 > timeout:
            raise CommError(f"timed out after {timeout:.0f}s waiting for {what}")


class Work:
    """A collective in flight on the communication thread."""

    def __init__(self, name, group=None):
        self.name = name
        self._group = group
        self._done = threading.Event()
        self._error = None
        self.result = None

    def wait(self):
        g = self._group
        if g is not None and g.tracing and not self._done.is_set():
            t0 = time.perf_counter()
            self._done.wait()
            g.waits.append((self.name, t0, time.perf_counter()))
        self._done.wait()
        if self._error is not None:
            raise CommError(f"{self.name} failed") from self._error
        return self.result

    def is_completed(self):
        return self._done.is_set()


class Group:
    """One rank's view of the group: its mailboxes, the control block, and
    the thread that runs its collectives."""

    def __init__(self, rank, size, device, ctrl, queues, slot_bytes=DEFAULT_SLOT_BYTES,
                 slots=DEFAULT_SLOTS, timeout=DEFAULT_TIMEOUT):
        self.rank, self.size = rank, size
        self.device = torch.device(device)
        self.slot_bytes, self.slots, self.timeout = slot_bytes, slots, timeout
        self._ctrl_t = ctrl  # keeps the shared memory alive
        self.ctrl = ctrl.numpy()
        self._queues = queues
        self.out = {}  # dst -> [slots, slot_bytes] uint8, ours
        self.inbox = {}  # src -> [slots, slot_bytes] uint8, theirs
        self.trace = []  # (name, start, end, bytes) of each collective, host clock
        self.waits = []  # (name, start, end): time a caller spent blocked on a collective
        self.tracing = False
        self._jobs = queue.Queue()
        self._closed = False
        self._rendezvous()
        self._thread = threading.Thread(target=self._loop, name=f"tandem-comm-{rank}", daemon=True)
        self._thread.start()
        self.barrier()

    # ---- control block layout: published[src, dst], consumed[src, dst], barrier[rank]
    def _pub(self, src, dst):
        return src * self.size + dst

    def _con(self, src, dst):
        return self.size * self.size + src * self.size + dst

    def _bar(self, rank):
        return 2 * self.size * self.size + rank

    @staticmethod
    def ctrl_len(size):
        return 2 * size * size + size

    def _rendezvous(self):
        for dst in range(self.size):
            if dst == self.rank:
                continue
            buf = torch.zeros(self.slots, self.slot_bytes, dtype=torch.uint8, device=self.device)
            if buf.device.type == "cpu":
                buf.share_memory_()
            self.out[dst] = buf
            self._queues[dst].put((self.rank, buf))
        while len(self.inbox) < self.size - 1:
            src, buf = self._queues[self.rank].get(timeout=self.timeout)
            self.inbox[src] = buf

    # ---- the communication thread
    def _loop(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            self._stream = torch.cuda.Stream(self.device)
        while True:
            job = self._jobs.get()
            if job is None:
                return
            work, fn, ready, nbytes = job
            try:
                if ready is not None:
                    ready.synchronize()
                t0 = time.perf_counter()
                if self.device.type == "cuda":
                    with torch.cuda.stream(self._stream):
                        work.result = fn()
                else:
                    work.result = fn()
                if self.tracing:
                    self.trace.append((work.name, t0, time.perf_counter(), nbytes))
            except BaseException as e:  # reported to whoever waits
                work._error = e
            work._done.set()

    def submit(self, name, fn, ready=None, nbytes=0, async_op=False):
        """Runs fn on the communication thread after `ready` (a CUDA event
        recorded where its inputs were produced) completes."""
        if self._closed:
            raise CommError("group is closed")
        work = Work(name, self)
        self._jobs.put((work, fn, ready, nbytes))
        return work if async_op else work.wait()

    def _ready_event(self, *tensors):
        if self.device.type != "cuda":
            return None
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.device))
        return ev

    def _finish(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    # ---- pieces
    def _piece_elems(self, t):
        return max(1, self.slot_bytes // t.element_size())

    def _put(self, dst, piece):
        seq = int(self.ctrl[self._pub(self.rank, dst)])
        con = self._con(self.rank, dst)
        _spin(lambda: seq - self.ctrl[con] < self.slots, self.timeout, f"rank {dst} to free a slot")
        nb = piece.numel() * piece.element_size()
        slot = self.out[dst][seq % self.slots, :nb]
        slot.copy_(piece.reshape(-1).view(torch.uint8), non_blocking=True)
        self._finish()
        self.ctrl[self._pub(self.rank, dst)] = seq + 1

    def _take(self, src, into, reduce=False):
        con = self._con(src, self.rank)
        seq = int(self.ctrl[con])
        pub = self._pub(src, self.rank)
        _spin(lambda: self.ctrl[pub] > seq, self.timeout, f"a message from rank {src}")
        nb = into.numel() * into.element_size()
        slot = self.inbox[src][seq % self.slots, :nb].view(into.dtype)
        if reduce:
            if slot.device == into.device:
                into.add_(slot)
            else:
                into.add_(slot.to(into.device, non_blocking=True))
        else:
            into.copy_(slot, non_blocking=True)
        self._finish()
        self.ctrl[con] = seq + 1

    def _exchange(self, send, dst, recv, src, reduce):
        """Sends `send` to dst while receiving into `recv` from src, piece by
        piece and in lockstep, so a ring of ranks all sending at once never
        waits on itself."""
        pe = self._piece_elems(send if send is not None else recv)
        ns = math.ceil(send.numel() / pe) if send is not None else 0
        nr = math.ceil(recv.numel() / pe) if recv is not None else 0
        for k in range(max(ns, nr)):
            if k < ns:
                self._put(dst, send[k * pe:(k + 1) * pe])
            if k < nr:
                self._take(src, recv[k * pe:(k + 1) * pe], reduce)

    # ---- collectives (run on the communication thread)
    def _ring_reduce_scatter(self, chunks):
        n, r = self.size, self.rank
        nxt, prv = (r + 1) % n, (r - 1) % n
        for s in range(n - 1):
            self._exchange(chunks[(r - s - 1) % n], nxt, chunks[(r - s - 2) % n], prv, True)

    def _ring_all_gather(self, chunks):
        n, r = self.size, self.rank
        nxt, prv = (r + 1) % n, (r - 1) % n
        for s in range(n - 1):
            self._exchange(chunks[(r - s) % n], nxt, chunks[(r - s - 1) % n], prv, False)

    @staticmethod
    def _flat(t):
        if not t.is_contiguous():
            raise CommError("collectives need contiguous tensors")
        return t.reshape(-1)

    def _chunks(self, flat, sizes):
        if sizes is None:
            return list(torch.tensor_split(flat, self.size))
        if len(sizes) != self.size or sum(sizes) != flat.numel():
            raise CommError(f"chunk sizes {sizes} do not partition {flat.numel()} elements")
        return list(torch.split(flat, list(sizes)))

    def all_reduce(self, t, op="sum", async_op=False):
        """Ring all-reduce in place: reduce-scatter, then all-gather."""
        if op not in ("sum", "avg"):
            raise CommError(f"unknown op {op}")
        flat = self._flat(t)

        def run():
            if op == "avg":
                # Divided before summing, as torch DDP and FSDP do: exact for
                # a power-of-two group, and no overflow in the sum.
                flat.div_(self.size)
            if self.size > 1:
                chunks = self._chunks(flat, None)
                self._ring_reduce_scatter(chunks)
                self._ring_all_gather(chunks)
            self._finish()
            return t

        return self.submit("all_reduce", run, self._ready_event(t), t.numel() * t.element_size(), async_op)

    def reduce_scatter(self, t, sizes=None, op="sum", async_op=False):
        """Sums `t` across ranks, leaving rank r's chunk (chunk r of
        torch.tensor_split, or of `sizes`) reduced in place; returns that view.
        The other chunks hold partial sums afterwards."""
        flat = self._flat(t)

        def run():
            if op == "avg":
                flat.div_(self.size)
            chunks = self._chunks(flat, sizes)
            if self.size > 1:
                self._ring_reduce_scatter(chunks)
            mine = chunks[self.rank]
            self._finish()
            return mine

        return self.submit("reduce_scatter", run, self._ready_event(t), t.numel() * t.element_size(), async_op)

    def all_gather(self, t, sizes=None, async_op=False):
        """Each rank holds its own chunk of `t` (chunk r); fills in the rest."""
        flat = self._flat(t)

        def run():
            if self.size > 1:
                self._ring_all_gather(self._chunks(flat, sizes))
            self._finish()
            return t

        return self.submit("all_gather", run, self._ready_event(t), t.numel() * t.element_size(), async_op)

    def broadcast(self, t, root=0, async_op=False):
        """Pipelined along the ring from root: each rank forwards each piece
        as soon as it has it."""
        flat = self._flat(t)

        def run():
            n = self.size
            pos = (self.rank - root) % n
            nxt, prv = (self.rank + 1) % n, (self.rank - 1) % n
            pe = self._piece_elems(flat)
            for k in range(math.ceil(flat.numel() / pe)):
                piece = flat[k * pe:(k + 1) * pe]
                if pos > 0:
                    self._take(prv, piece)
                if pos < n - 1:
                    self._put(nxt, piece)
            self._finish()
            return t

        return self.submit("broadcast", run, self._ready_event(t), t.numel() * t.element_size(), async_op)

    def send(self, t, dst, async_op=False):
        flat = self._flat(t)

        def run():
            self._exchange(flat, dst, None, None, False)
            return t

        return self.submit(f"send->{dst}", run, self._ready_event(t), t.numel() * t.element_size(), async_op)

    def recv(self, t, src, async_op=False):
        flat = self._flat(t)

        def run():
            self._exchange(None, None, flat, src, False)
            return t

        return self.submit(f"recv<-{src}", run, None, t.numel() * t.element_size(), async_op)

    def sendrecv(self, send, dst, recv, src, async_op=False):
        """Sends one tensor while receiving another, piece by piece in
        lockstep (pipeline parallelism's paired transfers)."""
        sf, rf = self._flat(send), self._flat(recv)

        def run():
            self._exchange(sf, dst, rf, src, False)
            return recv

        return self.submit(f"sendrecv->{dst}<-{src}", run, self._ready_event(send),
                           (send.numel() + recv.numel()) * send.element_size(), async_op)

    def barrier(self):
        def run():
            i = self._bar(self.rank)
            gen = int(self.ctrl[i]) + 1
            self.ctrl[i] = gen
            for q in range(self.size):
                j = self._bar(q)
                _spin(lambda: self.ctrl[j] >= gen, self.timeout, f"rank {q} at the barrier")

        return self.submit("barrier", run)

    def close(self):
        if self._closed:
            return
        self.barrier()
        self._closed = True
        self._jobs.put(None)
        self._thread.join()
        self.inbox.clear()  # release the peers' buffers before they exit


def _worker(rank, size, fn, args, ctrl, queues, device, slot_bytes, slots, results):
    if device == "cuda":
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
    else:
        dev = torch.device("cpu")
        torch.set_num_threads(max(1, int(os.environ.get("TANDEM_CPU_THREADS", "1"))))
    g = Group(rank, size, dev, ctrl, queues, slot_bytes, slots)
    try:
        out = fn(g, *args)
        g.close()
    except BaseException:
        g._closed = True
        raise
    if out is not None:
        torch.save(out, os.path.join(results, f"rank{rank}.pt"))


def launch(fn, size, *args, device="cpu", slot_bytes=DEFAULT_SLOT_BYTES, slots=DEFAULT_SLOTS):
    """Runs fn(group, *args) on `size` processes (one per GPU with
    device="cuda") and returns each rank's return value, in rank order.
    Return values travel back pickled, so keep them small (CPU tensors,
    numbers, lists)."""
    import tempfile

    ctx = mp.get_context("spawn")
    ctrl = torch.zeros(Group.ctrl_len(size), dtype=torch.int64).share_memory_()
    queues = [ctx.Queue() for _ in range(size)]
    with tempfile.TemporaryDirectory(prefix="tandem-") as results:
        mp.spawn(_worker, args=(size, fn, args, ctrl, queues, device, slot_bytes, slots, results),
                 nprocs=size, join=True)
        out = []
        for r in range(size):
            path = os.path.join(results, f"rank{r}.pt")
            out.append(torch.load(path, weights_only=False) if os.path.exists(path) else None)
    return out
