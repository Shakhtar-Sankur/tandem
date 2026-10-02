"""Collectives between processes on one machine, without torch.distributed or NCCL.

Each ordered pair of ranks (src, dst) has a mailbox: `slots` buffers of
`slot_bytes` each, allocated by src, and two counting semaphores: pieces
waiting to be read, and slots free to write. A message larger than a slot
travels as a sequence of pieces. Where the buffers live is the transport:

  CPU ranks          shared memory.
  GPU, "device"      on src's GPU, shared through CUDA IPC; dst copies
                     each piece straight from src's GPU.
  GPU, "host"        pinned shared host memory, as NCCL's shared-memory
                     transport: src copies GPU to host, dst host to GPU, and
                     no process ever queues work on another's GPU.

  put (src):  take a free slot, copy the piece in, finish the copy (on a
              GPU, synchronize the stream), then signal a piece.
  take (dst): take a piece, copy (or add) it out of the slot, finish the
              copy, then signal a free slot.

A waiting thread sleeps in the kernel, without Python's global lock, so a
collective in flight never slows the training thread down (an earlier
version polled counters from Python, and on 2 T4s its communication took
5x longer while backward ran beside it). Semaphores also order memory: what
was written before a signal is visible after the wait. The
collectives are built from these two operations: ring reduce-scatter and
all-gather (so all-reduce), a pipelined broadcast, and point-to-point
send/recv for pipeline parallelism.

Every collective runs on one background thread per rank (and, on GPUs, on
its own CUDA stream, high priority by default as torch FSDP's are, so its
small kernels are scheduled ahead of waiting compute), in submission order,
so ranks agree on the order of their collectives and the training thread
can keep computing while gradients travel: `all_reduce(..., async_op=True)`
returns a Work to wait on.
"""

import gc
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
DEFAULT_TRANSPORT = os.environ.get("TANDEM_TRANSPORT", "device")
DEFAULT_PRIORITY = int(os.environ.get("TANDEM_PRIORITY", "-1"))  # lower is more urgent; 0 is normal
DEFAULT_BACKEND = os.environ.get("TANDEM_BACKEND", "python")  # or "cpp": the C++ engine (CPU), docs/engine.md


class CommError(RuntimeError):
    pass


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


class _CppWork(Work):
    """A collective in flight on the C++ engine's thread."""

    def __init__(self, name, group, handle, result, nbytes):
        super().__init__(name, group)
        self._handle, self.result, self._nbytes = handle, result, nbytes
        self._recorded = False

    def wait(self):
        g = self._group
        t0 = time.perf_counter() if g.tracing and not self._handle.done() else None
        try:
            self._handle.wait()
        except Exception as e:
            raise CommError(f"{self.name} failed") from e
        if t0 is not None:
            g.waits.append((self.name, t0, time.perf_counter()))
        if g.tracing and not self._recorded:  # steady clock == perf_counter's clock
            g.trace.append((self.name, self._handle.started, self._handle.finished, self._nbytes))
        self._recorded = True
        return self.result

    def is_completed(self):
        return self._handle.done()


class Group:
    """One rank's view of the group: its mailboxes, the control block, and
    the thread that runs its collectives."""

    def __init__(self, rank, size, device, sync, queues, slot_bytes=DEFAULT_SLOT_BYTES,
                 slots=DEFAULT_SLOTS, timeout=DEFAULT_TIMEOUT, transport=None, priority=None,
                 backend=None, job=None):
        self.rank, self.size = rank, size
        self.device = torch.device(device)
        self.slot_bytes, self.slots, self.timeout = slot_bytes, slots, timeout
        self.transport = transport or DEFAULT_TRANSPORT
        self.priority = DEFAULT_PRIORITY if priority is None else priority
        if self.transport not in ("device", "host"):
            raise CommError(f"unknown transport {self.transport}")
        self._pinned = []  # host buffers registered with CUDA
        self._full, self._free, self._barrier = sync
        self._queues = queues
        self.out = {}  # dst -> [slots, slot_bytes] uint8, ours
        self.inbox = {}  # src -> [slots, slot_bytes] uint8, theirs
        self._sent = {q: 0 for q in range(size)}  # pieces sent to q
        self._got = {q: 0 for q in range(size)}  # pieces taken from q
        self._scratch = None  # for adding a piece that lives on another GPU
        self.trace = []  # (name, start, end, bytes) of each collective, host clock
        self.waits = []  # (name, start, end): time a caller spent blocked on a collective
        self.tracing = False
        self._jobs = queue.Queue()
        self._closed = False
        self.backend = backend or DEFAULT_BACKEND
        self._engine = None
        if self.backend == "cpp":
            self._start_engine(job)
            return
        if self.backend != "python":
            raise CommError(f"unknown backend {self.backend}")
        self._rendezvous()
        self._thread = threading.Thread(target=self._loop, name=f"tandem-comm-{rank}", daemon=True)
        self._thread.start()
        self.barrier()

    def _start_engine(self, job):
        if self.device.type != "cpu":
            raise CommError("the C++ engine runs on CPU tensors so far (CUDA is milestone M3)")
        if job is None:
            raise CommError("the C++ engine needs a job name (launch() makes one)")
        from .engine import lib

        self._engine = lib().Engine(job, self.rank, self.size, self.slots, self.slot_bytes, self.timeout)
        self._barrier.wait(timeout=self.timeout)  # every rank has created its outgoing channels
        self._engine.connect()
        self.barrier()

    def _cpp(self, name, start, result, nbytes, async_op):
        if self._closed:
            raise CommError("group is closed")
        try:
            handle = start()
        except (ValueError, RuntimeError) as e:
            raise CommError(f"{name}: {e}") from e
        work = _CppWork(name, self, handle, result, nbytes)
        return work if async_op else work.wait()

    def _sem(self, sems, src, dst):
        return sems[src * self.size + dst]

    @staticmethod
    def make_sync(ctx, size, slots):
        """The semaphores and barrier, created by the launcher and inherited
        by every rank."""
        full = [ctx.Semaphore(0) for _ in range(size * size)]
        free = [ctx.Semaphore(slots) for _ in range(size * size)]
        return full, free, ctx.Barrier(size)

    def _acquire(self, sem, what):
        if not sem.acquire(timeout=self.timeout):
            raise CommError(f"timed out after {self.timeout:.0f}s waiting for {what}")

    def _rendezvous(self):
        on_host = self.device.type == "cpu" or self.transport == "host"
        for dst in range(self.size):
            if dst == self.rank:
                continue
            buf = torch.zeros(self.slots, self.slot_bytes, dtype=torch.uint8,
                              device="cpu" if on_host else self.device)
            if on_host:
                buf.share_memory_()
            self.out[dst] = buf
            self._queues[dst].put((self.rank, buf))
        while len(self.inbox) < self.size - 1:
            src, buf = self._queues[self.rank].get(timeout=self.timeout)
            self.inbox[src] = buf
        if self.device.type == "cuda":
            self._scratch = torch.empty(self.slot_bytes, dtype=torch.uint8, device=self.device)
            if on_host:  # page-locked, so copies to and from it are asynchronous DMA
                for buf in [*self.out.values(), *self.inbox.values()]:
                    err = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), buf.numel(), 0)
                    if int(err) != 0:
                        raise CommError(f"cudaHostRegister failed: {err}")
                    self._pinned.append(buf.data_ptr())

    # ---- the communication thread
    def _loop(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            self._stream = torch.cuda.Stream(self.device, priority=self.priority)
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
        self._acquire(self._sem(self._free, self.rank, dst), f"rank {dst} to free a slot")
        seq = self._sent[dst]
        self._sent[dst] = seq + 1
        nb = piece.numel() * piece.element_size()
        slot = self.out[dst][seq % self.slots, :nb]
        slot.copy_(piece.reshape(-1).view(torch.uint8), non_blocking=True)
        self._finish()
        self._sem(self._full, self.rank, dst).release()

    def _take(self, src, into, reduce=False):
        self._acquire(self._sem(self._full, src, self.rank), f"a message from rank {src}")
        seq = self._got[src]
        self._got[src] = seq + 1
        nb = into.numel() * into.element_size()
        slot = self.inbox[src][seq % self.slots, :nb].view(into.dtype)
        if reduce:
            if slot.device == into.device:
                into.add_(slot)
            else:  # copy across GPUs first, into a reused buffer
                tmp = self._scratch[:nb].view(into.dtype)
                tmp.copy_(slot, non_blocking=True)
                into.add_(tmp)
        else:
            into.copy_(slot, non_blocking=True)
        self._finish()
        self._sem(self._free, src, self.rank).release()

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
        if self._engine is not None:
            return self._cpp("all_reduce", lambda: self._engine.all_reduce(flat, op == "avg"), t,
                             t.numel() * t.element_size(), async_op)

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
        if self._engine is not None:
            mine = self._chunks(flat, sizes)[self.rank]
            return self._cpp("reduce_scatter", lambda: self._engine.reduce_scatter(flat, list(sizes or []), op == "avg"),
                             mine, t.numel() * t.element_size(), async_op)

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
        if self._engine is not None:
            self._chunks(flat, sizes)
            return self._cpp("all_gather", lambda: self._engine.all_gather(flat, list(sizes or [])), t,
                             t.numel() * t.element_size(), async_op)

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
        if self._engine is not None:
            return self._cpp("broadcast", lambda: self._engine.broadcast(flat, root), t,
                             t.numel() * t.element_size(), async_op)

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
        if self._engine is not None:
            return self._cpp(f"send->{dst}", lambda: self._engine.send(flat, dst), t,
                             t.numel() * t.element_size(), async_op)

        def run():
            self._exchange(flat, dst, None, None, False)
            return t

        return self.submit(f"send->{dst}", run, self._ready_event(t), t.numel() * t.element_size(), async_op)

    def recv(self, t, src, async_op=False):
        flat = self._flat(t)
        if self._engine is not None:
            return self._cpp(f"recv<-{src}", lambda: self._engine.recv(flat, src), t,
                             t.numel() * t.element_size(), async_op)

        def run():
            self._exchange(None, None, flat, src, False)
            return t

        return self.submit(f"recv<-{src}", run, None, t.numel() * t.element_size(), async_op)

    def sendrecv(self, send, dst, recv, src, async_op=False):
        """Sends one tensor while receiving another, piece by piece in
        lockstep (pipeline parallelism's paired transfers)."""
        sf, rf = self._flat(send), self._flat(recv)
        if self._engine is not None:
            return self._cpp(f"sendrecv->{dst}<-{src}", lambda: self._engine.sendrecv(sf, dst, rf, src), recv,
                             (send.numel() + recv.numel()) * send.element_size(), async_op)

        def run():
            self._exchange(sf, dst, rf, src, False)
            return recv

        return self.submit(f"sendrecv->{dst}<-{src}", run, self._ready_event(send),
                           (send.numel() + recv.numel()) * send.element_size(), async_op)

    def barrier(self):
        if self._engine is not None:
            return self._cpp("barrier", self._engine.barrier, None, 0, False)

        def run():
            try:
                self._barrier.wait(timeout=self.timeout)
            except threading.BrokenBarrierError as e:
                raise CommError("barrier broken: a rank failed or timed out") from e

        return self.submit("barrier", run)

    def close(self):
        if self._closed:
            return
        self.barrier()
        if self._engine is not None:
            # A rank's shared memory outlives its name and mapping for as long
            # as a peer still maps it, so each rank simply lets go of its own.
            self._closed = True
            self._engine.close()
            self._engine = None
            return

        def release():
            # Every rank drops the other ranks' buffers, then, once all have,
            # frees its own: no rank frees memory another still maps.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                for ptr in self._pinned:
                    torch.cuda.cudart().cudaHostUnregister(ptr)
                self._pinned.clear()
            self.inbox.clear()
            gc.collect()
            self._barrier.wait(timeout=self.timeout)
            self.out.clear()
            if self.device.type == "cuda":
                torch.cuda.ipc_collect()

        self.submit("release", release)
        self.barrier()
        self._closed = True
        self._jobs.put(None)
        self._thread.join()


def _worker(rank, size, fn, args, sync, queues, device, slot_bytes, slots, transport, priority, backend, job,
            results):
    if device == "cuda":
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
    else:
        dev = torch.device("cpu")
        torch.set_num_threads(max(1, int(os.environ.get("TANDEM_CPU_THREADS", "1"))))
    g = Group(rank, size, dev, sync, queues, slot_bytes, slots, transport=transport, priority=priority,
              backend=backend, job=job)
    try:
        out = fn(g, *args)
        g.close()
    except BaseException:
        g._closed = True
        raise
    if out is not None:
        torch.save(out, os.path.join(results, f"rank{rank}.pt"))


def launch(fn, size, *args, device="cpu", slot_bytes=DEFAULT_SLOT_BYTES, slots=DEFAULT_SLOTS,
           transport=None, priority=None, backend=None):
    """Runs fn(group, *args) on `size` processes (one per GPU with
    device="cuda") and returns each rank's return value, in rank order.
    transport ("device" or "host") and priority (the communication
    stream's) only matter on GPUs. backend: "python" (default) or "cpp", the
    C++ engine (or set TANDEM_BACKEND).
    Return values travel back pickled, so keep them small (CPU tensors,
    numbers, lists)."""
    import tempfile
    import uuid

    backend = backend or DEFAULT_BACKEND
    if backend == "cpp":
        from .engine import lib

        lib()  # build once here, not in every rank at once
    job = uuid.uuid4().hex[:12]
    ctx = mp.get_context("spawn")
    sync = Group.make_sync(ctx, size, slots)
    queues = [ctx.Queue() for _ in range(size)]
    with tempfile.TemporaryDirectory(prefix="tandem-") as results:
        mp.spawn(_worker, args=(size, fn, args, sync, queues, device, slot_bytes, slots, transport, priority,
                                backend, job, results),
                 nprocs=size, join=True)
        out = []
        for r in range(size):
            path = os.path.join(results, f"rank{r}.pt")
            out.append(torch.load(path, weights_only=False) if os.path.exists(path) else None)
    return out
