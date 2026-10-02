"""Where a training step's time goes, per rank: compute spans recorded on
the training thread, communication spans recorded by the communication
thread (Group.trace), and a Chrome trace (chrome://tracing, Perfetto) with
one row for each. On GPUs a span waits for the device at both ends, so
profiling slows training slightly; it is off unless asked for."""

import contextlib
import json
import time

import torch


class Timeline:
    def __init__(self, enabled=True, device=None):
        self.enabled = enabled
        self.device = torch.device(device) if device is not None else None
        self.spans = []  # (name, start, end)

    def _sync(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def span(self, name):
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.spans.append((name, t0, time.perf_counter()))


def _union(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _intersection(a, b):
    a, b = _union(a), _union(b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def summarize(compute_spans, comm_spans, waits, step_time):
    """Communication time on the communication thread, and how much of it
    the training thread saw: the time it spent blocked on a collective
    while one was running (exposed). Waiting for the GPU to finish the
    backward a collective needs is compute, not communication, so it does
    not count. overlap = the share of communication hidden behind compute."""
    comm = sum(e - s for _, s, e, _ in comm_spans)
    exposed = _intersection([(s, e) for _, s, e in waits], [(s, e) for _, s, e, _ in comm_spans])
    return {
        "step_s": step_time,
        "compute_s": sum(e - s for _, s, e in compute_spans),
        "comm_s": comm,
        "exposed_comm_s": exposed,
        "overlap": max(0.0, 1 - exposed / comm) if comm > 0 else None,
        "comm_bytes": sum(b for *_, b in comm_spans),
    }


def chrome_trace(per_rank, path):
    """per_rank: list of (compute_spans, comm_spans) on a shared clock."""
    t0 = min([s for comp, comm in per_rank for _, s, _ in comp] + [s for comp, comm in per_rank for _, s, _, _ in comm] or [0])
    events = []
    for r, (comp, comm) in enumerate(per_rank):
        events.append({"ph": "M", "pid": r, "name": "process_name", "args": {"name": f"rank {r}"}})
        for tid, name in ((0, "compute"), (1, "communication")):
            events.append({"ph": "M", "pid": r, "tid": tid, "name": "thread_name", "args": {"name": name}})
        for name, s, e in comp:
            events.append({"ph": "X", "pid": r, "tid": 0, "name": name, "ts": (s - t0) * 1e6, "dur": (e - s) * 1e6})
        for name, s, e, b in comm:
            events.append({"ph": "X", "pid": r, "tid": 1, "name": name, "ts": (s - t0) * 1e6,
                           "dur": (e - s) * 1e6, "args": {"bytes": b}})
    with open(path, "w") as f:
        json.dump({"traceEvents": events}, f)
