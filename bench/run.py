"""tandem against torch.distributed, on 2 GPUs (Kaggle's 2x T4) or on CPU.

  verify     a small GPT, deterministic kernels: every strategy's losses and
             weights against torch DDP (and torch FSDP), bit for bit; the
             pipeline against one process accumulating gradients.
  allreduce  tandem's ring all-reduce against NCCL (gloo on CPU), by size.
  bench      a larger GPT: step time, tokens/s and peak memory per GPU for
             each strategy, and how much of tandem DDP's communication
             hides behind backward.

usage: python bench/run.py [verify|allreduce|bench|all] [--device cuda|cpu]
       [--out results.jsonl]. Prints one JSON line per measurement.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.multiprocessing as mp

from tandem import launch
from tandem.data import Text, synthetic
from tandem.ddp import DDP
from tandem.model import GPT, Config, count_params, loss_fn
from tandem.optim import ShardedAdamW
from tandem.pipeline import Pipeline
from tandem.profile import Timeline, summarize
from tandem.zero import FSDP, ZeRO

LR, WD = 3e-4, 0.1
OUT = None


def emit(row):
    line = json.dumps(row)
    print(line, flush=True)
    if OUT:
        with open(OUT, "a") as f:
            f.write(line + "\n")


def setup(device, rank, deterministic):
    if device == "cuda":
        torch.cuda.set_device(rank)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            # The math attention kernel: deterministic in backward.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        return torch.device("cuda", rank)
    torch.set_num_threads(1)
    return torch.device("cpu")


def corpus(kind):
    if kind == "synthetic":
        return synthetic(200_000)
    path = os.path.join(tempfile.gettempdir(), "tinyshakespeare.txt")
    if not os.path.exists(path):
        import urllib.request

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            return synthetic(1_000_000)
    return Text(path)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def flat_weights(params):
    return torch.cat([p.detach().reshape(-1).float().cpu() for p in params])


# ---------------------------------------------------------------- tandem
def tandem_rank(g, strategy, cfg, steps, batch, micro, deterministic, data, want_weights, profile):
    dev = setup(g.device.type, g.rank, deterministic)
    torch.manual_seed(0)
    text = corpus(data)
    model = GPT(cfg).to(dev)
    tl = Timeline(enabled=profile, device=dev)
    world, rank = g.size, g.rank
    pipe = None
    if strategy.startswith("ddp"):
        wrapped = DDP(model, g, overlap=strategy != "ddp-nooverlap")
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=False)
    elif strategy in ("zero1", "zero2"):
        sopt = ShardedAdamW(lr=LR, weight_decay=WD)
        wrapped = ZeRO(model, g, int(strategy[-1]), sopt)
    elif strategy == "zero3":
        sopt = ShardedAdamW(lr=LR, weight_decay=WD)
        wrapped = FSDP(model, g, sopt)
    else:  # pp-gpipe, pp-1f1b
        pipe = Pipeline(model, g, micro, strategy[3:], timeline=tl)
        opt = torch.optim.AdamW(pipe.parameters(), lr=LR, weight_decay=WD, foreach=False)
        world, rank = 1, 0  # every stage sees the whole batch
    losses, times = [], []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for step in range(steps):
        x, y = text.batch(step, batch, cfg.seq, rank=rank, world=world, device=dev)
        if profile and step == steps - 1:
            g.trace.clear()
            g.waits.clear()
            g.tracing = True
            tl.spans.clear()
        sync(dev)
        t0 = time.perf_counter()
        if pipe is not None:
            loss = pipe.train_step(x, y)
            opt.step()
            opt.zero_grad()
        else:
            with tl.span("forward"):
                loss = wrapped(x, y)
            with tl.span("backward"):
                loss.backward()
            with tl.span("optimizer"):
                if strategy.startswith("ddp"):
                    opt.step()
                    opt.zero_grad()
                else:
                    wrapped.step()
                    wrapped.zero_grad()
        sync(dev)
        times.append(time.perf_counter() - t0)
        losses.append(None if loss is None else loss.item())
    g.tracing = False
    out = {"losses": losses, "times": times,
           "peak_mem": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None}
    if profile:
        comm = [s for s in g.trace if s[0] not in ("barrier",)]
        out["profile"] = summarize(tl.spans, comm, list(g.waits), times[-1])
        out["trace"] = (tl.spans, comm)
    if want_weights:
        if strategy == "zero3":
            out["weights"] = wrapped.full_parameters().float().cpu()
        elif pipe is not None:
            out["weights"] = flat_weights(pipe.parameters())
        else:
            out["weights"] = flat_weights(model.parameters())
    return out


def run_tandem(strategy, device, **kw):
    return launch(tandem_rank, 2, strategy, kw["cfg"], kw["steps"], kw["batch"], kw.get("micro", 4),
                  kw.get("deterministic", False), kw.get("data", "synthetic"), kw.get("want_weights", False),
                  kw.get("profile", False), device=device)


# ---------------------------------------------------------------- torch.distributed
def torch_rank(rank, world, init, out, device, strategy, cfg, steps, batch, deterministic, data, want_weights):
    import torch.distributed as dist

    dev = setup(device, rank, deterministic)
    dist.init_process_group("nccl" if device == "cuda" else "gloo", init_method=f"file://{init}",
                            rank=rank, world_size=world)
    torch.manual_seed(0)
    text = corpus(data)
    model = GPT(cfg).to(dev)
    if strategy == "torch-ddp":
        from torch.nn.parallel import DistributedDataParallel

        wrapped = DistributedDataParallel(model, device_ids=[rank] if device == "cuda" else None)
    else:
        from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        from tandem.model import Block, Embed, Head

        wrapped = FullyShardedDataParallel(
            model, auto_wrap_policy=ModuleWrapPolicy({Embed, Block, Head}),
            sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=dev if device == "cuda" else None,
            use_orig_params=True)
    opt = torch.optim.AdamW(wrapped.parameters(), lr=LR, weight_decay=WD, foreach=False)
    losses, times = [], []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for step in range(steps):
        x, y = text.batch(step, batch, cfg.seq, rank=rank, world=world, device=dev)
        sync(dev)
        t0 = time.perf_counter()
        loss = wrapped(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        sync(dev)
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
    res = {"losses": losses, "times": times,
           "peak_mem": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None}
    if want_weights:
        if strategy == "torch-fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel

            with FullyShardedDataParallel.summon_full_params(wrapped):
                res["weights"] = flat_weights(model.parameters())
        else:
            res["weights"] = flat_weights(model.parameters())
    torch.save(res, f"{out}.{rank}")
    dist.destroy_process_group()


def run_torch(strategy, device, cfg, steps, batch, deterministic=False, data="synthetic", want_weights=False):
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(torch_rank, args=(2, os.path.join(d, "init"), os.path.join(d, "out"), device, strategy, cfg,
                                   steps, batch, deterministic, data, want_weights), nprocs=2)
        return [torch.load(os.path.join(d, f"out.{r}"), weights_only=False) for r in range(2)]


def single_rank(rank, device, cfg, steps, batch, micro, deterministic, data, out):
    dev = setup(device, 0, deterministic)
    torch.manual_seed(0)
    text = corpus(data)
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=False)
    losses, times = [], []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for step in range(steps):
        x, y = text.batch(step, batch, cfg.seq, device=dev)
        sync(dev)
        t0 = time.perf_counter()
        total = 0
        for xb, yb in zip(x.chunk(micro), y.chunk(micro)):
            loss = loss_fn(model(xb), yb) / micro
            loss.backward()
            total = total + loss.detach()
        opt.step()
        opt.zero_grad()
        sync(dev)
        times.append(time.perf_counter() - t0)
        losses.append(total.item())
    torch.save({"losses": losses, "times": times, "weights": flat_weights(model.parameters()),
                "peak_mem": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None}, out)


def run_single(device, cfg, steps, batch, micro, deterministic=False, data="synthetic"):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "single.pt")
        mp.spawn(single_rank, args=(device, cfg, steps, batch, micro, deterministic, data, out), nprocs=1)
        return torch.load(out, weights_only=False)


# ---------------------------------------------------------------- sections
def compare(name, ours, ref, ref_name):
    a, b = ours[0], ref[0]
    emit({"section": "verify", "strategy": name, "against": ref_name, "losses_equal": a["losses"] == b["losses"],
          "weights_equal": torch.equal(a["weights"], b["weights"]),
          "max_weight_diff": float((a["weights"] - b["weights"]).abs().max()), "steps": len(b["losses"]),
          "final_loss": a["losses"][-1]})


def verify(device):
    # cuBLAS reads this when it starts, in each child process.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cfg = Config(seq=128, layers=4, heads=4, dim=256)
    kw = dict(cfg=cfg, steps=20, batch=16, deterministic=True, want_weights=True)
    ref = run_torch("torch-ddp", device, **kw)
    emit({"section": "verify", "model_params": count_params(GPT(cfg)), "device": device,
          "torch": torch.__version__})
    try:
        compare("torch-fsdp", run_torch("torch-fsdp", device, **kw), ref, "torch-ddp")
    except Exception as e:  # a baseline failing should not hide tandem's results
        emit({"section": "verify", "strategy": "torch-fsdp", "error": repr(e)[:300]})
    for s in ("ddp", "zero1", "zero2", "zero3"):
        compare(s, run_tandem(s, device, **kw), ref, "torch-ddp")
    single = run_single(device, cfg, 20, 16, 4, deterministic=True)
    for s in ("pp-gpipe", "pp-1f1b"):
        ours = run_tandem(s, device, micro=4, **kw)
        n0 = ours[0]["weights"].numel()
        both = torch.cat([ours[0]["weights"], ours[1]["weights"]])
        emit({"section": "verify", "strategy": s, "against": "single-process accumulation",
              "losses_equal": ours[1]["losses"] == single["losses"],
              "weights_equal": torch.equal(both, single["weights"]),
              "max_weight_diff": float((both - single["weights"]).abs().max()), "stage0_params": n0})


def ar_rank(g, sizes, reps):
    dev = setup(g.device.type, g.rank, False)
    out = {}
    for mb in sizes:
        x = torch.ones(mb * (1 << 20) // 4, device=dev)
        g.all_reduce(x)
        sync(dev)
        ts = []
        for _ in range(reps):
            g.barrier()
            t = time.perf_counter()
            g.all_reduce(x)
            sync(dev)
            ts.append(time.perf_counter() - t)
        out[mb] = statistics.median(ts)
    return out


def ar_torch(rank, init, out, device, sizes, reps):
    import torch.distributed as dist

    dev = setup(device, rank, False)
    dist.init_process_group("nccl" if device == "cuda" else "gloo", init_method=f"file://{init}", rank=rank, world_size=2)
    res = {}
    for mb in sizes:
        x = torch.ones(mb * (1 << 20) // 4, device=dev)
        dist.all_reduce(x)
        sync(dev)
        ts = []
        for _ in range(reps):
            dist.barrier()
            t = time.perf_counter()
            dist.all_reduce(x)
            sync(dev)
            ts.append(time.perf_counter() - t)
        res[mb] = statistics.median(ts)
    torch.save(res, f"{out}.{rank}")
    dist.destroy_process_group()


def allreduce(device):
    sizes, reps = [1, 4, 16, 64, 256], 10
    ours = launch(ar_rank, 2, sizes, reps, device=device)[0]
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(ar_torch, args=(os.path.join(d, "init"), os.path.join(d, "o"), device, sizes, reps), nprocs=2)
        ref = torch.load(os.path.join(d, "o.0"))
    base = "nccl" if device == "cuda" else "gloo"
    for mb in sizes:
        emit({"section": "allreduce", "MB": mb, "tandem_ms": ours[mb] * 1e3, f"{base}_ms": ref[mb] * 1e3,
              "tandem_GBps": mb / 1024 / ours[mb], f"{base}_GBps": mb / 1024 / ref[mb]})


def bench(device, preset, steps, batch):
    cfg = Config.preset(preset)
    tokens = batch * cfg.seq
    emit({"section": "bench", "model": preset, "params": count_params(GPT(cfg)), "global_batch": batch,
          "seq": cfg.seq, "device": device})

    def report(name, res, per_rank_batch=True):
        med = statistics.median(res[0]["times"][2:]) if len(res[0]["times"]) > 2 else res[0]["times"][-1]
        row = {"section": "bench", "strategy": name, "step_ms": med * 1e3, "tokens_per_s": tokens / med,
               "peak_mem_MB": [None if r.get("peak_mem") is None else round(r["peak_mem"] / 2**20) for r in res],
               # data parallel: the mean of the ranks' losses on their halves of the batch
               "final_loss": statistics.mean(r["losses"][-1] for r in res if r["losses"][-1] is not None)}
        if "profile" in res[0]:
            row["profile_rank0"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[0]["profile"].items()}
        emit(row)

    single = run_single(device, cfg, steps, batch, 1, data="shakespeare")
    report("single-gpu", [single])
    for s in ("torch-ddp", "torch-fsdp"):
        try:
            report(s, run_torch(s, device, cfg, steps, batch, data="shakespeare"))
        except Exception as e:
            emit({"section": "bench", "strategy": s, "error": repr(e)[:300]})
    for s in ("ddp", "ddp-nooverlap", "zero1", "zero2", "zero3", "pp-gpipe", "pp-1f1b"):
        try:
            report(s, run_tandem(s, device, cfg=cfg, steps=steps, batch=batch, micro=4, data="shakespeare",
                                 profile=s in ("ddp", "ddp-nooverlap", "zero3")))
        except Exception as e:
            emit({"section": "bench", "strategy": s, "error": repr(e)[:300]})


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["verify", "allreduce", "bench", "all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results.jsonl")
    a = ap.parse_args()
    OUT = a.out
    if a.device == "cuda" and torch.cuda.device_count() < 2:
        sys.exit(f"needs 2 GPUs, found {torch.cuda.device_count()} (Kaggle: Accelerator 'GPU T4 x2')")
    if a.what in ("verify", "all"):
        verify(a.device)
    if a.what in ("allreduce", "all"):
        allreduce(a.device)
    if a.what in ("bench", "all"):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        bench(a.device, a.model, a.steps, a.batch)


if __name__ == "__main__":
    main()
