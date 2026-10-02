# tandem

Distributed training written from scratch on PyTorch tensors and autograd,
with its own collectives: no `torch.distributed`, no NCCL.

| | What it is | Checked against |
|---|---|---|
| **Collectives** | ring all-reduce, reduce-scatter and all-gather, pipelined broadcast, send/recv; shared-memory mailboxes between processes, CUDA IPC between GPUs; asynchronous, on a communication thread and stream | NCCL (speed); exact results in tests at 2, 3 and 4 ranks |
| **DDP** | gradient buckets filled as backward produces them, each all-reduced as soon as it is full; buckets rebuilt in the order gradients really arrive; `no_sync` for accumulation | `torch.nn.parallel.DistributedDataParallel`, bit for bit |
| **ZeRO-1 / 2** | AdamW state sharded across ranks (stage 1); gradients reduce-scattered and sharded too (stage 2) | torch DDP, bit for bit |
| **ZeRO-3 (FSDP)** | parameters sharded per unit, all-gathered just before use and freed after, with the next unit's gather prefetched | torch DDP and torch FSDP, bit for bit |
| **Pipelines** | GPipe and 1F1B over 2-4 stages, in Megatron-LM's send/receive order | one process accumulating over the same micro-batches, bit for bit |
| **Profiler** | per-rank compute and communication spans, the time the training thread spent blocked on communication, Chrome traces | |

## Results on 2x T4 (Kaggle)

Raw output: [`bench/t4/2026-10-02.jsonl`](bench/t4/2026-10-02.jsonl) (torch 2.10.0+cu128, commit
`9e2b4ef`). A run on the previous commit gave the same verification results, and timings
within 7%.

**Correctness.** A 3.3M-parameter GPT trains for 20 steps with deterministic kernels. Every
tandem strategy ends with **exactly** the losses and weights of its reference (largest weight
difference: 0.0).

| tandem | reference | losses | weights |
|---|---|---|---|
| DDP | torch DDP (NCCL) | equal | equal |
| ZeRO-1 | torch DDP (NCCL) | equal | equal |
| ZeRO-2 | torch DDP (NCCL) | equal | equal |
| ZeRO-3 | torch DDP (NCCL), which torch FSDP also matches exactly | equal | equal |
| GPipe, 2 stages | single-process accumulation | equal | equal |
| 1F1B, 2 stages | single-process accumulation | equal | equal |

Exact equality is possible because tandem averages the way torch does (divide each gradient,
then sum), so both round the same way.

**All-reduce bandwidth** (float32, median of 10):

| size | tandem | NCCL | tandem / NCCL |
|---|---|---|---|
| 1 MB | 1.07 GB/s | 2.38 GB/s | 45% |
| 4 MB | 2.47 GB/s | 3.33 GB/s | 74% |
| 16 MB | 3.43 GB/s | 3.61 GB/s | 95% |
| 64 MB | 3.65 GB/s | 3.66 GB/s | 100% |
| 256 MB | 3.72 GB/s | 3.68 GB/s | 101% |

Large messages run at NCCL's speed. Small ones pay a fixed cost per piece (a semaphore
handshake and a stream synchronization for each 4 MB slot) that NCCL avoids.

**Training an 85.8M-parameter GPT** (12 layers, width 768, sequence 512, global batch 16,
Tiny Shakespeare): median step time after warm-up, and peak memory per GPU.

| strategy | step | tokens/s | peak memory per GPU |
|---|---|---|---|
| 1 GPU | 1280 ms | 6,400 | 5,764 MB |
| torch DDP | 684 ms | 11,984 | 3,740 MB |
| torch FSDP | 700 ms | 11,698 | 2,980 / 2,963 MB |
| tandem DDP | 798 ms | 10,265 | 3,775 MB |
| tandem DDP, no overlap | 769 ms | 10,649 | 3,775 MB |
| tandem ZeRO-1 | 853 ms | 9,601 | 3,410 MB |
| tandem ZeRO-2 | 802 ms | 10,218 | 3,268 MB |
| tandem ZeRO-3 | 792 ms | 10,341 | 3,144 MB |
| tandem GPipe | 930 ms | 8,809 | 3,069 / 3,085 MB |
| tandem 1F1B | 911 ms | 8,990 | 2,075 / 1,489 MB |

What the table shows:

- **Memory falls where the designs say it should.** Each ZeRO stage holds less than the one
  before; ZeRO-3 is 17% below DDP. 1F1B holds 32% and 52% less than GPipe on its two
  stages, because a stage keeps at most (stages - rank) micro-batches of activations, not all
  of them.
- **Speed: tandem DDP reaches 86% of torch DDP's throughput, and its overlap does not pay off
  on GPUs yet.** On their own, DDP's all-reduces take 96 ms per step. Started during
  backward, the same all-reduces take 517 ms: each piece's copy and add wait behind the
  backward pass's kernels. So DDP with overlap is slower than without it. (On CPU processes
  the same code hides 57% of its communication.) The next step measures whether a
  high-priority stream, or staging through pinned host memory as NCCL's shared-memory
  transport does, keeps communication at full speed beside compute.

## The C++ engine

`Group(backend="cpp")` runs the same collectives in a C++ extension instead of Python
([design](docs/engine.md), [explained line by line](docs/engine-walkthrough.md)):
lock-free single-producer, single-consumer rings in shared memory, a progress thread, and
on GPUs CUDA IPC buffers ordered by interprocess events, with a fused reduce kernel, so
the host never waits for a piece. Results are bit-identical to the Python engine's: the
whole test suite and the bit-for-bit training checks pass on it (CPU in CI; on a T4,
`tests/test_engine_cuda.py`).

All-reduce of float32 on one Tesla T4 (Colab, torch 2.11), two ranks sharing the GPU
through CUDA IPC, median of 20 ([raw](bench/t4/engine-one-gpu-2026-10-02.jsonl),
`python bench/engine_one_gpu.py`):

| size | Python engine | C++ engine | speedup |
|---|---|---|---|
| 64 KB | 1.79 ms | 0.64 ms | 2.8× |
| 256 KB | 1.49 ms | 0.80 ms | 1.9× |
| 1 MB | 1.53 ms | 0.83 ms | 1.9× |
| 4 MB | 1.65 ms | 0.75 ms | 2.2× |
| 16 MB | 3.27 ms | 1.72 ms | 1.9× |
| 64 MB | 10.00 ms | 6.24 ms | 1.6× |

Two ranks on one GPU measure the engines' own overhead, not the link between GPUs (NCCL
cannot run two ranks on one GPU, so it is not in this table). On two GPUs,
`python bench/run.py engine` compares all three; those numbers are still to be measured.
On CPU processes the C++ engine is 1.6 to 2.2× faster at 1 MB and level from 4 MB, where
memory bandwidth limits both.

## Run it

```sh
pip install torch pytest numpy
PYTHONPATH=.:tests python -m pytest -q tests     # on CPU processes
python bench/run.py all                          # on a machine with 2 GPUs, e.g. Kaggle "GPU T4 x2"
```

`python bench/run.py verify --device cpu` runs the bit-for-bit checks on CPU processes (with
gloo as torch's backend); CI runs it on every push.

## Layout

```
tandem/comm.py      mailboxes, ring collectives, the communication thread, launch()
tandem/ddp.py       DDP
tandem/zero.py      ZeRO-1/2 and FSDP (ZeRO-3)
tandem/optim.py     the sharded AdamW (torch's AdamW, elementwise, on a shard)
tandem/pipeline.py  GPipe and 1F1B
tandem/profile.py   timelines, exposed communication, Chrome traces
tandem/model.py     the GPT used in tests and benchmarks
bench/run.py        verification and benchmarks against torch.distributed
tandem/csrc/        the C++ engine (Group(backend="cpp")): docs/engine.md, docs/engine-walkthrough.md
tandem/engine.py    builds and loads it
```
