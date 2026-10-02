# The C++ engine

tandem's collectives are written in Python today (`tandem/comm.py`). On two T4s they
reach NCCL's bandwidth for large messages, but they lose where per-piece overhead
dominates:

| measured on 2x T4 | Python engine | NCCL |
|---|---|---|
| 1 MB all-reduce | 1.07 GB/s (45%) | 2.38 GB/s |
| 4 MB all-reduce | 2.47 GB/s (74%) | 3.33 GB/s |
| DDP's all-reduces, alone | 96 ms per step | |
| the same, during backward | 517 ms per step | |

Every 4 MB piece costs a Python call, two semaphore operations and a host-side stream
synchronization. The C++ engine moves that loop out of Python and off the host:

1. **Shared-memory channels**: single-producer, single-consumer rings whose head and
   tail are `std::atomic` counters in shared memory, instead of OS semaphores.
2. **A progress thread in C++** that runs collectives without holding Python's lock.
3. **A fused copy-and-add CUDA kernel** for the reduce step, instead of a copy to
   scratch followed by `add_`.
4. **CUDA IPC events** so that a GPU waits for its peer's copy on the GPU itself
   (`cudaStreamWaitEvent`), not through a host-side `synchronize()`.

`Group(backend="cpp")` selects the C++ engine; the Python engine stays as the reference,
and every test runs against both.

## Milestones

Each milestone has a spec (the header, written for you), tests that define "done",
and the C++ you will learn by writing it. Implement the bodies in `tandem/csrc/*.cpp`;
until you do, their tests are skipped with "not implemented".

### M0: shared memory with RAII (2-3 days)

`tandem/csrc/shm.h`: `SharedMemory` owns a POSIX shared-memory mapping
(`shm_open`, `ftruncate`, `mmap`; `munmap`, `close`, `shm_unlink` when done).

- Learn: classes, constructors and destructors, RAII, move-only types (deleted copy,
  `noexcept` move, the moved-from state), exceptions carrying `errno` (`std::system_error`).
- Done when: `pytest tests/test_engine.py -k m0` passes. Two processes see each other's
  writes; the creator's destructor unlinks the name; errors name the failing call.

### M1: a lock-free channel (4-5 days)

`tandem/csrc/channel.h`: `Channel`, a single-producer, single-consumer ring of
`slots` buffers of `slot_bytes` each, in one `SharedMemory` region.

- Learn: `std::atomic`, the C++ memory model (`memory_order_acquire` / `release`, and
  why `relaxed` is not enough), false sharing and `alignas(64)`, spin-then-sleep waiting,
  `static_assert` on `is_always_lock_free` (atomics shared between processes must be
  lock-free).
- Done when: `pytest tests/test_engine.py -k m1` passes: 20,000 messages of varying
  length cross between processes through 4 tiny slots in order and intact, full and
  empty rings block, and timeouts raise.
- Stretch: replace the sleep with a futex wait and wake (`syscall(SYS_futex, ...)`).

### M2: the progress thread and CPU collectives (5-6 days)

A C++ `Engine` with a job queue (`std::thread`, `std::mutex`,
`std::condition_variable`) and ring reduce-scatter, all-gather, all-reduce, broadcast
and send/recv over `Channel`s, on `at::Tensor` (CPU).

- Learn: threads and their lifetime, condition variables (and spurious wake-ups),
  `std::promise` / `std::future` or your own Work handle, `at::Tensor`'s C++ API,
  releasing the GIL (`py::gil_scoped_release`), exceptions crossing threads.
- Done when: the whole collective suite (`tests/test_comm.py`) and the bit-for-bit
  checks (`bench/run.py verify --device cpu`) pass with `TANDEM_BACKEND=cpp`, and the
  CPU all-reduce benchmark shows the change against the Python engine.

### M3: CUDA (4-5 days, needs a GPU)

- A fused `dst += src` kernel (vectorized with `float4`), launched on the engine's stream.
- Slots in GPU memory shared with `cudaIpcGetMemHandle`; one interprocess event per slot
  (`cudaEventInterprocess | cudaEventDisableTiming`) so the consumer's stream waits for
  the producer's copy with `cudaStreamWaitEvent`, and the host never synchronizes per piece.
- Learn: CUDA kernels and launch configuration, streams and events, CUDA IPC, error
  checking (`C10_CUDA_CHECK`), building `.cu` files in an extension.
- Done when: the CUDA tests pass on one GPU (Colab) for the kernel and on two GPUs
  (Kaggle or Modal) for IPC, and `bench/run.py verify` is still bit-for-bit.

### M4: measure and write up (2-3 days)

`bench/run.py` gains `--backend cpp`: all-reduce by size against the Python engine and
NCCL, the contention test, and DDP and ZeRO-3 step times. The README gets the
before/after table, including whatever did not improve.

## How we work

- You write the engine (`tandem/csrc/*.cpp`, and `*.cu` in M3). I write the specs,
  bindings, build, tests and benchmarks, and review every diff before it is pushed.
- Ask for an explanation whenever a concept is new; that is the point of the project.
- Build and test: `PYTHONPATH=.:tests python -m pytest -q tests/test_engine.py`.
  The first run compiles the extension (about a minute); later runs reuse it.
