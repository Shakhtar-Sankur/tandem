# The C++ engine, explained

A guided reading of `tandem/csrc/`, in the order the pieces build on each other.
Read each section next to its file. At the end of each section are the questions
an interviewer is likely to ask, with the answers you should be able to give in
your own words.

## The big picture

tandem trains one model on several processes (one per GPU, or several CPU processes).
They must constantly exchange data: averaging gradients after every backward pass
(all-reduce), gathering sharded parameters (all-gather), passing activations between
pipeline stages (send/recv). `tandem/comm.py` does this in Python. The C++ engine does
the same work in C++, with three layers:

```
shm.cpp          SharedMemory   a block of memory two processes can both see
channel.cpp      Channel        a one-way message queue inside such a block
collectives.cpp  Engine         a thread that runs all-reduce, all-gather, ... over channels
engine.cpp       bindings       makes the three visible to Python (pybind11)
```

`Group(backend="cpp")` in `comm.py` uses the Engine; everything above it (DDP, ZeRO,
pipelines) does not know or care which engine runs underneath. The proof that the C++
engine is right: the whole test suite and the bit-for-bit training checks pass with
`TANDEM_BACKEND=cpp`, and give results identical to the Python engine's to the last bit.

## 1. SharedMemory (`shm.h`, `shm.cpp`): RAII

**What it is.** Normally each process has its own private memory. POSIX shared memory
gives a block of memory a name (like a file path, `/tandem-...`); any process that opens
the name and maps it sees the same bytes. Four system calls:

- `shm_open(name, flags, mode)` creates or opens the named object, returning a file
  descriptor (an integer handle).
- `ftruncate(fd, bytes)` sets its size; new memory reads as zeros.
- `mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)` maps it into this
  process's address space and returns a pointer to it.
- Clean-up: `munmap` (unmap), `close` (the descriptor), `shm_unlink` (remove the name).

**RAII** (Resource Acquisition Is Initialization): the object that acquires a resource
owns it, and its destructor frees it. C++ runs destructors automatically when an object
goes out of scope, including when an exception is thrown, so cleanup can never be
forgotten. `~SharedMemory()` calls `release()`, which unmaps, closes, and, if this
object created the region (`owner_`), unlinks the name.

**Why copying is deleted.** If two `SharedMemory` objects held the same pointer, both
destructors would `munmap` it: a double free. `= delete` on the copy constructor and copy
assignment makes the compiler reject any copy. `engine.cpp` checks this with
`static_assert(!std::is_copy_constructible_v<SharedMemory>)`.

**Moving.** Ownership can still be handed over. The move constructor takes the other
object's fields and leaves it empty (`data_ = nullptr`, `fd_ = -1`, `owner_ = false`), so
the emptied object's destructor does nothing. `std::exchange(x, v)` returns `x`'s old
value and sets it to `v` in one step, which is exactly "take it and leave it empty".
Move assignment first releases what the target already owns, and checks
`this != &other` so that `a = std::move(a)` does not destroy `a`'s own region.

**`noexcept`.** A promise that moving never throws. It matters: `std::vector` only moves
elements during reallocation if their move is `noexcept` (otherwise it copies, which here
would not compile). The `static_assert`s check it.

**Errors.** System calls report failure by returning -1 (or `MAP_FAILED`) and setting the
global `errno`. `std::system_error(errno, std::generic_category(), "shm_open " + name)`
turns that into an exception whose message is like
`shm_open /tandem-x: File exists`. `os_error` captures `errno` *before* any clean-up call,
because `close` or `shm_unlink` could overwrite it. If `ftruncate` or `mmap` fails,
`create` undoes the steps that already succeeded (close, unlink), so a failed `create`
leaves nothing behind.

**Interview questions**
- *What is RAII and why use it?* Tie a resource's lifetime to an object; the destructor
  frees it, so it is freed exactly once, on every path, including exceptions.
- *What is the rule of five?* If a class manages a resource, decide all five of:
  destructor, copy constructor, copy assignment, move constructor, move assignment.
  Here: destructor frees; copies deleted; moves transfer ownership.
- *What state is a moved-from object in?* Valid but empty: safe to destroy or assign to.
- *Why check self-assignment in move assignment?* `release()` would free the region
  before reading it back from `other`, which is the same object.
- *Why is the name unlinked by the creator only?* The name is shared; only one process
  should remove it. Unlinking does not unmap: processes that already mapped the region
  keep it until they unmap (a test checks this).

## 2. Channel (`channel.h`, `channel.cpp`): lock-free, single producer, single consumer

**What it is.** A one-way queue of messages from one process to another, living inside
one `SharedMemory` block: a header, then `slots` fixed-size slots. The sender writes a
message into the next free slot; the receiver reads the oldest unread one.

**Two counters.** `head` counts messages ever sent (only the sender writes it); `tail`
counts messages ever received (only the receiver writes it). Message `k` lives in slot
`k % slots`. The queue is empty when `head == tail` and full when `head - tail == slots`.
Because they only grow (64 bits never overflow in practice), full and empty can never be
confused, which is the classic bug of ring buffers that store wrapped indices.

**Why atomics.** The two processes run truly in parallel on different cores. Ordinary
variables give no guarantee about when, or in what order, one core's writes become
visible to another; compilers and CPUs both reorder memory operations. `std::atomic`
makes each load and store indivisible and lets us state the ordering we need.

**Acquire and release (the heart of it).** The sender does:
1. copy the payload into the slot (ordinary writes);
2. `head.store(head + 1, std::memory_order_release)`.

The receiver does:
1. `head.load(std::memory_order_acquire)` and sees the new value;
2. read the payload.

*Release* means: every write before this store is visible to whoever *acquires* this
value. So once the receiver sees the new `head`, the payload is guaranteed to be there.
Without it (`relaxed`), the receiver could see the new `head` and still read an old,
half-written payload. The same pairing runs the other way for `tail`: the receiver
releases the slot only after finishing its reads, so the sender cannot overwrite a slot
still being read. A process reading *its own* counter uses `relaxed`, since nobody else
writes it.

**Lock-free and in shared memory.** A `std::atomic` that is not lock-free is implemented
with a hidden lock that lives in one process's private memory, which the other process
cannot see. `static_assert(std::atomic<std::uint64_t>::is_always_lock_free)` guarantees
real hardware atomic instructions instead.

**False sharing.** CPUs move memory between cores in 64-byte cache lines. If `head` and
`tail` sat in the same line, every write by one side would invalidate the other side's
cached copy, and the line would bounce between cores. `alignas(64)` gives each its own
line.

**Placement new.** `new (shm.data()) Header` constructs a `Header` object (including its
atomics) in memory we already have, instead of allocating new memory.

**Waiting.** When the queue is full (or empty), the waiting side calls `wait_until`.
Inside a collective, the other side usually answers within microseconds, so it first
spins (checks again immediately), then yields its core to other threads between checks;
only after a millisecond with no progress (an idle peer) does it sleep 10 µs at a time.
Every 64 rounds it checks the timeout and throws `Timeout` past it. The first version
slept almost immediately, and that cost a measurable amount of latency per message.

**Zero-copy receive (`peek` / `release`).** `recv` copies the message out. For a
reduction, the engine instead `peek`s, which returns a pointer to the message inside the
slot; it adds directly from there into the result, then calls `release()` to free the
slot. That removed one full copy of every byte reduced, the main reason the first C++
version was slower than Python.

**Interview questions**
- *What does memory_order_release / acquire guarantee?* Writes before a release store
  are visible to a thread that acquire-loads that stored value. That is how the payload
  is "published".
- *Why is SPSC simpler than MPMC?* Each counter has exactly one writer, so plain stores
  suffice; no compare-and-swap loops, no ABA problem.
- *What is false sharing?* Two independent variables on one cache line make cores
  invalidate each other's caches on every write.
- *Why spin before sleeping?* Sleeping costs a system call and wake-up latency
  (tens of microseconds); when the answer is microseconds away, spinning is cheaper.
- *How would you avoid spinning entirely?* A futex: sleep in the kernel on the
  counter's address and have the other side wake you, as the Python engine's semaphores
  do internally. That is the remaining stretch goal.

## 3. Engine (`collectives.h`, `collectives.cpp`): the progress thread and the algorithms

**Channels between every pair.** Rank `r` creates a channel to every other rank `d`, named
`/tandem-<job>-r-d`, and after a barrier (so all exist) opens the one every other rank
created to it. `out_[d]` sends to `d`; `in_[s]` receives from `s`.

**The progress thread.** Collectives must run in the order they were requested, on every
rank, and without blocking the training loop. So `submit()` puts a job (a `std::function`)
on a queue and returns a `Work`; one `std::thread` runs `loop()`, taking jobs in order.
The queue is protected by a `std::mutex`; the thread sleeps on a
`std::condition_variable` until a job arrives. The wait takes a predicate
(`!jobs_.empty() || stopping_`), because condition variables can wake up spuriously and
the predicate is re-checked. `close()` sets `stopping_`, wakes the thread, and joins it;
the loop only exits once the queue is empty, so queued collectives finish first.

**Work.** A small object the caller waits on: `done_` behind a mutex, a condition
variable, the start and end times (for the profiler), and an `std::exception_ptr`. If a
collective throws on the progress thread, `loop()` catches it with
`std::current_exception()` and stores it; `wait()` rethrows it in the caller's thread with
`std::rethrow_exception`. That is how errors cross threads in C++.

**Ownership across threads.** Jobs capture `at::Tensor` handles by value. A Tensor is a
reference-counted handle to its storage (the count is atomic), so the tensor's memory
stays alive until the job finishes, even if the caller drops its own reference.

**The GIL.** The progress thread never calls into Python, so it never needs Python's
global lock. `Work.wait` is bound with `py::call_guard<py::gil_scoped_release>`: while a
Python thread waits for a collective, it releases the GIL so other Python threads (and
autograd's hooks) keep running.

**The algorithms.** The same as `comm.py`, step for step:
- *Ring reduce-scatter.* The tensor is cut into one chunk per rank. In `n - 1` steps, each
  rank sends one chunk to its right neighbour and adds the chunk arriving from its left
  into its own copy. Afterwards rank `r` holds the full sum of chunk `r`.
- *Ring all-gather.* The same ring, copying instead of adding: after `n - 1` steps every
  rank has every finished chunk.
- *All-reduce* = reduce-scatter + all-gather. Each rank sends and receives about
  `2 (n-1)/n` of the tensor, independent of `n`, which is why the ring is bandwidth-optimal.
- *exchange()* sends and receives piece by piece in lockstep (send piece k, receive
  piece k). If every rank first sent everything, all would block on full channels and
  none would read: a deadlock. Lockstep needs only one free slot.
- *Average.* Each rank divides by `n` *before* the sum, as PyTorch DDP does. Combined with
  using the same ATen calls (`div_`, `add_`) in the same order, results are bit-identical
  to PyTorch's.
- *Barrier.* Every rank sends an empty message to every other and waits for one from
  each. Channels are first in, first out, so a barrier also waits for every collective
  queued before it.

**Interview questions**
- *Why is ring all-reduce bandwidth-optimal?* Each rank transfers `2(n-1)/n × size`,
  which is the lower bound; it does not grow with the number of ranks.
- *Why might a collective deadlock, and how does lockstep prevent it?* Everyone sending
  before receiving fills every buffer; alternating send and receive per piece frees a
  slot each step.
- *How does an exception get from one thread to another?* `std::exception_ptr`, captured
  with `current_exception()`, rethrown with `rethrow_exception()`.
- *Why does the condition-variable wait need a predicate?* Spurious wake-ups, and a
  notify that happens before the wait starts.
- *How do you make results bit-identical to PyTorch?* Same operations in the same order
  on the same dtypes: floating-point addition is not associative, so order matters.

## 4. The bindings (`engine.cpp`) and the build (`tandem/engine.py`)

`PYBIND11_MODULE` declares the Python module; `py::class_` exposes each C++ class and
`.def` each method. `py::register_exception` maps C++ exceptions to Python ones
(`Timeout` → `TimeoutError`; the default turns `std::invalid_argument` into `ValueError`
and other `std::exception`s into `RuntimeError`). `tandem/engine.py` compiles everything
with `torch.utils.cpp_extension.load` the first time it is used (about a minute) and caches
the result; `launch()` builds it once before starting the ranks, so they do not all
compile at the same time.

## Measured (CPU processes, 2 ranks, 3 runs on a shared 4-core machine)

| all-reduce | Python engine | C++ engine |
|---|---|---|
| 1 MB | 0.66 to 0.96 ms | 0.36 to 0.60 ms (1.6 to 2.2× faster) |
| 4 MB and up | | within run-to-run noise of Python: memory bandwidth limits both |

On CPU, large messages are limited by memory bandwidth whichever engine moves them; the
C++ engine's gain is in per-message overhead, which is what dominates small messages, and
what limited the Python engine on GPUs. M3 brings the engine to CUDA, where that overhead
was measured.
