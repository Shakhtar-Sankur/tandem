// M2: the progress thread and the collectives. M3: the same on GPUs (built
// with TANDEM_CUDA), moving pieces through CudaLinks instead of host memory.
//
// An Engine is one rank's half of the group. It owns a Channel to every other
// rank (created by this rank, named after the job and the pair) and opens the
// Channel every other rank created to it. Collectives are queued and run, in
// submission order, on one C++ thread that never touches Python, so a caller
// can keep computing while gradients travel and wait on the returned Work.
//
// The algorithms are the Python engine's (tandem/comm.py), step for step, and
// the arithmetic is the same ATen calls (div_ before the sum, then add_ of each
// received piece), so the two engines produce bit-identical results.
//
// On GPUs a collective is ordered by streams, as NCCL's are: it starts on the
// engine's stream once the caller's current stream reaches the point where it
// was submitted, and Work::wait() makes the caller's current stream wait for
// it to finish, without blocking the host.
#pragma once

#include <ATen/ATen.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "channel.h"
#ifdef TANDEM_CUDA
#include <cuda_runtime_api.h>

#include "cuda_links.h"
#endif

namespace tandem {

// A collective in flight.
class Work {
 public:
  ~Work();
  // Blocks until the collective has run (on GPUs: has been enqueued), then, on
  // GPUs, makes the caller's current stream wait for it. Rethrows its error.
  void wait();
  bool done() const;
  double started() const;   // seconds on the steady clock (CLOCK_MONOTONIC,
  double finished() const;  // the clock of Python's time.perf_counter)

 private:
  friend class Engine;
  void finish(std::exception_ptr error, double t0, double t1);

#ifdef TANDEM_CUDA
  cudaEvent_t done_event_ = nullptr;  // recorded on the engine's stream after the collective
  int device_ = -1;
#endif
  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  std::exception_ptr error_;
  double t0_ = 0, t1_ = 0;
};

class Engine {
 public:
  // Creates this rank's outgoing channels (and, with device >= 0, its GPU
  // slots and events). Call connect() once every rank has constructed its Engine.
  Engine(const std::string& job, int rank, int size, std::size_t slots, std::size_t slot_bytes, double timeout_s,
         int device = -1);
  void connect();  // opens the channels the other ranks created to this one
  void close();    // finishes the queued collectives and stops the thread
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Tensors must be contiguous, on the engine's device; results are written in place.
  // sizes: how to cut the flat tensor into one chunk per rank (empty: as even
  // as possible, as torch.tensor_split does).
  std::shared_ptr<Work> all_reduce(at::Tensor t, bool average);
  std::shared_ptr<Work> reduce_scatter(at::Tensor t, std::vector<int64_t> sizes, bool average);
  std::shared_ptr<Work> all_gather(at::Tensor t, std::vector<int64_t> sizes);
  std::shared_ptr<Work> broadcast(at::Tensor t, int root);
  std::shared_ptr<Work> send(at::Tensor t, int dst);
  std::shared_ptr<Work> recv(at::Tensor t, int src);
  std::shared_ptr<Work> sendrecv(at::Tensor send, int dst, at::Tensor recv, int src);
  std::shared_ptr<Work> barrier();

  int rank() const { return rank_; }
  int size() const { return size_; }

 private:
  std::shared_ptr<Work> submit(std::function<void()> job, std::vector<at::Tensor> tensors = {});
  void loop();
  void barrier_now();
  void check(const at::Tensor& t, const char* what) const;

  void put(int dst, const at::Tensor& piece);
  void take(int src, at::Tensor& piece, bool reduce);
  void exchange(const at::Tensor& send, int dst, const at::Tensor& recv, int src, bool reduce);
  void ring_reduce_scatter(const std::vector<at::Tensor>& chunks);
  void ring_all_gather(const std::vector<at::Tensor>& chunks);
  std::vector<at::Tensor> chunks(const at::Tensor& flat, const std::vector<int64_t>& sizes) const;
  int64_t piece_elems(const at::Tensor& t) const;
  std::string channel_name(int src, int dst) const;

  std::string job_;
  int rank_, size_;
  std::size_t slots_, slot_bytes_;
  double timeout_s_;
  int device_;  // -1: CPU
  std::vector<Channel> out_, in_;  // indexed by peer rank; empty at our own rank

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::pair<std::function<void()>, std::shared_ptr<Work>>> jobs_;
  bool stopping_ = false;
#ifdef TANDEM_CUDA
  std::unique_ptr<CudaLinks> cuda_;
#endif
  std::thread thread_;
};

}  // namespace tandem
