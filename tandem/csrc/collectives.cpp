// M2: Engine and Work (see collectives.h).

#include "collectives.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace tandem {

namespace {

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int mod(int a, int n) { return ((a % n) + n) % n; }

void check(const at::Tensor& t, const char* what) {
  if (!t.defined()) throw std::invalid_argument(std::string(what) + ": undefined tensor");
  if (!t.device().is_cpu()) throw std::invalid_argument(std::string(what) + ": the C++ engine runs on CPU tensors");
  if (!t.is_contiguous()) throw std::invalid_argument(std::string(what) + ": collectives need contiguous tensors");
}

}  // namespace

// ---------------------------------------------------------------- Work
void Work::wait() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this] { return done_; });  // the predicate guards against spurious wake-ups
  if (error_) std::rethrow_exception(error_);
}

bool Work::done() const {
  std::lock_guard<std::mutex> lock(mu_);
  return done_;
}

double Work::started() const {
  std::lock_guard<std::mutex> lock(mu_);
  return t0_;
}

double Work::finished() const {
  std::lock_guard<std::mutex> lock(mu_);
  return t1_;
}

void Work::finish(std::exception_ptr error, double t0, double t1) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    error_ = std::move(error);
    t0_ = t0;
    t1_ = t1;
    done_ = true;
  }
  cv_.notify_all();
}

// ---------------------------------------------------------------- Engine: setup
Engine::Engine(const std::string& job, int rank, int size, std::size_t slots, std::size_t slot_bytes,
               double timeout_s)
    : job_(job), rank_(rank), size_(size), slots_(slots), slot_bytes_(slot_bytes), timeout_s_(timeout_s),
      out_(size), in_(size) {
  if (size < 1 || rank < 0 || rank >= size) throw std::invalid_argument("rank must be in [0, size)");
  for (int dst = 0; dst < size_; ++dst) {
    if (dst != rank_) out_[dst] = Channel::create(channel_name(rank_, dst), slots_, slot_bytes_);
  }
  thread_ = std::thread([this] { loop(); });
}

void Engine::connect() {
  for (int src = 0; src < size_; ++src) {
    if (src != rank_) in_[src] = Channel::open(channel_name(src, rank_));
  }
}

std::string Engine::channel_name(int src, int dst) const {
  return "/tandem-" + job_ + "-" + std::to_string(src) + "-" + std::to_string(dst);
}

void Engine::close() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stopping_) return;
    stopping_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

Engine::~Engine() { close(); }

// ---------------------------------------------------------------- the progress thread
std::shared_ptr<Work> Engine::submit(std::function<void()> job) {
  auto work = std::make_shared<Work>();
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stopping_) throw std::runtime_error("the engine is closed");
    jobs_.emplace_back(std::move(job), work);
  }
  cv_.notify_one();
  return work;
}

void Engine::loop() {
  for (;;) {
    std::pair<std::function<void()>, std::shared_ptr<Work>> next;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
      if (jobs_.empty()) return;  // stopping, and everything queued has run
      next = std::move(jobs_.front());
      jobs_.pop_front();
    }
    std::exception_ptr error;
    const double t0 = now();
    try {
      next.first();
    } catch (...) {  // reported to whoever waits on the Work
      error = std::current_exception();
    }
    next.second->finish(error, t0, now());
  }
}

// ---------------------------------------------------------------- pieces
int64_t Engine::piece_elems(const at::Tensor& t) const {
  return std::max<int64_t>(1, static_cast<int64_t>(slot_bytes_) / t.element_size());
}

void Engine::put(int dst, const at::Tensor& piece) {
  out_[dst].send(piece.data_ptr(), piece.numel() * piece.element_size(), timeout_s_);
}

void Engine::take(int src, at::Tensor& piece, bool reduce) {
  const std::size_t nbytes = piece.numel() * piece.element_size();
  if (reduce) {
    // Added straight from the slot, with no copy out of it first.
    auto [data, n] = in_[src].peek(timeout_s_);
    if (n != nbytes) throw std::runtime_error("a piece arrived with the wrong length");
    piece.add_(at::from_blob(const_cast<void*>(data), piece.sizes(), piece.options()));
    in_[src].release();
  } else {
    const std::size_t n = in_[src].recv(piece.data_ptr(), nbytes, timeout_s_);
    if (n != nbytes) throw std::runtime_error("a piece arrived with the wrong length");
  }
}

// Sends `send` to dst while receiving `recv` from src, one piece of each at a
// time, so a ring of ranks all sending at once never waits on itself.
void Engine::exchange(const at::Tensor& send, int dst, const at::Tensor& recv, int src, bool reduce) {
  const int64_t pe = piece_elems(send.defined() ? send : recv);
  const int64_t ns = send.defined() ? (send.numel() + pe - 1) / pe : 0;
  const int64_t nr = recv.defined() ? (recv.numel() + pe - 1) / pe : 0;
  for (int64_t k = 0; k < std::max(ns, nr); ++k) {
    if (k < ns) put(dst, send.slice(0, k * pe, std::min((k + 1) * pe, send.numel())));
    if (k < nr) {
      at::Tensor piece = recv.slice(0, k * pe, std::min((k + 1) * pe, recv.numel()));
      take(src, piece, reduce);
    }
  }
}

std::vector<at::Tensor> Engine::chunks(const at::Tensor& flat, const std::vector<int64_t>& sizes) const {
  if (sizes.empty()) return at::tensor_split(flat, size_);
  int64_t total = 0;
  for (int64_t s : sizes) total += s;
  if (static_cast<int>(sizes.size()) != size_ || total != flat.numel()) {
    throw std::invalid_argument("chunk sizes do not partition the tensor");
  }
  return at::split_with_sizes(flat, sizes);
}

void Engine::ring_reduce_scatter(const std::vector<at::Tensor>& c) {
  const int n = size_, r = rank_;
  for (int s = 0; s < n - 1; ++s) {
    exchange(c[mod(r - s - 1, n)], mod(r + 1, n), c[mod(r - s - 2, n)], mod(r - 1, n), true);
  }
}

void Engine::ring_all_gather(const std::vector<at::Tensor>& c) {
  const int n = size_, r = rank_;
  for (int s = 0; s < n - 1; ++s) {
    exchange(c[mod(r - s, n)], mod(r + 1, n), c[mod(r - s - 1, n)], mod(r - 1, n), false);
  }
}

// ---------------------------------------------------------------- collectives
std::shared_ptr<Work> Engine::all_reduce(at::Tensor t, bool average) {
  check(t, "all_reduce");
  at::Tensor flat = t.view(-1);
  return submit([this, flat, average]() mutable {
    // Divided before summing, as torch DDP does: exact for a power-of-two group.
    if (average) flat.div_(size_);
    if (size_ > 1) {
      auto c = chunks(flat, {});
      ring_reduce_scatter(c);
      ring_all_gather(c);
    }
  });
}

std::shared_ptr<Work> Engine::reduce_scatter(at::Tensor t, std::vector<int64_t> sizes, bool average) {
  check(t, "reduce_scatter");
  at::Tensor flat = t.view(-1);
  chunks(flat, sizes);  // validate now, in the caller's thread
  return submit([this, flat, sizes, average]() mutable {
    if (average) flat.div_(size_);
    if (size_ > 1) ring_reduce_scatter(chunks(flat, sizes));
  });
}

std::shared_ptr<Work> Engine::all_gather(at::Tensor t, std::vector<int64_t> sizes) {
  check(t, "all_gather");
  at::Tensor flat = t.view(-1);
  chunks(flat, sizes);
  return submit([this, flat, sizes] {
    if (size_ > 1) ring_all_gather(chunks(flat, sizes));
  });
}

std::shared_ptr<Work> Engine::broadcast(at::Tensor t, int root) {
  check(t, "broadcast");
  at::Tensor flat = t.view(-1);
  return submit([this, flat, root] {
    const int n = size_, pos = mod(rank_ - root, n);
    const int nxt = mod(rank_ + 1, n), prv = mod(rank_ - 1, n);
    const int64_t pe = piece_elems(flat);
    for (int64_t k = 0; k * pe < flat.numel(); ++k) {
      at::Tensor piece = flat.slice(0, k * pe, std::min((k + 1) * pe, flat.numel()));
      if (pos > 0) take(prv, piece, false);
      if (pos < n - 1) put(nxt, piece);
    }
  });
}

std::shared_ptr<Work> Engine::send(at::Tensor t, int dst) {
  check(t, "send");
  at::Tensor flat = t.view(-1);
  return submit([this, flat, dst] { exchange(flat, dst, at::Tensor(), -1, false); });
}

std::shared_ptr<Work> Engine::recv(at::Tensor t, int src) {
  check(t, "recv");
  at::Tensor flat = t.view(-1);
  return submit([this, flat, src] { exchange(at::Tensor(), -1, flat, src, false); });
}

std::shared_ptr<Work> Engine::sendrecv(at::Tensor send, int dst, at::Tensor recv, int src) {
  check(send, "sendrecv");
  check(recv, "sendrecv");
  at::Tensor s = send.view(-1), r = recv.view(-1);
  return submit([this, s, dst, r, src] { exchange(s, dst, r, src, false); });
}

// Every rank sends an empty message to every other, then waits for one from
// each. Channels are first in, first out, so this also orders the barrier after
// every collective queued before it.
std::shared_ptr<Work> Engine::barrier() {
  return submit([this] {
    for (int p = 0; p < size_; ++p) {
      if (p != rank_) out_[p].send(nullptr, 0, timeout_s_);
    }
    for (int p = 0; p < size_; ++p) {
      if (p == rank_) continue;
      unsigned char none;
      if (in_[p].recv(&none, 0, timeout_s_) != 0) throw std::runtime_error("barrier out of step");
    }
  });
}

}  // namespace tandem
