// M1: Channel (see channel.h).

#include "channel.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <new>
#include <thread>

namespace tandem {

namespace {

constexpr std::uint64_t kMagic = 0x74616e64656d4331ull;  // "tandemC1"
constexpr std::size_t kLine = 64;                        // a cache line

static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "atomics shared between processes must be lock-free");

// The start of the region. head and tail each get a cache line of their own,
// so the sender's writes to head never invalidate the line holding tail, and
// the other way round.
struct alignas(kLine) Header {
  std::uint64_t magic;
  std::uint64_t slots;
  std::uint64_t slot_bytes;
  alignas(kLine) std::atomic<std::uint64_t> head{0};  // written by the sender
  alignas(kLine) std::atomic<std::uint64_t> tail{0};  // written by the receiver
};

// Each slot: an 8-byte length, then the payload, rounded up to a cache line.
std::size_t entry_bytes(std::size_t slot_bytes) { return (8 + slot_bytes + kLine - 1) / kLine * kLine; }

Header* header(const SharedMemory& shm) { return static_cast<Header*>(shm.data()); }

unsigned char* entry(const SharedMemory& shm, std::uint64_t k) {
  Header* h = header(shm);
  return static_cast<unsigned char*>(shm.data()) + sizeof(Header) + (k % h->slots) * entry_bytes(h->slot_bytes);
}

// Waits until ready() holds. Inside a collective the other side answers within
// microseconds, so this spins, then yields the core to other threads; only a
// wait longer than a millisecond (an idle peer) falls back to short sleeps,
// which cost wake-up latency but no CPU.
template <class Ready>
void wait_until(Ready ready, double timeout_s, const char* what) {
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  const auto deadline = start + std::chrono::duration<double>(timeout_s);
  const auto patience = start + std::chrono::milliseconds(1);
  for (std::uint64_t i = 0; !ready(); ++i) {
    if (i < 64) continue;
    if ((i & 63) != 0) {
      std::this_thread::yield();
      continue;
    }
    const auto now = clock::now();  // every 64 rounds: check the clocks
    if (now > deadline) throw Timeout(what);
    if (now > patience) std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

}  // namespace

Channel Channel::create(const std::string& name, std::size_t slots, std::size_t slot_bytes) {
  if (slots == 0 || slot_bytes == 0) throw std::invalid_argument("a channel needs slots >= 1 and slot_bytes >= 1");
  SharedMemory shm = SharedMemory::create(name, sizeof(Header) + slots * entry_bytes(slot_bytes));
  Header* h = new (shm.data()) Header;  // construct the atomics in place
  h->magic = kMagic;
  h->slots = slots;
  h->slot_bytes = slot_bytes;
  return Channel(std::move(shm));
}

Channel Channel::open(const std::string& name) {
  SharedMemory shm = SharedMemory::open(name);
  if (shm.size() < sizeof(Header) || header(shm)->magic != kMagic) {
    throw std::runtime_error("not a tandem channel: " + name);
  }
  return Channel(std::move(shm));
}

void Channel::send(const void* data, std::size_t n, double timeout_s) {
  Header* h = header(shm_);
  if (n > h->slot_bytes) throw std::invalid_argument("message larger than a slot");
  // Only this process writes head, so reading our own value needs no ordering.
  const std::uint64_t head = h->head.load(std::memory_order_relaxed);
  // acquire: once we see the receiver's tail, its reads of that slot are done.
  wait_until([&] { return head - h->tail.load(std::memory_order_acquire) < h->slots; }, timeout_s,
             "send: the channel stayed full");
  unsigned char* e = entry(shm_, head);
  const std::uint64_t length = n;
  std::memcpy(e, &length, sizeof length);
  if (n > 0) std::memcpy(e + sizeof length, data, n);
  // release: the payload above becomes visible before the new head does.
  h->head.store(head + 1, std::memory_order_release);
}

std::pair<const void*, std::size_t> Channel::peek(double timeout_s) {
  Header* h = header(shm_);
  const std::uint64_t tail = h->tail.load(std::memory_order_relaxed);
  // acquire: pairs with the sender's release store of head, so the payload
  // written before that store is visible here.
  wait_until([&] { return h->head.load(std::memory_order_acquire) > tail; }, timeout_s,
             "recv: the channel stayed empty");
  const unsigned char* e = entry(shm_, tail);
  std::uint64_t length;
  std::memcpy(&length, e, sizeof length);
  return {e + sizeof length, static_cast<std::size_t>(length)};
}

void Channel::release() {
  Header* h = header(shm_);
  // release: our reads of the slot finish before the sender may reuse it.
  h->tail.store(h->tail.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}

std::size_t Channel::recv(void* out, std::size_t capacity, double timeout_s) {
  auto [data, length] = peek(timeout_s);
  if (length > capacity) throw std::length_error("message longer than the receive buffer");
  if (length > 0) std::memcpy(out, data, length);
  release();
  return length;
}

std::size_t Channel::slots() const noexcept { return shm_.data() ? header(shm_)->slots : 0; }

std::size_t Channel::slot_bytes() const noexcept { return shm_.data() ? header(shm_)->slot_bytes : 0; }

}  // namespace tandem
