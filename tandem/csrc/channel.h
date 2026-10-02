// M1: a one-way channel between two processes, in shared memory.
//
// One process sends, one receives (single producer, single consumer). The
// channel is a ring of `slots` buffers of `slot_bytes` each, plus two counters:
//
//   head  messages written so far, advanced only by the sender
//   tail  messages read so far, advanced only by the receiver
//
// The ring is empty when head == tail and full when head - tail == slots;
// message k lives in slot k % slots. Both counters only grow (64 bits never
// wrap in practice), so there is no ambiguity between full and empty.
//
// The counters are std::atomic<std::uint64_t> placed in the shared region.
// They must be lock-free (static_assert it), and each on its own cache line
// (alignas(64)), so the sender and receiver do not slow each other down by
// writing to the same line. The ordering is the whole point of the exercise:
// the sender writes the payload, then publishes it by storing head with
// memory_order_release; the receiver loads head with memory_order_acquire
// before reading the payload, and the same pairing runs the other way for tail.
//
// Suggested layout of the region (you define the struct in channel.cpp):
//   header: magic number, slots, slot_bytes, then head and tail on their own lines
//   then `slots` entries of { std::uint64_t length; unsigned char bytes[slot_bytes]; }
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "shm.h"

namespace tandem {

// Thrown when send or recv waits longer than its timeout.
struct Timeout : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Channel {
 public:
  Channel() = default;  // empty; only for moving into

  // Creates the region `name` and initialises its header. slots >= 1 and
  // slot_bytes >= 1, or std::invalid_argument.
  static Channel create(const std::string& name, std::size_t slots, std::size_t slot_bytes);

  // Opens a channel another process created; std::runtime_error if the region
  // does not start with the channel's magic number.
  static Channel open(const std::string& name);

  // Copies n bytes into the next slot, waiting while the ring is full.
  // n > slot_bytes(): std::invalid_argument. Waits longer than timeout_s: Timeout.
  void send(const void* data, std::size_t n, double timeout_s);

  // Copies the next message into out and returns its length, waiting while the
  // ring is empty. A message longer than capacity: std::length_error, and the
  // message stays in the channel. Waits longer than timeout_s: Timeout.
  std::size_t recv(void* out, std::size_t capacity, double timeout_s);

  std::size_t slots() const noexcept;
  std::size_t slot_bytes() const noexcept;

 private:
  explicit Channel(SharedMemory shm) noexcept : shm_(std::move(shm)) {}

  SharedMemory shm_;
};

}  // namespace tandem
