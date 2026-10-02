// M1: implement Channel (see channel.h). Replace each NotImplemented.
//
// Headers you will need: <atomic>, <chrono>, <cstring> (std::memcpy), <thread>.
// Waiting: spin briefly (checking the counter), then std::this_thread::yield(),
// then sleep for a few microseconds at a time, until the deadline passes.

#include "channel.h"

#include "not_implemented.h"

namespace tandem {

Channel Channel::create(const std::string& name, std::size_t slots, std::size_t slot_bytes) {
  // TODO(M1): validate, SharedMemory::create(name, <header + slots entries>),
  // placement-new the header into shm.data(), fill it in.
  (void)name;
  (void)slots;
  (void)slot_bytes;
  throw NotImplemented("Channel::create");
}

Channel Channel::open(const std::string& name) {
  // TODO(M1): SharedMemory::open(name), check the magic number.
  (void)name;
  throw NotImplemented("Channel::open");
}

void Channel::send(const void* data, std::size_t n, double timeout_s) {
  // TODO(M1): wait until head - tail < slots, copy into slot head % slots,
  // then publish: head.store(head + 1, std::memory_order_release).
  (void)data;
  (void)n;
  (void)timeout_s;
  throw NotImplemented("Channel::send");
}

std::size_t Channel::recv(void* out, std::size_t capacity, double timeout_s) {
  // TODO(M1): wait until head (loaded with acquire) > tail, copy out of slot
  // tail % slots, then free it: tail.store(tail + 1, std::memory_order_release).
  (void)out;
  (void)capacity;
  (void)timeout_s;
  throw NotImplemented("Channel::recv");
}

std::size_t Channel::slots() const noexcept {
  // TODO(M1): read it from the header.
  return 0;
}

std::size_t Channel::slot_bytes() const noexcept {
  // TODO(M1): read it from the header.
  return 0;
}

}  // namespace tandem
