// M3: moving pieces between GPUs in different processes, without the host
// waiting for any copy to finish.
//
// For every ordered pair (src, dst), src allocates `slots` buffers of
// `slot_bytes` on its own GPU and shares them with dst through CUDA IPC
// (cudaIpcGetMemHandle). Two sets of interprocess events order the GPU work:
//
//   full[j]  recorded by src after copying a piece into slot j; dst's stream
//            waits on it before reading the slot.
//   free[j]  recorded by dst after reading slot j; src's stream waits on it
//            before copying the next piece into the slot.
//
// The host only says *which* slot is next and how long the piece is: an
// 8-byte message on the pair's Channel, sent right after the copy and the
// event are *enqueued*, not after they finish. Because the event record is
// enqueued before the message is sent, and the other side waits on the event
// only after receiving the message, every wait refers to the right record.
// No host thread ever synchronizes with the GPU per piece; that was the cost
// of the Python engine's transport.
//
// Each rank publishes its handles in a small shared-memory region,
// "/tandem-<job>-<rank>-cuda"; connect() reads the other ranks' regions.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "channel.h"
#include "shm.h"

namespace tandem {

// Throws std::runtime_error naming the call if a CUDA runtime call fails.
void cuda_check(cudaError_t err, const char* what);

class CudaLinks {
 public:
  CudaLinks(const std::string& job, int rank, int size, int device, std::size_t slots, std::size_t slot_bytes);
  void connect();  // after every rank has constructed its CudaLinks
  void close();    // after every rank has finished its last collective
  ~CudaLinks();

  CudaLinks(const CudaLinks&) = delete;
  CudaLinks& operator=(const CudaLinks&) = delete;

  int device() const { return device_; }
  cudaStream_t stream() const { return stream_; }
  bool peer_access(int peer) const { return p2p_[peer]; }

  // Enqueues the copy of n bytes at src (on this GPU) into the next slot to dst,
  // then tells dst over `signal`. Waits on the host only while all slots are
  // taken (dst has not released them yet).
  void put(Channel& signal, int dst, const void* src, std::size_t n, double timeout_s);

  // Waits for the next piece from src and makes this rank's stream wait until
  // its copy has landed. Returns where it can be read on the stream: the
  // peer's slot itself, or (local = true) a copy of it in this GPU's memory.
  const void* receive(Channel& signal, int src, std::size_t n, bool local, double timeout_s);

  // After the stream's reads of the piece are enqueued: marks the slot free
  // for src (on the stream) and releases the message.
  void done_reading(Channel& signal, int src);

 private:
  std::size_t region_bytes() const;
  unsigned char* entry(void* region, int peer) const;

  std::string job_;
  int rank_, size_, device_;
  std::size_t slots_, slot_bytes_;
  cudaStream_t stream_ = nullptr;
  SharedMemory region_;  // this rank's published handles

  // Indexed [peer] or [peer][slot].
  std::vector<void*> out_mem_;                       // our slots for sending to peer
  std::vector<void*> scratch_;                       // local copies of peer pieces (no peer access)
  std::vector<std::vector<cudaEvent_t>> full_;       // ours: our slot to peer is full
  std::vector<std::vector<cudaEvent_t>> free_;       // ours: we finished reading peer's slot
  std::vector<void*> peer_mem_;                      // peer's slots for sending to us
  std::vector<std::vector<cudaEvent_t>> peer_full_;  // peer's: its slot to us is full
  std::vector<std::vector<cudaEvent_t>> peer_free_;  // peer's: it finished reading our slot
  std::vector<char> p2p_;
  std::vector<std::uint64_t> sent_, got_;            // pieces sent to / received from each peer
  bool closed_ = false;
};

}  // namespace tandem
