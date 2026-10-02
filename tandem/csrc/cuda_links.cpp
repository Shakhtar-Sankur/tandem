// M3: CudaLinks (see cuda_links.h).

#include "cuda_links.h"

#include <cstring>
#include <stdexcept>

namespace tandem {

namespace {

constexpr std::uint64_t kRegionMagic = 0x74616e64656d4344ull;  // "tandemCD"

// The region: this header, then one entry per peer (our own slot is unused):
//   cudaIpcMemHandle_t    out    our slots for sending to that peer
//   cudaIpcEventHandle_t  full[slots]
//   cudaIpcEventHandle_t  free[slots]   ours, for reading that peer's slots
struct RegionHeader {
  std::uint64_t magic;
  std::int32_t device;
  std::int32_t size;
  std::uint64_t slots;
};

constexpr unsigned kEventFlags = cudaEventDisableTiming | cudaEventInterprocess;

}  // namespace

void cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

std::size_t CudaLinks::region_bytes() const {
  const std::size_t entry = sizeof(cudaIpcMemHandle_t) + 2 * slots_ * sizeof(cudaIpcEventHandle_t);
  return sizeof(RegionHeader) + static_cast<std::size_t>(size_) * entry;
}

unsigned char* CudaLinks::entry(void* region, int peer) const {
  const std::size_t entry = sizeof(cudaIpcMemHandle_t) + 2 * slots_ * sizeof(cudaIpcEventHandle_t);
  return static_cast<unsigned char*>(region) + sizeof(RegionHeader) + static_cast<std::size_t>(peer) * entry;
}

CudaLinks::CudaLinks(const std::string& job, int rank, int size, int device, std::size_t slots,
                     std::size_t slot_bytes)
    : job_(job), rank_(rank), size_(size), device_(device), slots_(slots), slot_bytes_(slot_bytes),
      out_mem_(size, nullptr), scratch_(size, nullptr), full_(size), free_(size), peer_mem_(size, nullptr),
      peer_full_(size), peer_free_(size), p2p_(size, 0), sent_(size, 0), got_(size, 0) {
  cuda_check(cudaSetDevice(device_), "cudaSetDevice");
  int least, greatest;  // numerically: greatest priority is the smallest number
  cuda_check(cudaDeviceGetStreamPriorityRange(&least, &greatest), "cudaDeviceGetStreamPriorityRange");
  cuda_check(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, greatest), "cudaStreamCreateWithPriority");

  region_ = SharedMemory::create("/tandem-" + job_ + "-" + std::to_string(rank_) + "-cuda", region_bytes());
  auto* h = static_cast<RegionHeader*>(region_.data());
  h->magic = kRegionMagic;
  h->device = device_;
  h->size = size_;
  h->slots = slots_;
  for (int p = 0; p < size_; ++p) {
    if (p == rank_) continue;
    unsigned char* e = entry(region_.data(), p);
    cuda_check(cudaMalloc(&out_mem_[p], slots_ * slot_bytes_), "cudaMalloc (slots)");
    cuda_check(cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t*>(e), out_mem_[p]), "cudaIpcGetMemHandle");
    e += sizeof(cudaIpcMemHandle_t);
    full_[p].resize(slots_);
    free_[p].resize(slots_);
    for (std::size_t j = 0; j < slots_; ++j) {
      cuda_check(cudaEventCreateWithFlags(&full_[p][j], kEventFlags), "cudaEventCreateWithFlags");
      cuda_check(cudaIpcGetEventHandle(reinterpret_cast<cudaIpcEventHandle_t*>(e + j * sizeof(cudaIpcEventHandle_t)),
                                       full_[p][j]),
                 "cudaIpcGetEventHandle");
    }
    e += slots_ * sizeof(cudaIpcEventHandle_t);
    for (std::size_t j = 0; j < slots_; ++j) {
      cuda_check(cudaEventCreateWithFlags(&free_[p][j], kEventFlags), "cudaEventCreateWithFlags");
      cuda_check(cudaIpcGetEventHandle(reinterpret_cast<cudaIpcEventHandle_t*>(e + j * sizeof(cudaIpcEventHandle_t)),
                                       free_[p][j]),
                 "cudaIpcGetEventHandle");
    }
  }
}

void CudaLinks::connect() {
  cuda_check(cudaSetDevice(device_), "cudaSetDevice");
  for (int p = 0; p < size_; ++p) {
    if (p == rank_) continue;
    SharedMemory theirs = SharedMemory::open("/tandem-" + job_ + "-" + std::to_string(p) + "-cuda");
    const auto* h = static_cast<const RegionHeader*>(theirs.data());
    if (theirs.size() < sizeof(RegionHeader) || h->magic != kRegionMagic || h->size != size_ || h->slots != slots_) {
      throw std::runtime_error("rank " + std::to_string(p) + "'s CUDA region does not match ours");
    }
    int can = 0;
    cuda_check(cudaDeviceCanAccessPeer(&can, device_, h->device), "cudaDeviceCanAccessPeer");
    p2p_[p] = h->device != device_ && can;

    // In p's region, the entry for us: p's slots for sending to us, their
    // full events, and the free events p made for reading *our* slots.
    unsigned char* e = entry(theirs.data(), rank_);
    cudaIpcMemHandle_t mem;
    std::memcpy(&mem, e, sizeof mem);
    cuda_check(cudaIpcOpenMemHandle(&peer_mem_[p], mem, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle");
    e += sizeof(cudaIpcMemHandle_t);
    peer_full_[p].resize(slots_);
    peer_free_[p].resize(slots_);
    for (std::size_t j = 0; j < slots_; ++j) {
      cudaIpcEventHandle_t ev;
      std::memcpy(&ev, e + j * sizeof ev, sizeof ev);
      cuda_check(cudaIpcOpenEventHandle(&peer_full_[p][j], ev), "cudaIpcOpenEventHandle");
    }
    e += slots_ * sizeof(cudaIpcEventHandle_t);
    for (std::size_t j = 0; j < slots_; ++j) {
      cudaIpcEventHandle_t ev;
      std::memcpy(&ev, e + j * sizeof ev, sizeof ev);
      cuda_check(cudaIpcOpenEventHandle(&peer_free_[p][j], ev), "cudaIpcOpenEventHandle");
    }
    if (!p2p_[p]) cuda_check(cudaMalloc(&scratch_[p], slots_ * slot_bytes_), "cudaMalloc (scratch)");
  }
}

void CudaLinks::put(Channel& signal, int dst, const void* src, std::size_t n, double timeout_s) {
  if (n > slot_bytes_) throw std::invalid_argument("piece larger than a slot");
  const std::uint64_t k = sent_[dst];
  const std::size_t j = k % slots_;
  signal.wait_writable(timeout_s);  // dst has released slot j's previous piece...
  if (k >= slots_) {
    // ...and enqueued its reads of it before recording free[j]: wait for those reads.
    cuda_check(cudaStreamWaitEvent(stream_, peer_free_[dst][j], 0), "cudaStreamWaitEvent (free)");
  }
  void* slot = static_cast<unsigned char*>(out_mem_[dst]) + j * slot_bytes_;
  cuda_check(cudaMemcpyAsync(slot, src, n, cudaMemcpyDeviceToDevice, stream_), "cudaMemcpyAsync (put)");
  cuda_check(cudaEventRecord(full_[dst][j], stream_), "cudaEventRecord (full)");
  const std::uint64_t length = n;
  signal.send(&length, sizeof length, timeout_s);  // never blocks: we waited for room above
  sent_[dst] = k + 1;
}

const void* CudaLinks::receive(Channel& signal, int src, std::size_t n, bool local, double timeout_s) {
  auto [msg, len] = signal.peek(timeout_s);
  std::uint64_t length = 0;
  if (len != sizeof length) throw std::runtime_error("collectives out of step: expected a piece, got another message");
  std::memcpy(&length, msg, sizeof length);
  if (length != n) throw std::runtime_error("a piece arrived with the wrong length");
  const std::size_t j = got_[src] % slots_;
  cuda_check(cudaStreamWaitEvent(stream_, peer_full_[src][j], 0), "cudaStreamWaitEvent (full)");
  const void* slot = static_cast<const unsigned char*>(peer_mem_[src]) + j * slot_bytes_;
  if (!local) return slot;
  if (scratch_[src] == nullptr) {  // peer access exists, but the caller wants a local copy
    cuda_check(cudaMalloc(&scratch_[src], slots_ * slot_bytes_), "cudaMalloc (scratch)");
  }
  void* copy = static_cast<unsigned char*>(scratch_[src]) + j * slot_bytes_;
  cuda_check(cudaMemcpyAsync(copy, slot, n, cudaMemcpyDefault, stream_), "cudaMemcpyAsync (receive)");
  return copy;
}

void CudaLinks::done_reading(Channel& signal, int src) {
  const std::size_t j = got_[src] % slots_;
  cuda_check(cudaEventRecord(free_[src][j], stream_), "cudaEventRecord (free)");
  got_[src] += 1;
  signal.release();
}

void CudaLinks::close() {
  if (closed_) return;
  closed_ = true;
  cudaSetDevice(device_);
  cudaStreamSynchronize(stream_);
  for (int p = 0; p < size_; ++p) {
    if (p == rank_) continue;
    if (peer_mem_[p]) cudaIpcCloseMemHandle(peer_mem_[p]);
    for (cudaEvent_t e : peer_full_[p]) cudaEventDestroy(e);
    for (cudaEvent_t e : peer_free_[p]) cudaEventDestroy(e);
    for (cudaEvent_t e : full_[p]) cudaEventDestroy(e);
    for (cudaEvent_t e : free_[p]) cudaEventDestroy(e);
    if (out_mem_[p]) cudaFree(out_mem_[p]);
    if (scratch_[p]) cudaFree(scratch_[p]);
  }
  if (stream_) cudaStreamDestroy(stream_);
  stream_ = nullptr;
}

CudaLinks::~CudaLinks() { close(); }

}  // namespace tandem
