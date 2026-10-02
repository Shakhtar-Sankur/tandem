// M0: a POSIX shared-memory region, owned by one object (RAII).
//
// The creator makes the region (shm_open with O_CREAT | O_EXCL, ftruncate to
// `bytes`, mmap) and removes its name (shm_unlink) when it is destroyed; other
// processes open it by name and only unmap it. Copying is forbidden (two owners
// would unmap twice); moving transfers ownership and leaves the source empty.
#pragma once

#include <cstddef>
#include <string>

namespace tandem {

class SharedMemory {
 public:
  // An empty object: data() == nullptr, size() == 0. Destroying it does nothing.
  SharedMemory() noexcept = default;

  // Creates a new region called `name` ("/tandem-..."), zero-filled, of `bytes`
  // bytes, mapped read-write. Throws std::system_error naming the failing call
  // (for example "shm_open /tandem-x: File exists") if any step fails, and leaves
  // nothing behind: no name, no descriptor, no mapping.
  static SharedMemory create(const std::string& name, std::size_t bytes);

  // Maps an existing region read-write; its size comes from fstat. Throws
  // std::system_error if it does not exist.
  static SharedMemory open(const std::string& name);

  SharedMemory(const SharedMemory&) = delete;
  SharedMemory& operator=(const SharedMemory&) = delete;
  SharedMemory(SharedMemory&& other) noexcept;
  SharedMemory& operator=(SharedMemory&& other) noexcept;  // releases what *this held first
  ~SharedMemory();

  void* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  const std::string& name() const noexcept { return name_; }
  bool owner() const noexcept { return owner_; }

 private:
  void release() noexcept;  // munmap, close, and shm_unlink if owner; then empty

  std::string name_;
  void* data_ = nullptr;
  std::size_t size_ = 0;
  int fd_ = -1;
  bool owner_ = false;
};

}  // namespace tandem
