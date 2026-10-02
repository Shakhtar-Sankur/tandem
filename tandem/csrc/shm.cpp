// M0: implement SharedMemory (see shm.h). Replace each NotImplemented.
//
// Headers you will need: <fcntl.h> (O_CREAT, O_EXCL, O_RDWR), <sys/mman.h>
// (shm_open, mmap, munmap, shm_unlink), <sys/stat.h> (fstat), <unistd.h>
// (ftruncate, close), <cerrno> and <system_error>.
//
// To report a failed call:
//   throw std::system_error(errno, std::generic_category(), "shm_open " + name);
// Read errno before calling anything else (close, shm_unlink): they can change it.

#include "shm.h"

#include "not_implemented.h"

namespace tandem {

SharedMemory SharedMemory::create(const std::string& name, std::size_t bytes) {
  // TODO(M0): shm_open(O_CREAT | O_EXCL | O_RDWR, 0600), ftruncate, mmap.
  // On any failure, undo what already succeeded before throwing.
  (void)name;
  (void)bytes;
  throw NotImplemented("SharedMemory::create");
}

SharedMemory SharedMemory::open(const std::string& name) {
  // TODO(M0): shm_open(O_RDWR), fstat for the size, mmap.
  (void)name;
  throw NotImplemented("SharedMemory::open");
}

SharedMemory::SharedMemory(SharedMemory&& other) noexcept {
  // TODO(M0): take other's fields; leave other empty (so its destructor does nothing).
  (void)other;
}

SharedMemory& SharedMemory::operator=(SharedMemory&& other) noexcept {
  // TODO(M0): release what *this holds, then take other's fields. Mind self-assignment.
  (void)other;
  return *this;
}

SharedMemory::~SharedMemory() { release(); }

void SharedMemory::release() noexcept {
  // TODO(M0): munmap, close, shm_unlink if owner_; then reset every field.
}

}  // namespace tandem
