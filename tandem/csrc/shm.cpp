// M0: SharedMemory (see shm.h).

#include "shm.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <system_error>
#include <utility>

namespace tandem {

namespace {

// Captures errno now, before any cleanup call can overwrite it.
std::system_error os_error(const std::string& what) {
  return std::system_error(errno, std::generic_category(), what);
}

}  // namespace

SharedMemory SharedMemory::create(const std::string& name, std::size_t bytes) {
  int fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) throw os_error("shm_open " + name);
  if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    auto e = os_error("ftruncate " + name);
    ::close(fd);
    ::shm_unlink(name.c_str());
    throw e;
  }
  void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) {
    auto e = os_error("mmap " + name);
    ::close(fd);
    ::shm_unlink(name.c_str());
    throw e;
  }
  SharedMemory s;
  s.name_ = name;
  s.data_ = p;
  s.size_ = bytes;
  s.fd_ = fd;
  s.owner_ = true;
  return s;
}

SharedMemory SharedMemory::open(const std::string& name) {
  int fd = ::shm_open(name.c_str(), O_RDWR, 0600);
  if (fd < 0) throw os_error("shm_open " + name);
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    auto e = os_error("fstat " + name);
    ::close(fd);
    throw e;
  }
  auto bytes = static_cast<std::size_t>(st.st_size);
  void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) {
    auto e = os_error("mmap " + name);
    ::close(fd);
    throw e;
  }
  SharedMemory s;
  s.name_ = name;
  s.data_ = p;
  s.size_ = bytes;
  s.fd_ = fd;
  return s;
}

SharedMemory::SharedMemory(SharedMemory&& other) noexcept
    : name_(std::move(other.name_)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      fd_(std::exchange(other.fd_, -1)),
      owner_(std::exchange(other.owner_, false)) {
  other.name_.clear();
}

SharedMemory& SharedMemory::operator=(SharedMemory&& other) noexcept {
  if (this != &other) {
    release();
    name_ = std::move(other.name_);
    other.name_.clear();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    fd_ = std::exchange(other.fd_, -1);
    owner_ = std::exchange(other.owner_, false);
  }
  return *this;
}

SharedMemory::~SharedMemory() { release(); }

void SharedMemory::release() noexcept {
  if (data_ != nullptr) ::munmap(data_, size_);
  if (fd_ >= 0) ::close(fd_);
  if (owner_) ::shm_unlink(name_.c_str());
  name_.clear();
  data_ = nullptr;
  size_ = 0;
  fd_ = -1;
  owner_ = false;
}

}  // namespace tandem
