// Thrown by the parts of the engine that are not written yet; Python sees
// NotImplementedError, and the tests for that part are skipped.
#pragma once

#include <stdexcept>
#include <string>

namespace tandem {

struct NotImplemented : std::logic_error {
  explicit NotImplemented(const std::string& what) : std::logic_error("not implemented: " + what) {}
};

}  // namespace tandem
