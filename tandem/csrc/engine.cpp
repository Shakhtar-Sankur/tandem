// Python bindings for the engine (tandem.engine.lib()). The engine itself is in
// shm.cpp, channel.cpp and collectives.cpp; this file only exposes it.

#include <torch/extension.h>

#include <string>
#include <type_traits>

#include "channel.h"
#include "collectives.h"
#include "not_implemented.h"
#include "shm.h"

namespace py = pybind11;
using tandem::Channel;
using tandem::Engine;
using tandem::Work;
using tandem::SharedMemory;

// The ownership rules in shm.h, checked when this file compiles.
static_assert(!std::is_copy_constructible_v<SharedMemory>, "SharedMemory must not be copyable");
static_assert(!std::is_copy_assignable_v<SharedMemory>, "SharedMemory must not be copyable");
static_assert(std::is_nothrow_move_constructible_v<SharedMemory>, "moving must not throw");
static_assert(std::is_nothrow_move_assignable_v<SharedMemory>, "moving must not throw");

namespace {

void check_range(const SharedMemory& s, std::size_t offset, std::size_t n) {
  if (s.data() == nullptr) throw std::invalid_argument("shared memory is closed");
  if (offset > s.size() || n > s.size() - offset) throw std::out_of_range("outside the region");
}

// Exercises the move operations from C++, where Python cannot reach them.
py::dict move_check(const std::string& a_name, const std::string& b_name) {
  py::dict out;
  SharedMemory a = SharedMemory::create(a_name, 4096);
  void* a_data = a.data();
  SharedMemory b(std::move(a));  // move construction
  out["source_emptied"] = a.data() == nullptr && a.size() == 0 && !a.owner();
  out["target_took_over"] = b.data() == a_data && b.size() == 4096 && b.owner();
  SharedMemory c = SharedMemory::create(b_name, 4096);
  c = std::move(b);  // move assignment: c must first release b_name's region
  out["assigned"] = c.data() == a_data && c.name() == a_name && b.data() == nullptr;
  SharedMemory& alias = c;
  c = std::move(alias);
  out["self_assign_safe"] = c.data() == a_data;
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::register_exception<tandem::NotImplemented>(m, "NotImplemented", PyExc_NotImplementedError);
  py::register_exception<tandem::Timeout>(m, "Timeout", PyExc_TimeoutError);

  py::class_<SharedMemory>(m, "SharedMemory")
      .def_static("create", &SharedMemory::create, py::arg("name"), py::arg("bytes"))
      .def_static("open", &SharedMemory::open, py::arg("name"))
      .def_property_readonly("size", &SharedMemory::size)
      .def_property_readonly("name", &SharedMemory::name)
      .def_property_readonly("owner", &SharedMemory::owner)
      .def_property_readonly("closed", [](const SharedMemory& s) { return s.data() == nullptr; })
      .def("write", [](SharedMemory& s, std::size_t offset, const py::bytes& b) {
        std::string data = b;
        check_range(s, offset, data.size());
        std::memcpy(static_cast<char*>(s.data()) + offset, data.data(), data.size());
      })
      .def("read", [](const SharedMemory& s, std::size_t offset, std::size_t n) {
        check_range(s, offset, n);
        return py::bytes(static_cast<const char*>(s.data()) + offset, n);
      })
      // Releases the region now (moving it into a temporary that is destroyed).
      .def("close", [](SharedMemory& s) { SharedMemory gone(std::move(s)); });

  m.def("_move_check", &move_check);

  py::class_<Channel>(m, "Channel")
      .def_static("create", &Channel::create, py::arg("name"), py::arg("slots"), py::arg("slot_bytes"))
      .def_static("open", &Channel::open, py::arg("name"))
      .def_property_readonly("slots", &Channel::slots)
      .def_property_readonly("slot_bytes", &Channel::slot_bytes)
      .def("send", [](Channel& c, const py::bytes& b, double timeout) {
        std::string data = b;
        py::gil_scoped_release unlocked;  // other Python threads run while this waits
        c.send(data.data(), data.size(), timeout);
      }, py::arg("data"), py::arg("timeout") = 10.0)
      .def("recv", [](Channel& c, std::size_t capacity, double timeout) {
        std::string out(capacity, '\0');
        std::size_t n;
        {
          py::gil_scoped_release unlocked;
          n = c.recv(out.data(), capacity, timeout);
        }
        return py::bytes(out.data(), n);
      }, py::arg("capacity"), py::arg("timeout") = 10.0);

  py::class_<Work, std::shared_ptr<Work>>(m, "Work")
      .def("wait", &Work::wait, py::call_guard<py::gil_scoped_release>())
      .def("done", &Work::done)
      .def_property_readonly("started", &Work::started)
      .def_property_readonly("finished", &Work::finished);

  py::class_<Engine>(m, "Engine")
      .def(py::init<const std::string&, int, int, std::size_t, std::size_t, double>(), py::arg("job"),
           py::arg("rank"), py::arg("size"), py::arg("slots"), py::arg("slot_bytes"), py::arg("timeout"))
      .def("connect", &Engine::connect, py::call_guard<py::gil_scoped_release>())
      .def("close", &Engine::close, py::call_guard<py::gil_scoped_release>())
      .def("all_reduce", &Engine::all_reduce, py::arg("t"), py::arg("average"))
      .def("reduce_scatter", &Engine::reduce_scatter, py::arg("t"), py::arg("sizes"), py::arg("average"))
      .def("all_gather", &Engine::all_gather, py::arg("t"), py::arg("sizes"))
      .def("broadcast", &Engine::broadcast, py::arg("t"), py::arg("root"))
      .def("send", &Engine::send, py::arg("t"), py::arg("dst"))
      .def("recv", &Engine::recv, py::arg("t"), py::arg("src"))
      .def("sendrecv", &Engine::sendrecv, py::arg("send"), py::arg("dst"), py::arg("recv"), py::arg("src"))
      .def("barrier", &Engine::barrier);
}
