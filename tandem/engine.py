"""The C++ engine (tandem/csrc), compiled on first use with torch's extension
builder and cached under ~/.cache/torch_extensions. See docs/engine.md."""

import functools
import os

_CSRC = os.path.join(os.path.dirname(__file__), "csrc")
SOURCES = ["engine.cpp", "shm.cpp", "channel.cpp", "collectives.cpp"]


@functools.cache
def lib():
    from torch.utils.cpp_extension import load

    return load(
        name="tandem_engine",
        sources=[os.path.join(_CSRC, s) for s in SOURCES],
        extra_cflags=["-O2", "-Wall", "-Wextra"],
        extra_ldflags=["-lrt"],
        verbose=os.environ.get("TANDEM_BUILD_VERBOSE") == "1",
    )
