"""The C++ engine (tandem/csrc), compiled on first use with torch's extension
builder and cached under ~/.cache/torch_extensions. With a GPU visible it is
built with its CUDA path (cuda_links.cpp, reduce.cu); otherwise CPU only.
See docs/engine.md."""

import functools
import os

_CSRC = os.path.join(os.path.dirname(__file__), "csrc")
SOURCES = ["engine.cpp", "shm.cpp", "channel.cpp", "collectives.cpp"]
CUDA_SOURCES = ["cuda_links.cpp", "reduce.cu"]


def _want_cuda():
    import torch

    if os.environ.get("TANDEM_ENGINE_CUDA") in ("0", "1"):
        return os.environ["TANDEM_ENGINE_CUDA"] == "1"
    return torch.cuda.is_available()


@functools.cache
def lib():
    from torch.utils.cpp_extension import load

    cuda = _want_cuda()
    sources = SOURCES + (CUDA_SOURCES if cuda else [])
    flags = ["-O2", "-Wall", "-Wextra"] + (["-DTANDEM_CUDA"] if cuda else [])
    return load(
        name="tandem_engine_cuda" if cuda else "tandem_engine",
        sources=[os.path.join(_CSRC, s) for s in sources],
        extra_cflags=flags,
        extra_cuda_cflags=["-O3", "-DTANDEM_CUDA"],
        extra_ldflags=["-lrt"],
        with_cuda=cuda,
        verbose=os.environ.get("TANDEM_BUILD_VERBOSE") == "1",
    )
