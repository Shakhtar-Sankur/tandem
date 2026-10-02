"""The C++ engine (docs/engine.md), milestone by milestone. A milestone whose
code is not written yet raises NotImplementedError, and its tests skip."""

import multiprocessing as mp
import os
import time
import zlib

import pytest

from tandem.engine import lib


def name(tag):
    return f"/tandem-test-{os.getpid()}-{tag}-{time.monotonic_ns()}"


def needs(fn, *args, **kwargs):
    """Calls fn; skips the test if that part of the engine is not written yet."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(str(e))


def run_in_process(target, *args, timeout=60):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_report, args=(q, target, args))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        raise AssertionError(f"{target.__name__} did not finish in {timeout}s")
    ok, value = q.get(timeout=5)
    if not ok:
        raise AssertionError(f"{target.__name__} failed in the child: {value}")
    return value


def _report(q, target, args):
    try:
        q.put((True, target(*args)))
    except BaseException as e:  # sent back to the test
        q.put((False, repr(e)))


# ---------------------------------------------------------------- M0: SharedMemory
def _child_reads_and_writes(region):
    s = lib().SharedMemory.open(region)
    seen = s.read(0, 5)
    s.write(100, b"from the child")
    return seen, s.size, s.owner


def test_m0_create_write_read():
    e = lib()
    s = needs(e.SharedMemory.create, name("rw"), 4096)
    assert s.size == 4096 and s.owner and not s.closed
    assert s.read(0, 16) == bytes(16), "a new region is zero-filled"
    s.write(10, b"hello")
    assert s.read(10, 5) == b"hello"
    s.close()


def test_m0_two_processes_share_memory():
    e = lib()
    region = name("share")
    s = needs(e.SharedMemory.create, region, 8192)
    s.write(0, b"hello")
    seen, size, owner = run_in_process(_child_reads_and_writes, region)
    assert seen == b"hello" and size == 8192 and owner is False
    assert s.read(100, 14) == b"from the child"
    s.close()


def test_m0_creator_unlinks_the_name():
    e = lib()
    region = name("unlink")
    s = needs(e.SharedMemory.create, region, 4096)
    other = e.SharedMemory.open(region)
    s.close()
    assert s.closed
    with pytest.raises(RuntimeError, match="shm_open"):
        e.SharedMemory.open(region)  # the name is gone...
    other.write(0, b"x")  # ...but a mapping that is still open keeps working
    assert other.read(0, 1) == b"x"
    other.close()


def test_m0_errors_name_the_call():
    e = lib()
    with pytest.raises(RuntimeError, match="shm_open"):
        needs(e.SharedMemory.open, name("missing"))
    region = name("twice")
    s = needs(e.SharedMemory.create, region, 4096)
    with pytest.raises(RuntimeError, match="shm_open"):
        e.SharedMemory.create(region, 4096)  # O_EXCL: already exists
    s.close()


def test_m0_failed_create_leaves_nothing_behind():
    e = lib()
    region = name("huge")
    with pytest.raises(RuntimeError):
        needs(e.SharedMemory.create, region, 1 << 62)  # ftruncate or mmap fails
    with pytest.raises(RuntimeError, match="shm_open"):
        e.SharedMemory.open(region)


def test_m0_move_operations():
    e = lib()
    a, b = name("move-a"), name("move-b")
    r = needs(e._move_check, a, b)
    assert r == {"source_emptied": True, "target_took_over": True, "assigned": True, "self_assign_safe": True}
    with pytest.raises(RuntimeError, match="shm_open"):
        e.SharedMemory.open(b)  # move assignment released b's region
    with pytest.raises(RuntimeError, match="shm_open"):
        e.SharedMemory.open(a)  # and everything was released at the end


# ---------------------------------------------------------------- M1: Channel
def _message(i):
    n = (i * 37) % 61 + 1  # 1..61 bytes, every length many times
    return bytes((i + k) % 256 for k in range(n))


def _send_many(region, count):
    c = lib().Channel.open(region)
    for i in range(count):
        c.send(_message(i), timeout=30)
    return count


def _recv_many(region, count):
    c = lib().Channel.open(region)
    crc = 0
    for i in range(count):
        got = c.recv(64, timeout=30)
        if got != _message(i):
            return f"message {i}: got {got[:8]!r}..., length {len(got)}"
        crc = zlib.crc32(got, crc)
    return crc


def test_m1_roundtrip_in_one_process():
    e = lib()
    c = needs(e.Channel.create, name("one"), 4, 64)
    assert (c.slots, c.slot_bytes) == (4, 64)
    for msg in (b"a", b"hello", bytes(range(64))):
        c.send(msg)
        assert c.recv(64) == msg
    c.send(b"")
    assert c.recv(64) == b"", "an empty message is still a message"


def test_m1_validation():
    e = lib()
    with pytest.raises(ValueError):
        needs(e.Channel.create, name("zero"), 0, 64)
    c = needs(e.Channel.create, name("val"), 2, 8)
    with pytest.raises(ValueError):
        c.send(b"123456789")  # longer than a slot
    c.send(b"12345678")
    with pytest.raises(ValueError):
        c.recv(4)  # does not fit: std::length_error...
    assert c.recv(8) == b"12345678"  # ...and the message stayed
    region = name("notachannel")
    s = e.SharedMemory.create(region, 4096)
    with pytest.raises(RuntimeError):
        e.Channel.open(region)  # wrong magic number
    s.close()


def test_m1_full_and_empty_block_until_timeout():
    e = lib()
    c = needs(e.Channel.create, name("block"), 2, 8)
    t = time.monotonic()
    with pytest.raises(TimeoutError):
        c.recv(8, timeout=0.2)
    assert 0.15 < time.monotonic() - t < 2.0
    c.send(b"1")
    c.send(b"2")
    with pytest.raises(TimeoutError):
        c.send(b"3", timeout=0.2)  # full
    assert c.recv(8) == b"1"
    c.send(b"3")  # room again
    assert [c.recv(8), c.recv(8)] == [b"2", b"3"]


@pytest.mark.parametrize("slots", [1, 4])
def test_m1_many_messages_between_processes(slots):
    e = lib()
    region = name(f"many{slots}")
    keep = needs(e.Channel.create, region, slots, 64)  # the creator keeps the name alive
    count = 20_000
    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        r = pool.apply_async(_recv_many, (region, count))
        s = pool.apply_async(_send_many, (region, count))
        assert s.get(timeout=120) == count
        got = r.get(timeout=120)
    expect = 0
    for i in range(count):
        expect = zlib.crc32(_message(i), expect)
    assert got == expect, got
    del keep
