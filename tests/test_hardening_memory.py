"""Adversarial hardening tests: streaming/memory-bound invariant.

Proves that ``SafetensorsWeightSource.iter_layers()`` is a lazy generator
whose peak memory is bounded by the largest single tensor in the
checkpoint, never by the whole checkpoint. We do this by monkeypatching
``ByteRangeReader.read`` to record the size and order of every byte-range
read performed while the caller pulls items off the iterator one at a
time, and asserting that:

  1. ``iter_layers()`` returns a real generator (no eager work at call
     time beyond generator construction).
  2. Tensor data for item ``k + 1`` is not fetched until the caller has
     advanced *past* item ``k`` (lazy pull, not eager push).
  3. At any point in the iteration, at most one tensor's data buffer has
     been read that hasn't yet been "released" (i.e. yielded and moved
     past) -- the source never holds two tensors' raw bytes at once.
  4. Consuming a strict prefix of the tensors never triggers reads whose
     sizes correspond to the untouched suffix.
  5. These invariants hold for both single-file and HF-sharded
     checkpoints, including across shard boundaries.
"""

from __future__ import annotations

import inspect
import json
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures_safetensors import write_sharded, write_single
from weightlens.io.byte_range import ByteRangeReader
from weightlens.models import LayerTensor
from weightlens.sources.safetensors import SafetensorsWeightSource

HEADER_LEN_BYTES = 8


def _header_order(path: str) -> list[str]:
    """Ground-truth tensor order as physically laid out in the file.

    ``safetensors.torch.save_file`` does not preserve dict insertion
    order (it lays tensors out by internal size/order heuristics), so
    tests must not assume "small"/"medium"/"large" style names come out
    in a particular sequence. Read the real on-disk header to get the
    authoritative order that ``SafetensorsWeightSource`` will iterate.
    """
    with open(path, "rb") as f:
        raw = f.read()
    (header_len,) = struct.unpack("<Q", raw[:HEADER_LEN_BYTES])
    header = json.loads(raw[HEADER_LEN_BYTES : HEADER_LEN_BYTES + header_len])
    return [name for name in header if name != "__metadata__"]


class ReadRecorder:
    """Records (offset, length) of every ByteRangeReader.read() call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def wrap(
        self, original: Callable[[ByteRangeReader, int, int], bytes]
    ) -> Callable[[ByteRangeReader, int, int], bytes]:
        def _recording_read(
            reader_self: ByteRangeReader, offset: int, length: int
        ) -> bytes:
            data = original(reader_self, offset, length)
            self.calls.append((offset, length))
            return data

        return _recording_read


def _install_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> ReadRecorder:
    recorder = ReadRecorder()
    original = ByteRangeReader.read
    monkeypatch.setattr(
        "weightlens.io.byte_range.ByteRangeReader.read",
        recorder.wrap(original),
    )
    return recorder


def _is_header_call(offset: int, length: int) -> bool:
    """Header reads always start at offset 0 (length-prefix or full header)."""
    return offset == 0


def _data_call_lengths(calls: list[tuple[int, int]]) -> list[int]:
    """Byte-range reads for tensor payloads: everything after offset 0."""
    return [length for offset, length in calls if not _is_header_call(offset, length)]


def test_iter_layers_returns_a_generator(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {
        "w": np.random.randn(4, 4).astype(np.float32),
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)

    result = SafetensorsWeightSource(path).iter_layers()
    assert inspect.isgenerator(result)


def test_calling_iter_layers_does_not_read_any_bytes_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensors: dict[str, np.ndarray] = {
        "a": np.random.randn(4, 4).astype(np.float32),
        "b": np.random.randn(4, 4).astype(np.float32),
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)
    recorder = _install_recorder(monkeypatch)

    SafetensorsWeightSource(path).iter_layers()

    # Generator construction must not touch the reader at all: no header
    # read, no data read. Real work only happens once the generator body
    # starts executing via next()/for.
    assert recorder.calls == []


def test_tensor_data_is_fetched_lazily_one_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Three distinctly-sized tensors so we can identify which read
    # corresponds to which tensor purely from its byte length. Note:
    # safetensors.torch.save_file does not preserve dict insertion
    # order, so we derive the true on-disk order from the header
    # instead of assuming names come out as written.
    tensors: dict[str, np.ndarray] = {
        "small": np.random.randn(2, 2).astype(np.float32),  # 16 bytes
        "medium": np.random.randn(8, 8).astype(np.float32),  # 256 bytes
        "large": np.random.randn(32, 32).astype(np.float32),  # 4096 bytes
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)
    order = _header_order(path)
    assert set(order) == set(tensors)
    recorder = _install_recorder(monkeypatch)

    gen = SafetensorsWeightSource(path).iter_layers()

    # Nothing read yet -- generator hasn't started.
    assert _data_call_lengths(recorder.calls) == []

    first = next(gen)
    assert first.name == order[0]
    lens_after_first = _data_call_lengths(recorder.calls)
    assert lens_after_first == [tensors[order[0]].nbytes]
    # The remaining tensors' bytes must NOT have been read yet.
    assert tensors[order[1]].nbytes not in lens_after_first[1:]
    remaining_after_first = {tensors[order[1]].nbytes, tensors[order[2]].nbytes}
    assert not remaining_after_first.issubset(set(lens_after_first))

    second = next(gen)
    assert second.name == order[1]
    lens_after_second = _data_call_lengths(recorder.calls)
    assert lens_after_second == [
        tensors[order[0]].nbytes,
        tensors[order[1]].nbytes,
    ]
    assert tensors[order[2]].nbytes not in lens_after_second

    third = next(gen)
    assert third.name == order[2]
    lens_after_third = _data_call_lengths(recorder.calls)
    assert lens_after_third == [
        tensors[order[0]].nbytes,
        tensors[order[1]].nbytes,
        tensors[order[2]].nbytes,
    ]

    with pytest.raises(StopIteration):
        next(gen)


def test_consuming_one_item_does_not_read_all_tensor_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensors: dict[str, np.ndarray] = {
        f"t{i}": np.random.randn(16, 16).astype(np.float32) for i in range(10)
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)
    recorder = _install_recorder(monkeypatch)

    gen = SafetensorsWeightSource(path).iter_layers()
    first = next(gen)

    data_lens = _data_call_lengths(recorder.calls)
    # Exactly one tensor's worth of data has been fetched so far.
    assert len(data_lens) == 1
    assert data_lens[0] == tensors[first.name].nbytes

    total_bytes_all_tensors = sum(t.nbytes for t in tensors.values())
    assert sum(data_lens) < total_bytes_all_tensors


def test_peak_single_read_never_exceeds_largest_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensors: dict[str, np.ndarray] = {
        "tiny": np.random.randn(2).astype(np.float32),
        "huge": np.random.randn(64, 64).astype(np.float32),
        "tiny2": np.random.randn(3).astype(np.float32),
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)
    recorder = _install_recorder(monkeypatch)

    consumed = list(SafetensorsWeightSource(path).iter_layers())
    assert {lt.name for lt in consumed} == set(tensors)

    data_lens = _data_call_lengths(recorder.calls)
    largest_tensor_bytes = max(t.nbytes for t in tensors.values())

    # No single byte-range read for tensor payload ever exceeds the
    # largest tensor in the checkpoint -- proves boundedness by max
    # tensor size, not total checkpoint size.
    assert max(data_lens) == largest_tensor_bytes
    assert all(length <= largest_tensor_bytes for length in data_lens)

    # Sum across all reads equals total bytes exactly once each
    # (no re-reads, no over-fetching, no duplication).
    assert sorted(data_lens) == sorted(t.nbytes for t in tensors.values())


def test_at_most_one_tensor_buffer_len_recorded_between_yields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After each next(), exactly one *new* data read has occurred.

    This directly proves the source never holds more than one tensor's
    raw byte buffer at a time: between two consecutive advances of the
    iterator, precisely one additional byte-range read for tensor
    payload appears in the trace.
    """
    tensors: dict[str, np.ndarray] = {
        f"layer{i}": np.random.randn(6, 6).astype(np.float32) for i in range(5)
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)
    recorder = _install_recorder(monkeypatch)

    gen = SafetensorsWeightSource(path).iter_layers()
    previous_count = 0
    for _ in range(len(tensors)):
        next(gen)
        current_count = len(_data_call_lengths(recorder.calls))
        assert current_count == previous_count + 1
        previous_count = current_count


def test_sharded_lazy_across_shard_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard_a: dict[str, np.ndarray] = {
        "model.layers.0.w": np.random.randn(4, 4).astype(np.float32),
    }
    shard_b: dict[str, np.ndarray] = {
        "model.layers.1.w": np.random.randn(10, 10).astype(np.float32),
    }
    index_path = write_sharded(str(tmp_path / "ckpt"), [shard_a, shard_b])
    recorder = _install_recorder(monkeypatch)

    gen = SafetensorsWeightSource(index_path).iter_layers()

    first = next(gen)
    assert first.name == "model.layers.0.w"
    lens_after_first = _data_call_lengths(recorder.calls)
    assert lens_after_first == [shard_a["model.layers.0.w"].nbytes]
    # Shard B's tensor bytes must not have been fetched merely because
    # we advanced past shard A's only tensor -- only pulling the next
    # item should open/read shard B.
    assert shard_b["model.layers.1.w"].nbytes not in lens_after_first

    second = next(gen)
    assert second.name == "model.layers.1.w"
    lens_after_second = _data_call_lengths(recorder.calls)
    assert lens_after_second == [
        shard_a["model.layers.0.w"].nbytes,
        shard_b["model.layers.1.w"].nbytes,
    ]


def test_partial_consumption_of_many_tensors_reads_only_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_tensors = 20
    tensors: dict[str, np.ndarray] = {
        f"t{i:02d}": np.random.randn(4, 4).astype(np.float32) for i in range(n_tensors)
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)
    recorder = _install_recorder(monkeypatch)

    gen = SafetensorsWeightSource(path).iter_layers()
    take = 3
    consumed: list[LayerTensor] = []
    for _ in range(take):
        consumed.append(next(gen))

    data_lens = _data_call_lengths(recorder.calls)
    assert len(data_lens) == take
    consumed_bytes = sum(lt.values.nbytes for lt in consumed)
    assert sum(data_lens) == consumed_bytes

    # The remaining (n_tensors - take) tensors' bytes were never read.
    total_bytes_all = sum(t.nbytes for t in tensors.values())
    assert sum(data_lens) < total_bytes_all
    remaining_expected = total_bytes_all - consumed_bytes
    assert remaining_expected > 0

    # Do not exhaust the generator: prove the reader never fetched more
    # than `take` tensors' worth of payload while only `take` were
    # pulled.
    assert len(data_lens) == take


def test_full_consumption_reads_exactly_once_per_tensor_no_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensors: dict[str, np.ndarray] = {
        "x": np.random.randn(5, 5).astype(np.float32),
        "y": np.random.randn(3, 7).astype(np.float32),
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)
    recorder = _install_recorder(monkeypatch)

    list(SafetensorsWeightSource(path).iter_layers())

    data_lens = _data_call_lengths(recorder.calls)
    # Exactly one data-read per tensor: no double-buffering, no re-reads.
    assert len(data_lens) == len(tensors)
    assert sorted(data_lens) == sorted(t.nbytes for t in tensors.values())


def test_header_is_read_before_any_tensor_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensors: dict[str, np.ndarray] = {
        "only": np.random.randn(4).astype(np.float32),
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)
    recorder = _install_recorder(monkeypatch)

    gen = SafetensorsWeightSource(path).iter_layers()
    next(gen)

    # First two reads are the header-length probe and full-header read
    # (both start at offset 0); the tensor payload read must come after
    # both, never before the header has been fully parsed.
    assert len(recorder.calls) >= 3
    header_calls = [c for c in recorder.calls if _is_header_call(*c)]
    data_calls = [c for c in recorder.calls if not _is_header_call(*c)]
    assert len(header_calls) == 2
    assert len(data_calls) == 1
    first_data_call_index = recorder.calls.index(data_calls[0])
    last_header_call_index = max(
        recorder.calls.index(c) for c in header_calls
    )
    assert first_data_call_index > last_header_call_index


def test_generator_never_started_holds_zero_reads(tmp_path: Path) -> None:
    """Merely obtaining the generator object must not touch storage."""
    tensors: dict[str, np.ndarray] = {
        "w": np.random.randn(4, 4).astype(np.float32),
    }
    path = str(tmp_path / "m.safetensors")
    write_single(path, tensors)

    reads: list[tuple[int, int]] = []
    original_init = ByteRangeReader.__init__

    def _tracking_init(self: ByteRangeReader, uri: str) -> None:
        reads.append((-1, -1))  # sentinel: reader constructed
        original_init(self, uri)

    # Use direct attribute patch/restore (no monkeypatch fixture needed
    # since this test doesn't take it as a param) to keep the file's
    # signature list honest about which tests need MonkeyPatch.
    ByteRangeReader.__init__ = _tracking_init  # type: ignore[method-assign]
    try:
        gen = SafetensorsWeightSource(path).iter_layers()
        assert inspect.isgenerator(gen)
        # Constructing the generator must not construct a reader either:
        # generator function bodies don't run until the first next().
        assert reads == []
    finally:
        ByteRangeReader.__init__ = original_init  # type: ignore[method-assign]
