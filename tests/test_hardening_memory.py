"""Tests: SafetensorsWeightSource memory-bound invariant.

Proves iter_layers() is lazy with peak memory bounded by largest tensor,
never by whole checkpoint. Records byte-range reads to assert:

1. Returns real generator (no eager work)
2. Tensor k+1 not fetched until past k (lazy pull)
3. At most one tensor buffer in memory at a time
4. Consuming prefix never reads suffix bytes
5. Invariants hold for single-file and sharded checkpoints
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
    """Ground-truth tensor order as physically laid out in file."""
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

    assert recorder.calls == []


def test_tensor_data_is_fetched_lazily_one_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    assert _data_call_lengths(recorder.calls) == []

    first = next(gen)
    assert first.name == order[0]
    lens_after_first = _data_call_lengths(recorder.calls)
    assert lens_after_first == [tensors[order[0]].nbytes]
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

    assert max(data_lens) == largest_tensor_bytes
    assert all(length <= largest_tensor_bytes for length in data_lens)
    assert sorted(data_lens) == sorted(t.nbytes for t in tensors.values())


def test_at_most_one_tensor_buffer_len_recorded_between_yields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    total_bytes_all = sum(t.nbytes for t in tensors.values())
    assert sum(data_lens) < total_bytes_all
    remaining_expected = total_bytes_all - consumed_bytes
    assert remaining_expected > 0
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

    def _tracking_init(
        self: ByteRangeReader,
        uri: str,
        storage_options: dict[str, object] | None = None,
    ) -> None:
        reads.append((-1, -1))
        original_init(self, uri, storage_options)

    ByteRangeReader.__init__ = _tracking_init  # type: ignore[method-assign]
    try:
        gen = SafetensorsWeightSource(path).iter_layers()
        assert inspect.isgenerator(gen)
        assert reads == []
    finally:
        ByteRangeReader.__init__ = original_init  # type: ignore[method-assign]
