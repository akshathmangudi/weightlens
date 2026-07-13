from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np

from weightlens.contracts import WeightSource
from weightlens.formats.safetensors_header import (
    HEADER_LEN_BYTES,
    TensorSlice,
    decode_float_tensor,
    parse_header,
    read_header_len,
)
from weightlens.formats.safetensors_index import parse_index
from weightlens.io.byte_range import ByteRangeReader
from weightlens.io.uri import join_uri, parent_uri
from weightlens.models import LayerTensor

logger = logging.getLogger(__name__)

_NP_LE: dict[str, np.dtype] = {
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
    "F64": np.dtype(np.float64),
}


class SafetensorsWeightSource(WeightSource):
    """Stream single-file or HF-sharded safetensors as LayerTensor objects.

    Local files can use memory-mapped views (zero-copy) when
    *use_mmap* is ``True``.  Remote URIs always use byte-range reads
    through fsspec.
    """

    def __init__(
        self,
        uri: str,
        storage_options: dict[str, object] | None = None,
        *,
        use_mmap: bool = False,
    ) -> None:
        self._uri = uri
        self._storage_options = storage_options
        self._use_mmap = use_mmap

    def _is_index(self) -> bool:
        return self._uri.endswith(".index.json")

    @staticmethod
    def _is_local(uri: str) -> bool:
        return "://" not in uri

    def _reader(self, uri: str) -> ByteRangeReader:
        return ByteRangeReader(uri, self._storage_options)

    @staticmethod
    def _load_header_bytes(uri: str) -> tuple[int, bytes, str]:
        """Return (header_len, header_bytes, uri) for mmap-based loading."""
        with open(uri, "rb") as f:  # noqa: ASYNC230
            raw = f.read(HEADER_LEN_BYTES)
            header_len = read_header_len(raw)
            f.seek(0)
            header_bytes = f.read(HEADER_LEN_BYTES + header_len)
        return header_len, header_bytes, uri

    def _stream_mmap(
        self, uri: str, wanted: set[str] | None
    ) -> Iterator[LayerTensor]:
        header_len, header_bytes, resolved = self._load_header_bytes(uri)
        _, slices = parse_header(header_bytes)
        for name, sl in slices.items():
            if wanted is not None and name not in wanted:
                continue
            if not sl.is_float:
                logger.debug("Skipping non-float tensor %s (dtype=%s).", name, sl.dtype)
                continue
            offset, _length = sl.absolute_range(header_len)
            dtype = _NP_LE.get(sl.dtype)
            if dtype is None:
                logger.debug("Skipping unsupported dtype %s for %s.", sl.dtype, name)
                continue
            arr = np.memmap(
                resolved, dtype=dtype, mode="r", offset=offset, shape=sl.shape
            )
            logger.debug("Yielding %s shape=%s dtype=%s.", name, sl.shape, arr.dtype)
            yield LayerTensor(
                name=name, values=arr, shape=sl.shape, dtype=str(arr.dtype)
            )

    @staticmethod
    def _load_header(reader: ByteRangeReader) -> tuple[int, dict[str, TensorSlice]]:
        try:
            header_len = read_header_len(reader.read(0, HEADER_LEN_BYTES))
            header_bytes = reader.read(0, HEADER_LEN_BYTES + header_len)
        except ValueError as exc:
            if "Short read" in str(exc):
                raise ValueError("truncated safetensors file") from exc
            raise
        return parse_header(header_bytes)

    def _stream_reader(
        self, reader: ByteRangeReader, wanted: set[str] | None
    ) -> Iterator[LayerTensor]:
        header_len, slices = self._load_header(reader)
        for name, sl in slices.items():
            if wanted is not None and name not in wanted:
                continue
            if not sl.is_float:
                logger.debug("Skipping non-float tensor %s (dtype=%s).", name, sl.dtype)
                continue
            offset, length = sl.absolute_range(header_len)
            raw = reader.read(offset, length)
            values = decode_float_tensor(raw, sl)
            logger.debug("Yielding %s shape=%s dtype=%s.", name, sl.shape, values.dtype)
            yield LayerTensor(
                name=name, values=values, shape=sl.shape, dtype=str(values.dtype)
            )

    def _iter_single(self, uri: str) -> Iterator[LayerTensor]:
        if self._use_mmap and self._is_local(uri):
            yield from self._stream_mmap(uri, wanted=None)
        else:
            yield from self._stream_reader(self._reader(uri), wanted=None)

    def _iter_sharded(self) -> Iterator[LayerTensor]:
        index_reader = self._reader(self._uri)
        weight_map = parse_index(index_reader.read(0, index_reader.size()))
        base = parent_uri(self._uri)

        shards: dict[str, list[str]] = {}
        for tensor_name, shard_file in weight_map.items():
            shards.setdefault(shard_file, []).append(tensor_name)

        for shard_file, tensor_names in shards.items():
            shard_uri = join_uri(base, shard_file)
            logger.info(
                "Streaming shard %s (%d tensors).", shard_file, len(tensor_names)
            )
            if self._use_mmap and self._is_local(shard_uri):
                yield from self._stream_mmap(
                    shard_uri, wanted=set(tensor_names)
                )
            else:
                yield from self._stream_reader(
                    self._reader(shard_uri), wanted=set(tensor_names)
                )

    def iter_layers(self) -> Iterator[LayerTensor]:
        if self._is_index():
            yield from self._iter_sharded()
        else:
            yield from self._iter_single(self._uri)
