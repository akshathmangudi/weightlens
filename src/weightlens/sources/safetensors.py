from __future__ import annotations

import logging
from collections.abc import Iterator

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


class SafetensorsWeightSource(WeightSource):
    """Stream single-file or HF-sharded safetensors as LayerTensor objects.

    Memory is bounded by the largest single tensor: each tensor's bytes are
    fetched by byte-range, decoded, yielded, then released before the next.
    """

    def __init__(self, uri: str) -> None:
        self._uri = uri

    def _is_index(self) -> bool:
        return self._uri.endswith(".index.json")

    @staticmethod
    def _load_header(reader: ByteRangeReader) -> tuple[int, dict[str, TensorSlice]]:
        header_len = read_header_len(reader.read(0, HEADER_LEN_BYTES))
        header_bytes = reader.read(0, HEADER_LEN_BYTES + header_len)
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
        yield from self._stream_reader(ByteRangeReader(uri), wanted=None)

    def _iter_sharded(self) -> Iterator[LayerTensor]:
        index_reader = ByteRangeReader(self._uri)
        weight_map = parse_index(index_reader.read(0, index_reader.size()))
        base = parent_uri(self._uri)

        # Group tensors by shard, preserving first-seen shard order.
        shards: dict[str, list[str]] = {}
        for tensor_name, shard_file in weight_map.items():
            shards.setdefault(shard_file, []).append(tensor_name)

        for shard_file, tensor_names in shards.items():
            shard_uri = join_uri(base, shard_file)
            logger.info(
                "Streaming shard %s (%d tensors).", shard_file, len(tensor_names)
            )
            reader = ByteRangeReader(shard_uri)
            yield from self._stream_reader(reader, wanted=set(tensor_names))

    def iter_layers(self) -> Iterator[LayerTensor]:
        if self._is_index():
            yield from self._iter_sharded()
        else:
            yield from self._iter_single(self._uri)
