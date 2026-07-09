from __future__ import annotations

import logging
from math import prod

from weightlens.contracts import CheckpointValidator
from weightlens.formats.safetensors_header import (
    HEADER_LEN_BYTES,
    TensorSlice,
    parse_header,
    read_header_len,
)
from weightlens.formats.safetensors_index import parse_index
from weightlens.io.byte_range import ByteRangeReader
from weightlens.io.errors import MissingBackendError
from weightlens.io.uri import join_uri, parent_uri
from weightlens.models import CheckpointHealth

logger = logging.getLogger(__name__)


class SafetensorsCheckpointValidator(CheckpointValidator):
    """Validate a safetensors checkpoint using only its header(s)."""

    def __init__(self, uri: str) -> None:
        self._uri = uri

    @staticmethod
    def _header(reader: ByteRangeReader) -> tuple[int, dict[str, TensorSlice]]:
        header_len = read_header_len(reader.read(0, HEADER_LEN_BYTES))
        return parse_header(reader.read(0, HEADER_LEN_BYTES + header_len))

    @staticmethod
    def _check_offsets(
        header_len: int, slices: dict[str, TensorSlice], file_size: int
    ) -> None:
        """Reject a header whose tensor data extends past the end of the file."""
        base = HEADER_LEN_BYTES + header_len
        for s in slices.values():
            if base + s.end > file_size:
                raise ValueError(
                    f"tensor {s.name!r} data extends past end of file "
                    f"(needs {base + s.end} bytes, file is {file_size})"
                )

    def _validate_single(self, uri: str) -> tuple[int, int, int]:
        reader = ByteRangeReader(uri)
        size = reader.size()
        header_len, slices = self._header(reader)
        self._check_offsets(header_len, slices, size)
        floats = [s for s in slices.values() if s.is_float]
        params = sum(prod(s.shape) if s.shape else 1 for s in floats)
        return size, len(floats), params

    def _validate_sharded(self) -> tuple[int, int, int]:
        index_reader = ByteRangeReader(self._uri)
        weight_map = parse_index(index_reader.read(0, index_reader.size()))
        base = parent_uri(self._uri)
        shard_files = sorted(set(weight_map.values()))
        size = count = params = 0
        for shard_file in shard_files:
            reader = ByteRangeReader(join_uri(base, shard_file))
            shard_size = reader.size()
            header_len, slices = self._header(reader)
            self._check_offsets(header_len, slices, shard_size)
            floats = [s for s in slices.values() if s.is_float]
            size += shard_size
            count += len(floats)
            params += sum(prod(s.shape) if s.shape else 1 for s in floats)
        return size, count, params

    def validate(self) -> CheckpointHealth:
        try:
            if self._uri.endswith(".index.json"):
                size, count, params = self._validate_sharded()
            else:
                size, count, params = self._validate_single(self._uri)
        except (FileNotFoundError, MissingBackendError):
            # A genuinely missing file, or an uninstalled remote backend, must
            # propagate (the CLI maps MissingBackendError to a distinct exit
            # code + install hint) rather than be misreported as corruption.
            raise
        except Exception as exc:  # unparseable header / bad index
            logger.error("Safetensors validation failed: %s", exc)
            return CheckpointHealth(
                file_size_bytes=0,
                is_empty=True,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=[f"unreadable: {exc}"],
            )
        return CheckpointHealth(
            file_size_bytes=size,
            is_empty=(count == 0),
            loadable=True,
            tensor_count=count,
            total_params=params,
            corruption_flags=[],
        )
