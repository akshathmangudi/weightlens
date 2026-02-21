from __future__ import annotations

import io
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch

from weightlens.contracts import WeightSource
from weightlens.models import LayerTensor

logger = logging.getLogger(__name__)

_METADATA_FILENAMES = (".metadata", "metadata")
"""Ordered list of metadata filenames to probe.

Standard PyTorch DCP uses ``.metadata``; some Megatron-LM checkpoints
write ``metadata`` (no dot prefix).
"""

_FLOAT_DTYPES = frozenset(
    {torch.float16, torch.float32, torch.float64, torch.bfloat16}
)


def find_metadata_path(checkpoint_dir: str | Path) -> Path:
    """Return the first existing metadata file in *checkpoint_dir*.

    Raises ``FileNotFoundError`` if none of the known filenames exist.
    """
    checkpoint_dir = Path(checkpoint_dir)
    for name in _METADATA_FILENAMES:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No metadata file found in {checkpoint_dir}. "
        f"Looked for: {', '.join(_METADATA_FILENAMES)}"
    )


def make_reader(checkpoint_dir: str | Path) -> Any:
    """Build a ``FileSystemReader`` that can handle both naming conventions.

    If the directory uses the non-standard ``metadata`` filename, the reader
    is patched so that ``read_metadata()`` finds it.
    """
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    checkpoint_dir = Path(checkpoint_dir)
    meta_path = find_metadata_path(checkpoint_dir)
    reader = FileSystemReader(str(checkpoint_dir))

    if meta_path.name != ".metadata":
        actual_name = meta_path.name

        def _patched_get_metadata_path(
            rank: int | None = None,
        ) -> os.PathLike[str]:
            filename = actual_name if rank is None else f"__{rank}{actual_name}"
            return cast(Path, reader.fs.concat_path(reader.path, filename))

        reader._get_metadata_path = _patched_get_metadata_path  # type: ignore[method-assign]
        logger.debug(
            "Patched FileSystemReader to use non-standard metadata filename %r.",
            actual_name,
        )

    return reader


def _find_shard_dim(offsets_list: list[torch.Size]) -> int:
    """Return the dimension along which chunks are sharded.

    Compares chunk offsets to find the first dimension where values differ.
    Falls back to dim 0 if all offsets are identical (shouldn't happen for
    multi-chunk tensors, but safe default).
    """
    if len(offsets_list) <= 1 or len(offsets_list[0]) == 0:
        return 0
    for dim in range(len(offsets_list[0])):
        vals = {o[dim] for o in offsets_list}
        if len(vals) > 1:
            return dim
    return 0


class DCPWeightSource(WeightSource):
    """Stream DCP (Distributed Checkpoint) directories as LayerTensor objects.

    **EXPERIMENTAL** — reads one tensor at a time using direct byte-offset
    reads from shard files.  Memory usage is bounded by the size of the
    largest single tensor, not the total checkpoint size.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)

    def iter_layers(self) -> Iterator[LayerTensor]:  # noqa: C901
        from torch.distributed.checkpoint.metadata import (
            MetadataIndex,
            TensorStorageMetadata,
        )

        warnings.warn(
            "DCPWeightSource is EXPERIMENTAL.",
            stacklevel=2,
        )

        metadata = make_reader(self._checkpoint_dir).read_metadata()

        # Build lookup: tensor fqn → list of (chunk_offsets, storage_info)
        # storage_info has .relative_path, .offset, .length attributes.
        storage_lookup: dict[str, list[tuple[torch.Size, Any]]] = (
            defaultdict(list)
        )
        for idx, info in metadata.storage_data.items():
            if not isinstance(idx, MetadataIndex):
                continue
            if idx.offset is None:
                continue
            storage_lookup[idx.fqn].append((idx.offset, info))

        # Collect tensor entries from metadata.
        tensor_entries: list[tuple[str, TensorStorageMetadata]] = []
        for key, storage_meta in metadata.state_dict_metadata.items():
            if not isinstance(storage_meta, TensorStorageMetadata):
                logger.debug("Skipping non-tensor metadata entry %s.", key)
                continue
            tensor_entries.append((key, storage_meta))

        if not tensor_entries:
            logger.warning(
                "DCP checkpoint %s contains no tensor entries.",
                self._checkpoint_dir,
            )
            return

        logger.info(
            "Starting streaming for DCP checkpoint %s (%d tensors).",
            self._checkpoint_dir,
            len(tensor_entries),
        )
        layer_count = 0

        for name, storage_meta in tensor_entries:
            dtype = storage_meta.properties.dtype

            # Skip non-float tensors before loading data.
            if dtype not in _FLOAT_DTYPES:
                logger.debug(
                    "Skipping non-float tensor %s (dtype=%s).", name, dtype
                )
                continue

            # Read chunks via byte-offset seeks.
            chunk_info = storage_lookup.get(name, [])
            if not chunk_info:
                logger.warning(
                    "No storage_data entries for tensor %s, skipping.", name
                )
                continue

            # Sort chunks by their offset tuple for deterministic ordering.
            chunk_info.sort(key=lambda item: tuple(item[0]))

            loaded_chunks: list[tuple[torch.Size, torch.Tensor]] = []
            for chunk_offsets, sinfo in chunk_info:
                shard_path = self._checkpoint_dir / sinfo.relative_path
                with open(shard_path, "rb") as f:
                    f.seek(sinfo.offset)
                    chunk_bytes = f.read(sinfo.length)
                chunk_tensor = torch.load(
                    io.BytesIO(chunk_bytes),
                    weights_only=True,
                    map_location="cpu",
                )
                loaded_chunks.append((chunk_offsets, chunk_tensor))
                del chunk_bytes

            # Reconstruct the full tensor from chunks.
            if len(loaded_chunks) == 1:
                tensor = loaded_chunks[0][1]
            else:
                shard_dim = _find_shard_dim(
                    [offsets for offsets, _ in loaded_chunks]
                )
                # Chunks are already sorted by offset.
                tensor = torch.cat(
                    [t for _, t in loaded_chunks], dim=shard_dim
                )
                # Free individual chunks.
                for _, chunk_t in loaded_chunks:
                    del chunk_t

            materialized = tensor.detach().cpu().contiguous()
            values = materialized.reshape(-1).float().numpy()
            layer_count += 1
            logger.debug(
                "Yielding layer %s shape=%s dtype=%s param_count=%d.",
                name,
                tuple(tensor.shape),
                values.dtype,
                int(values.size),
            )

            yield LayerTensor(
                name=str(name),
                values=values,
                shape=tuple(tensor.shape),
                dtype=str(values.dtype),
            )

            # Eagerly free the loaded tensor.
            del loaded_chunks, tensor, materialized

        logger.info(
            "Completed streaming for DCP checkpoint %s (layers=%d).",
            self._checkpoint_dir,
            layer_count,
        )
