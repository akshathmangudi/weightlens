from __future__ import annotations

import io
import logging
import os
import pickle
import warnings
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch

from weightlens.contracts import WeightSource
from weightlens.models import LayerTensor
from weightlens.tensor_utils import tensor_to_numpy

logger = logging.getLogger(__name__)

_METADATA_FILENAMES = (".metadata", "metadata")
"""Ordered list of metadata filenames to probe.

Standard PyTorch DCP uses ``.metadata``; some Megatron-LM checkpoints
write ``metadata`` (no dot prefix).
"""

_FLOAT_DTYPES = frozenset({torch.float16, torch.float32, torch.float64, torch.bfloat16})


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


_UNSAFE_MODULES = frozenset({
    "os", "subprocess", "builtins", "socket", "shutil",
    "posix", "nt", "sys", "ctypes", "importlib",
    "pdb", "code", "compile", "pty",
})


class _SafeMetadataUnpickler(pickle.Unpickler):
    """Unpickler that blocks unsafe modules and stubs missing ones.

    DCP metadata files are pickled Python objects.  When a checkpoint was
    saved by a framework like Megatron-LM, the pickle may reference
    classes from ``megatron.core`` or similar packages that are not
    installed in the analysis environment.  This unpickler creates
    lightweight stub classes on the fly so the metadata can still be
    loaded.

    Potentially dangerous modules (os, subprocess, builtins, etc.) are
    explicitly blocked regardless of availability.
    """

    def find_class(self, module: str, name: str) -> type:
        if module in _UNSAFE_MODULES:
            raise pickle.UnpicklingError(
                f"Blocked import of potentially unsafe module: {module}.{name}"
            )
        try:
            cls: type = super().find_class(module, name)
            return cls
        except (ModuleNotFoundError, AttributeError):
            logger.debug(
                "Stubbing missing class %s.%s during metadata unpickling.",
                module,
                name,
            )
            stub: type = type(name, (), {"__module__": module})
            return stub


def read_metadata(checkpoint_dir: str | Path) -> Any:
    """Read DCP metadata with fault-tolerant unpickling.

    Falls back to a :class:`_SafeMetadataUnpickler` when the standard
    ``reader.read_metadata()`` raises due to missing third-party
    modules (e.g. ``megatron.core``).

    Note: the primary path delegates to ``reader.read_metadata()`` which
    uses torch's internal pickle loader — we cannot control its unpickling
    policy.  The fallback path uses our whitelist-based safe unpickler.
    """
    reader = make_reader(checkpoint_dir)
    try:
        return reader.read_metadata()
    except (ModuleNotFoundError, AttributeError):
        logger.info("Standard metadata read failed; retrying with tolerant unpickler.")
    meta_path = find_metadata_path(checkpoint_dir)
    with open(meta_path, "rb") as f:
        return _SafeMetadataUnpickler(f).load()


def _validate_shard_path(checkpoint_dir: Path, rel_path: str) -> Path:
    resolved = (checkpoint_dir / rel_path).resolve()
    if not str(resolved).startswith(str(checkpoint_dir.resolve())):
        raise ValueError(
            f"Shard path escapes checkpoint directory: {rel_path!r}"
        )
    return resolved


class DCPWeightSource(WeightSource):
    """Stream DCP (Distributed Checkpoint) directories as LayerTensor objects.

    **EXPERIMENTAL**: reads one tensor at a time using direct byte-offset
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

        metadata = read_metadata(self._checkpoint_dir)

        # Build lookup: tensor fqn → list of (chunk_offsets, storage_info)
        # storage_info has .relative_path, .offset, .length attributes.
        storage_lookup: dict[str, list[tuple[torch.Size, Any]]] = defaultdict(list)
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

        # Phase 1: Group chunks by shard file and collect per-tensor metadata.
        shard_groups: dict[Path, list[tuple[str, torch.Size, Any]]] = defaultdict(list)
        tensor_meta: dict[str, tuple[torch.Size, torch.dtype, int]] = {}

        for name, storage_meta in tensor_entries:
            dtype = storage_meta.properties.dtype
            if dtype not in _FLOAT_DTYPES:
                logger.debug("Skipping non-float tensor %s (dtype=%s).", name, dtype)
                continue

            chunk_info = storage_lookup.get(name, [])
            if not chunk_info:
                logger.warning("No storage_data entries for tensor %s, skipping.", name)
                continue

            chunk_info.sort(key=lambda item: tuple(item[0]))
            tensor_meta[name] = (torch.Size(storage_meta.size), dtype, len(chunk_info))

            for chunk_offsets, sinfo in chunk_info:
                shard_path = _validate_shard_path(
                    self._checkpoint_dir, sinfo.relative_path
                )
                shard_groups[shard_path].append((name, chunk_offsets, sinfo))

        # Sort chunks within each shard by offset for sequential reads.
        for chunks in shard_groups.values():
            chunks.sort(key=lambda item: item[2].offset)

        logger.info(
            "Starting streaming for DCP checkpoint %s (%d tensors, %d shards).",
            self._checkpoint_dir,
            len(tensor_meta),
            len(shard_groups),
        )

        # Phase 2: Iterate shards, read all chunks with one open per shard.
        pending: dict[str, list[tuple[torch.Size, torch.Tensor]]] = defaultdict(list)
        layer_count = 0

        for shard_path, chunks in shard_groups.items():
            with open(shard_path, "rb") as f:
                for name, chunk_offsets, sinfo in chunks:
                    f.seek(sinfo.offset)
                    chunk_bytes = f.read(sinfo.length)
                    chunk_tensor = torch.load(
                        io.BytesIO(chunk_bytes),
                        weights_only=True,
                        map_location="cpu",
                    )
                    pending[name].append((chunk_offsets, chunk_tensor))

            yielded: set[str] = set()
            for name in list(pending):
                full_shape, dtype, expected = tensor_meta[name]
                if len(pending[name]) != expected:
                    continue

                if expected == 1:
                    tensor = pending[name][0][1]
                else:
                    chunks_for_tensor = pending[name]
                    chunks_for_tensor.sort(key=lambda x: tuple(x[0]))
                    tensor = torch.empty(full_shape, dtype=dtype)
                    for chunk_offsets, chunk_tensor in chunks_for_tensor:
                        slices = tuple(
                            slice(int(off), int(off) + int(dim))
                            for off, dim in zip(
                                chunk_offsets, chunk_tensor.shape, strict=True
                            )
                        )
                        tensor[slices] = chunk_tensor
                    for _, ct in chunks_for_tensor:
                        del ct

                values = tensor_to_numpy(tensor)
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
                yielded.add(name)
                del tensor

            for name in yielded:
                del pending[name]

        logger.info(
            "Completed streaming for DCP checkpoint %s (layers=%d).",
            self._checkpoint_dir,
            layer_count,
        )
