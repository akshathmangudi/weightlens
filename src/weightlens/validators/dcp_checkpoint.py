from __future__ import annotations

import logging
from pathlib import Path

from weightlens.contracts import CheckpointValidator
from weightlens.models import CheckpointHealth

logger = logging.getLogger(__name__)

_METADATA_FILENAMES = (".metadata", "metadata")


class DCPCheckpointValidator(CheckpointValidator):
    """Validate DCP (Distributed Checkpoint) directory integrity.

    **EXPERIMENTAL** — performs metadata-only validation without loading
    tensor data.  NaN / zero-flood detection is handled by the analysis
    pipeline (``BasicStatsEngine``, ``DeadLayerRule``, ``ExtremeSpikeRule``).
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(self) -> CheckpointHealth:
        corruption_flags: list[str] = []

        # Phase 1: directory existence
        if not self._checkpoint_dir.is_dir():
            corruption_flags.append("not_a_directory")
            logger.error("%s is not a directory.", self._checkpoint_dir)
            return CheckpointHealth(
                file_size_bytes=0,
                is_empty=True,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=corruption_flags,
            )

        # Phase 2: total directory size
        dir_size = sum(
            f.stat().st_size
            for f in self._checkpoint_dir.rglob("*")
            if f.is_file()
        )
        logger.info(
            "Starting validation for DCP checkpoint %s (dir_size=%d bytes).",
            self._checkpoint_dir,
            dir_size,
        )

        if dir_size == 0:
            corruption_flags.append("empty_directory")
            return CheckpointHealth(
                file_size_bytes=dir_size,
                is_empty=True,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=corruption_flags,
            )

        # Phase 3: metadata file existence + non-zero size
        metadata_path = self._find_metadata()
        if metadata_path is None:
            corruption_flags.append("missing_metadata_file")
            return CheckpointHealth(
                file_size_bytes=dir_size,
                is_empty=False,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=corruption_flags,
            )

        if metadata_path.stat().st_size == 0:
            corruption_flags.append("empty_metadata_file")
            return CheckpointHealth(
                file_size_bytes=dir_size,
                is_empty=False,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=corruption_flags,
            )

        # Phase 4: shard file scan — flag zero-byte shards
        shard_files = list(self._checkpoint_dir.glob("*.distcp"))
        for shard in shard_files:
            if shard.stat().st_size == 0:
                flag = f"zero_byte_shard:{shard.name}"
                corruption_flags.append(flag)
                logger.debug("Flagged %s.", flag)

        # Phase 5: parse metadata
        try:
            from weightlens.sources.dcp import make_reader

            reader = make_reader(self._checkpoint_dir)
            metadata = reader.read_metadata()
        except Exception:
            logger.exception(
                "Failed to read DCP metadata for %s.", self._checkpoint_dir
            )
            corruption_flags.append("metadata_read_failed")
            return CheckpointHealth(
                file_size_bytes=dir_size,
                is_empty=False,
                loadable=False,
                tensor_count=0,
                total_params=0,
                corruption_flags=corruption_flags,
            )

        # Phase 6: count tensors + total params from metadata
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata

        tensor_count = 0
        total_params = 0

        for key, storage_meta in metadata.state_dict_metadata.items():
            if not isinstance(storage_meta, TensorStorageMetadata):
                logger.debug("Non-tensor metadata entry: %s.", key)
                continue
            tensor_count += 1
            numel = 1
            for dim in storage_meta.size:
                numel *= dim
            total_params += numel

            if numel == 0:
                flag = f"empty_tensor:{key}"
                corruption_flags.append(flag)
                logger.debug("Flagged %s.", flag)

        # Phase 7: cross-reference shard files from storage_data
        referenced_files: set[str] = set()
        for _idx, info in metadata.storage_data.items():
            referenced_files.add(info.relative_path)

        for ref_file in sorted(referenced_files):
            ref_path = self._checkpoint_dir / ref_file
            if not ref_path.exists():
                flag = f"missing_shard:{ref_file}"
                corruption_flags.append(flag)
                logger.debug("Flagged %s.", flag)

        is_empty = tensor_count == 0
        loadable = not any(
            f
            in (
                "missing_metadata_file",
                "empty_metadata_file",
                "metadata_read_failed",
            )
            for f in corruption_flags
        ) and not any(f.startswith("missing_shard:") for f in corruption_flags)

        health = CheckpointHealth(
            file_size_bytes=dir_size,
            is_empty=is_empty,
            loadable=loadable,
            tensor_count=tensor_count,
            total_params=total_params,
            corruption_flags=corruption_flags,
        )
        logger.info(
            "Validation completed for %s: %s.",
            self._checkpoint_dir,
            health.model_dump(),
        )
        return health

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_metadata(self) -> Path | None:
        for name in _METADATA_FILENAMES:
            candidate = self._checkpoint_dir / name
            if candidate.exists():
                return candidate
        return None
