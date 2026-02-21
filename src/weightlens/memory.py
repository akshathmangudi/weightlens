from __future__ import annotations

import os


def available_memory_bytes() -> int:
    """Return available system memory in bytes.

    Tries ``psutil`` first, then ``/proc/meminfo``, and falls back
    to a conservative 2 GiB default.
    """
    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except Exception:  # noqa: BLE001
        pass

    return 2 * 1024 * 1024 * 1024  # 2 GiB


_AUTO_MAX_WORKERS = 4
"""Hard ceiling for auto-detection to prevent OOM on large checkpoints."""


def compute_max_workers(
    avg_tensor_bytes: int = 325 * 1024 * 1024,
    memory_fraction: float = 0.25,
    max_workers: int | None = None,
) -> int:
    """Compute a safe number of parallel workers.

    Each worker holds one tensor buffer in flight, so the memory budget
    is ``available * memory_fraction / avg_tensor_bytes``.  The result
    is clamped to ``[1, min(cpu_count, _AUTO_MAX_WORKERS)]`` and
    optionally capped by *max_workers*.

    The *memory_fraction* default (0.25) is intentionally conservative
    because the actual tensor sizes are unknown at call time and the
    prefetch buffer adds an extra tensor to memory.
    """
    mem = available_memory_bytes()
    if avg_tensor_bytes > 0:
        budget = int(mem * memory_fraction / avg_tensor_bytes)
    else:
        budget = 1
    cpu_count = os.cpu_count() or 1
    workers = max(1, min(budget, cpu_count, _AUTO_MAX_WORKERS))
    if max_workers is not None:
        workers = min(workers, max(1, max_workers))
    return workers
