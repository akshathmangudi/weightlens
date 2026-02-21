from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")

_SENTINEL = object()


class PrefetchIterator(Generic[T]):
    """Wrap an iterator so the next item is read in a background thread.

    The GIL is released during I/O syscalls, so a single prefetch thread
    achieves true overlap between I/O (reading the next tensor) and
    compute (stats on the current tensor).  Memory cost: one extra item
    buffered in the queue.
    """

    def __init__(self, source: Iterator[T]) -> None:
        self._source = source
        self._queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def _producer(self) -> None:
        try:
            for item in self._source:
                self._queue.put(item)
            self._queue.put(_SENTINEL)
        except BaseException as exc:
            self._queue.put(exc)

    def __iter__(self) -> PrefetchIterator[T]:
        return self

    def __next__(self) -> T:
        item = self._queue.get()
        if item is _SENTINEL:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]

    def close(self) -> None:
        """Drain the queue so the producer thread can exit."""
        # Keep draining until the producer finishes.  The producer may
        # be blocked on queue.put() so we must keep pulling items.
        while self._thread.is_alive():
            with contextlib.suppress(queue.Empty):
                self._queue.get(timeout=0.05)
        self._thread.join(timeout=5.0)

    def __enter__(self) -> PrefetchIterator[T]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
