from __future__ import annotations

from collections.abc import Iterator

from weightlens.prefetch import PrefetchIterator


def test_order_preserved() -> None:
    items = list(range(100))
    result = list(PrefetchIterator(iter(items)))
    assert result == items


def test_empty_iterator() -> None:
    result: list[object] = list(PrefetchIterator(iter([])))
    assert result == []


def test_exception_propagated() -> None:
    def _failing() -> Iterator[int]:
        yield 1
        raise ValueError("boom")

    pf = PrefetchIterator(_failing())
    assert next(pf) == 1
    try:
        next(pf)
        assert False, "Should have raised"  # noqa: B011
    except ValueError as exc:
        assert "boom" in str(exc)


def test_close_on_early_exit() -> None:
    items = list(range(1000))
    pf = PrefetchIterator(iter(items))
    first = next(pf)
    assert first == 0
    pf.close()
    # Thread should be cleaned up (join succeeded in close)
    assert not pf._thread.is_alive()


def test_context_manager() -> None:
    items = [1, 2, 3]
    with PrefetchIterator(iter(items)) as pf:
        result = list(pf)
    assert result == items
