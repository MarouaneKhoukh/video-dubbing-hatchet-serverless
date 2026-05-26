"""Chunking helpers for fan-out fan-out — used by the Hatchet workflow to split
a stage's inputs into ``batch_size`` chunks before dispatching parallel Nebius
jobs. Pure utility; no Hatchet/Nebius coupling.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def chunk(items: list[T], batch_size: int) -> list[list[T]]:
    """Split a list into consecutive chunks of at most ``batch_size`` items.

    Order-preserving. Empty input → empty list. ``batch_size`` must be ≥ 1.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not items:
        return []
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
