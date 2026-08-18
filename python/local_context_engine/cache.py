"""Small in-memory TTL/LRU caches for retrieval and compression results.

A real production deployment can replace this with Redis or a disk cache; the
interface is intentionally tiny.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Hashable, Optional


class TTLCache:
    """A minimal TTL + LRU cache.

    Items are evicted when they expire or when the cache exceeds ``maxsize``.
    """

    def __init__(self, maxsize: int = 128, ttl_seconds: float = 60.0) -> None:
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._data: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._expires: dict = {}

    def get(self, key: Hashable) -> Optional[Any]:
        if key not in self._data:
            return None
        if time.time() > self._expires.get(key, 0):
            self._drop(key)
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: Hashable, value: Any) -> None:
        self._data[key] = value
        self._expires[key] = time.time() + self.ttl_seconds
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            oldest, _ = self._data.popitem(last=False)
            self._expires.pop(oldest, None)

    def clear(self) -> None:
        self._data.clear()
        self._expires.clear()

    def __len__(self) -> int:
        return len(self._data)

    def _drop(self, key: Hashable) -> None:
        self._data.pop(key, None)
        self._expires.pop(key, None)
