"""Thread-safe in-process TTL cache used for analytics results."""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from functools import lru_cache
from typing import Any, TypeVar

from app.core.config import get_settings

T = TypeVar("T")


class TTLCache:
    """Small LRU cache whose entries expire after ``ttl_seconds``."""

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_set(self, key: Hashable, factory: Callable[[], T]) -> T:
        """Return the cached value for ``key`` or compute and store it."""
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] > now:
                self._entries.move_to_end(key)
                self.hits += 1
                return entry[1]
            self.misses += 1
        value = factory()
        with self._lock:
            self._entries[key] = (self._clock() + self._ttl, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


@lru_cache(maxsize=1)
def get_analytics_cache() -> TTLCache:
    """Process-wide cache shared by all analytics requests."""
    settings = get_settings()
    return TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries)


def invalidate_analytics_cache() -> None:
    """Drop cached analytics after the underlying data changed."""
    if get_analytics_cache.cache_info().currsize:
        get_analytics_cache().clear()
