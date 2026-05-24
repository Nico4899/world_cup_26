"""Shared HTTP utilities used across the ingest layer."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimiter:
    """Sliding-window limiter: at most ``limit`` calls per ``window`` seconds.

    Thread-safe via an internal lock. Used by every external HTTP ingester so
    we never breach a free-tier rate cap (football-data.org 10/min,
    TheSportsDB 30/min, FBref polite 6/min).
    """

    limit: int
    window: float
    _calls: deque[float]
    _lock: threading.Lock

    @classmethod
    def make(cls, limit: int, window: float) -> RateLimiter:
        return cls(limit=limit, window=window, _calls=deque(), _lock=threading.Lock())

    def acquire(self, *, now: float | None = None, sleep=time.sleep) -> None:
        with self._lock:
            t = now if now is not None else time.monotonic()
            while self._calls and t - self._calls[0] >= self.window:
                self._calls.popleft()
            if len(self._calls) >= self.limit:
                wait = self.window - (t - self._calls[0]) + 0.01
                sleep(max(wait, 0.0))
                t2 = time.monotonic()
                while self._calls and t2 - self._calls[0] >= self.window:
                    self._calls.popleft()
                self._calls.append(t2)
            else:
                self._calls.append(t)


__all__ = ["RateLimiter"]
