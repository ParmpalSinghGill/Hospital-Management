"""Small in-process rate limiter / lockout counter (no external deps).

Good enough for a single-process deployment; swap for Redis when scaling out.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict


class RateLimiter:
    """Sliding-window limiter: at most *max_events* per *window_secs* per key."""

    def __init__(self, max_events: int, window_secs: float) -> None:
        self.max_events = int(max_events)
        self.window = float(window_secs)
        self._events: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, q: Deque[float], now: float) -> None:
        cutoff = now - self.window
        while q and q[0] < cutoff:
            q.popleft()

    def allow(self, key: str) -> bool:
        """Record an event for *key*; False when the key is over its budget."""
        now = time.monotonic()
        with self._lock:
            q = self._events.setdefault(key, deque())
            self._prune(q, now)
            if len(q) >= self.max_events:
                return False
            q.append(now)
            return True

    def count(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            q = self._events.get(key)
            if not q:
                return 0
            self._prune(q, now)
            return len(q)

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)

    def retry_after(self, key: str) -> float:
        now = time.monotonic()
        with self._lock:
            q = self._events.get(key)
            if not q:
                return 0.0
            self._prune(q, now)
            if len(q) < self.max_events:
                return 0.0
            return max(0.0, self.window - (now - q[0]))


class Lockout:
    """Count failures per key; lock the key for *lock_secs* after *max_failures*."""

    def __init__(self, max_failures: int, window_secs: float, lock_secs: float) -> None:
        self._fails = RateLimiter(max_failures, window_secs)
        self.max_failures = int(max_failures)
        self.lock_secs = float(lock_secs)
        self._locked_until: Dict[str, float] = {}
        self._lock = threading.Lock()

    def is_locked(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            until = self._locked_until.get(key, 0.0)
            if until and until > now:
                return True
            if until:
                self._locked_until.pop(key, None)
        return False

    def record_failure(self, key: str) -> bool:
        """Register a failure; returns True when the key is now locked (>= max_failures)."""
        self._fails.allow(key)
        if self._fails.count(key) >= self.max_failures:
            with self._lock:
                self._locked_until[key] = time.monotonic() + self.lock_secs
            return True
        return False

    def record_success(self, key: str) -> None:
        self._fails.reset(key)
        with self._lock:
            self._locked_until.pop(key, None)
