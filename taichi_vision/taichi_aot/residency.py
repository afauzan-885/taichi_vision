"""Lease-safe, byte-budgeted device residency primitives."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Reservation:
    soft_bytes: int = 0
    hard_bytes: Optional[int] = None
    weight: float = 1.0

    def __post_init__(self):
        if self.soft_bytes < 0 or (self.hard_bytes is not None and self.hard_bytes < self.soft_bytes):
            raise ValueError("reservation requires 0 <= soft_bytes <= hard_bytes")
        if self.weight <= 0:
            raise ValueError("reservation weight must be positive")


@dataclass
class ResidentEntry:
    key: str
    owner: str
    buffer: Any
    size_bytes: int
    generation: int = 0
    ref_count: int = 0
    pin_count: int = 0
    hit_count: int = 0
    last_access: float = 0.0
    dispose: Optional[Callable[[Any], None]] = None
    fence_ready: Optional[Callable[[], bool]] = None
    checksum: Any = None
    source_checksum: Any = None
    invalidated: bool = False

    @property
    def leased(self):
        return self.ref_count > 0 or self.pin_count > 0

    def can_evict(self):
        """Return whether disposal is safe, conservatively on fence errors.

        Fence callbacks are supplied by native runtimes and may transiently
        fail while a device/queue is being torn down.  An exception must never
        make cache maintenance dispose an in-flight buffer; treat it as
        ``not ready`` and let a later maintenance pass retry the callback.
        """
        if self.leased:
            return False
        if self.fence_ready is None:
            return True
        try:
            return bool(self.fence_ready())
        except Exception:
            return False


class DeviceResidencyCache:
    """Generic GPU-buffer cache; native ownership stays behind dispose callbacks."""

    def __init__(self, max_bytes=0):
        self.max_bytes = max(0, int(max_bytes))
        self._entries = OrderedDict()
        self._reservations = {}
        self._owner_bytes = {}
        self._size_bytes = 0
        self._generation = count(1)
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "admissions": 0,
            "rejects": 0,
            "evictions": 0,
            "bytes_evicted": 0,
            "dispose_errors": 0,
        }

    @property
    def size_bytes(self):
        with self._lock:
            return self._size_bytes

    def configure_owner(self, owner, soft_bytes=0, hard_bytes=None, weight=1.0):
        with self._lock:
            self._reservations[str(owner)] = Reservation(
                int(soft_bytes), None if hard_bytes is None else int(hard_bytes), float(weight)
            )

    def set_budget(self, max_bytes):
        disposals = []
        with self._lock:
            self.max_bytes = max(0, int(max_bytes))
            disposals.extend(self._reap_invalidated_locked())
            # Budget changes are authoritative immediately for admission and
            # logical visibility.  Entries that cannot be disposed yet are
            # marked invalidated so their final lease/fence release completes
            # the transition rather than letting them become reusable again.
            projected = self._size_bytes
            for key, entry in self._eviction_candidates(""):
                if projected <= self.max_bytes:
                    break
                if entry.invalidated:
                    projected -= entry.size_bytes
                    continue
                if entry.can_evict():
                    detached = self._detach_locked(key, entry)
                    if detached is not None:
                        disposals.append(detached)
                    projected = self._size_bytes
                else:
                    entry.invalidated = True
                    projected -= entry.size_bytes
        self._dispose_detached(disposals)

    def _owner_usage(self, owner):
        return self._owner_bytes.get(owner, 0)

    def _reservation(self, owner, requesting_owner=None):
        configured = self._reservations.get(owner)
        if configured is not None:
            return configured
        active = {name for name, size in self._owner_bytes.items() if size > 0}
        active.add(str(requesting_owner or owner))
        automatic = [name for name in active if name not in self._reservations]
        configured_soft = sum(
            reservation.soft_bytes
            for name, reservation in self._reservations.items()
            if name in active
        )
        available = max(0, self.max_bytes - configured_soft)
        soft = available // max(1, len(automatic))
        return Reservation(soft_bytes=soft, hard_bytes=self.max_bytes)

    def _eviction_candidates(self, requesting_owner):
        def priority(item):
            _, entry = item
            reservation = self._reservation(entry.owner, requesting_owner)
            borrowed = self._owner_usage(entry.owner) > reservation.soft_bytes
            same_owner = entry.owner == requesting_owner
            return (not borrowed, not same_owner, entry.hit_count, entry.last_access)

        return sorted(self._entries.items(), key=priority)

    def _detach_locked(self, key, entry):
        """Detach one entry atomically; caller must hold ``_lock``.

        Native/resource disposal is intentionally *not* executed here.  The
        returned callback payload is consumed after releasing the cache lock,
        preventing residency -> engine/native lock inversion.
        """
        current = self._entries.get(key)
        if current is not entry:
            return None
        self._entries.pop(key, None)
        self._size_bytes = max(0, self._size_bytes - entry.size_bytes)
        remaining = max(0, self._owner_usage(entry.owner) - entry.size_bytes)
        if remaining:
            self._owner_bytes[entry.owner] = remaining
        else:
            self._owner_bytes.pop(entry.owner, None)
        self._stats["evictions"] += 1
        self._stats["bytes_evicted"] += entry.size_bytes
        return (entry.dispose, entry.buffer)

    def _dispose_detached(self, disposals):
        """Run detached resource callbacks without holding the residency lock."""
        first_error = None
        for callback, buffer in disposals:
            if callback is None:
                continue
            try:
                callback(buffer)
            except Exception as exc:
                with self._lock:
                    self._stats["dispose_errors"] += 1
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _reap_invalidated_locked(self):
        """Detach invalidated entries whose lease/fence is now releasable."""
        disposals = []
        for key, entry in list(self._entries.items()):
            if entry.invalidated and entry.can_evict():
                detached = self._detach_locked(key, entry)
                if detached is not None:
                    disposals.append(detached)
        return disposals

    def _plan_admission_locked(self, incoming_bytes, requesting_owner, previous=None):
        """Return evictions needed for admission without mutating the cache.

        ``previous`` is a same-key entry that will be replaced atomically.  Its
        bytes are removed from the projected footprint before deciding whether
        additional entries need eviction, avoiding temporary double-counting.
        """
        projected = self._size_bytes - (previous.size_bytes if previous is not None else 0)
        planned = []
        for key, entry in self._eviction_candidates(requesting_owner):
            if projected + incoming_bytes <= self.max_bytes:
                break
            if previous is not None and entry is previous:
                continue
            if entry.invalidated or not entry.can_evict():
                continue
            planned.append((key, entry))
            projected -= entry.size_bytes
        return projected + incoming_bytes <= self.max_bytes, planned

    def put(
        self, key, owner, buffer, size_bytes, dispose=None, fence_ready=None,
        checksum=None, source_checksum=None,
    ):
        key, owner, size_bytes = str(key), str(owner), int(size_bytes)
        if size_bytes <= 0:
            raise ValueError("resident entry size must be positive")

        disposals = []
        entry = None
        with self._lock:
            disposals.extend(self._reap_invalidated_locked())
            previous = self._entries.get(key)
            if previous is not None and not previous.can_evict():
                self._stats["rejects"] += 1
                rejected = True
            else:
                reservation = self._reservation(owner, owner)
                replaced_owner_bytes = (
                    previous.size_bytes
                    if previous is not None and previous.owner == owner
                    else 0
                )
                projected_owner = self._owner_usage(owner) - replaced_owner_bytes + size_bytes
                rejected = (
                    self.max_bytes == 0
                    or size_bytes > self.max_bytes
                    or (
                        reservation.hard_bytes is not None
                        and projected_owner > reservation.hard_bytes
                    )
                )
                planned = []
                if not rejected:
                    fits, planned = self._plan_admission_locked(
                        size_bytes, owner, previous=previous
                    )
                    rejected = not fits

                if rejected:
                    self._stats["rejects"] += 1
                else:
                    # All admission checks succeeded without mutation.  Only
                    # now detach the old key/additional victims, then publish
                    # the replacement while the cache lock still serializes
                    # visibility/accounting.
                    if previous is not None:
                        detached = self._detach_locked(key, previous)
                        if detached is not None:
                            disposals.append(detached)
                    for victim_key, victim in planned:
                        detached = self._detach_locked(victim_key, victim)
                        if detached is not None:
                            disposals.append(detached)

                    entry = ResidentEntry(
                        key,
                        owner,
                        buffer,
                        size_bytes,
                        generation=next(self._generation),
                        last_access=time.monotonic(),
                        dispose=dispose,
                        fence_ready=fence_ready,
                        checksum=checksum,
                        source_checksum=source_checksum,
                    )
                    self._entries[key] = entry
                    self._size_bytes += size_bytes
                    self._owner_bytes[owner] = self._owner_usage(owner) + size_bytes
                    self._stats["admissions"] += 1

        self._dispose_detached(disposals)
        return entry

    def peek(self, key):
        with self._lock:
            return self._entries.get(str(key))

    def invalidate(self, key):
        key = str(key)
        disposals = []
        with self._lock:
            disposals.extend(self._reap_invalidated_locked())
            entry = self._entries.get(key)
            if entry is None:
                found = False
            else:
                found = True
                entry.invalidated = True
                if entry.can_evict():
                    detached = self._detach_locked(key, entry)
                    if detached is not None:
                        disposals.append(detached)
        self._dispose_detached(disposals)
        return found

    def invalidate_owner(self, owner):
        """Logically invalidate resident entries owned by a quarantined operation."""
        owner = str(owner)
        invalidated = 0
        disposals = []
        with self._lock:
            disposals.extend(self._reap_invalidated_locked())
            for key, entry in list(self._entries.items()):
                if entry.owner != owner:
                    continue
                entry.invalidated = True
                invalidated += 1
                if entry.can_evict():
                    detached = self._detach_locked(key, entry)
                    if detached is not None:
                        disposals.append(detached)
        self._dispose_detached(disposals)
        return invalidated

    def get(self, key):
        key = str(key)
        disposals = []
        with self._lock:
            disposals.extend(self._reap_invalidated_locked())
            entry = self._entries.get(key)
            if entry is None or entry.invalidated:
                if entry is not None and entry.can_evict():
                    detached = self._detach_locked(key, entry)
                    if detached is not None:
                        disposals.append(detached)
                self._stats["misses"] += 1
                result = None
            elif entry.fence_ready is not None:
                try:
                    ready = bool(entry.fence_ready())
                except Exception:
                    ready = False
                if not ready:
                    # The producer has not signalled completion yet. Keep the
                    # entry resident, but do not hand its native handle to a
                    # consumer that could race the upload/dispatch.
                    self._stats["misses"] += 1
                    result = None
                else:
                    entry.hit_count += 1
                    entry.last_access = time.monotonic()
                    self._entries.move_to_end(key)
                    self._stats["hits"] += 1
                    result = entry
            else:
                entry.hit_count += 1
                entry.last_access = time.monotonic()
                self._entries.move_to_end(key)
                self._stats["hits"] += 1
                result = entry
        self._dispose_detached(disposals)
        return result

    @contextmanager
    def lease(self, key):
        key = str(key)
        disposals = []
        with self._lock:
            disposals.extend(self._reap_invalidated_locked())
            entry = self._entries.get(key)
            if entry is None or entry.invalidated:
                entry = None
                generation = None
                self._stats["misses"] += 1
            else:
                if entry.fence_ready is not None:
                    try:
                        ready = bool(entry.fence_ready())
                    except Exception:
                        ready = False
                    if not ready:
                        entry = None
                if entry is None:
                    generation = None
                    self._stats["misses"] += 1
                else:
                    entry.hit_count += 1
                    entry.last_access = time.monotonic()
                    self._entries.move_to_end(key)
                    self._stats["hits"] += 1
                    entry.ref_count += 1
                    generation = entry.generation
        self._dispose_detached(disposals)

        if entry is None:
            yield None
            return
        try:
            yield entry
        finally:
            disposals = []
            with self._lock:
                current = self._entries.get(key)
                if current is not None and current.generation == generation:
                    current.ref_count = max(0, current.ref_count - 1)
                    if current.invalidated and current.can_evict():
                        detached = self._detach_locked(key, current)
                        if detached is not None:
                            disposals.append(detached)
            self._dispose_detached(disposals)

    def clear(self):
        disposals = []
        with self._lock:
            disposals.extend(self._reap_invalidated_locked())
            for key, entry in list(self._entries.items()):
                # Clear is a logical barrier immediately.  A current consumer
                # may finish its lease, but no new consumer may see the entry.
                entry.invalidated = True
                if entry.can_evict():
                    detached = self._detach_locked(key, entry)
                    if detached is not None:
                        disposals.append(detached)
        self._dispose_detached(disposals)

    def stats(self):
        with self._lock:
            return {
                **self._stats,
                "entries": len(self._entries),
                "size_bytes": self._size_bytes,
                "max_bytes": self.max_bytes,
                "owner_bytes": dict(self._owner_bytes),
            }
