"""Layer 08 - Metrics.

An in-process metrics sink with a Prometheus-shaped API. Deliberately small:
the point is that every call site is already instrumented, so swapping this for
prometheus_client or OTel is a change in one file, not a hundred.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    tags = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{tags}}}"


@dataclass
class _Histogram:
    count: int = 0
    total: float = 0.0
    values: list[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        self.count += 1
        self.total += v
        # Bounded reservoir - we only need percentiles, not a full time series.
        self.values.append(v)
        if len(self.values) > 1000:
            del self.values[: len(self.values) - 1000]

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        idx = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
        return ordered[idx]


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, _Histogram] = defaultdict(_Histogram)

    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[_key(name, labels)] += value

    def add(self, name: str, value: float, **labels: str) -> None:
        self.incr(name, value, **labels)

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._histograms[_key(name, labels)].add(value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {
                    k: {
                        "count": h.count,
                        "mean": round(h.total / h.count, 2) if h.count else 0.0,
                        "p50": round(h.percentile(50), 2),
                        "p95": round(h.percentile(95), 2),
                        "p99": round(h.percentile(99), 2),
                    }
                    for k, h in self._histograms.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


METRICS = Metrics()
