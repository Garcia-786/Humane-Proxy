"""Lightweight environment + resource profiling for benchmarks.

Used by ``hp benchmark`` to record the machine it ran on and the CPU/
memory it consumed, so published benchmark numbers are reproducible and
trustworthy. ``psutil`` is optional: without it, environment capture
still works (stdlib only) and resource sampling degrades to a no-op.
"""

from __future__ import annotations

import os
import platform
import threading

try:
    import psutil

    _PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _PSUTIL = False


def capture_environment() -> dict:
    """Return a dict describing the current machine and runtime."""
    env: dict = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "psutil_available": _PSUTIL,
    }
    if _PSUTIL:
        env["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    return env


class ResourceSampler:
    """Sample this process's CPU% and RSS on a background thread.

    Use as a context manager around the workload. A no-op when psutil is
    unavailable, so callers need no branching.
    """

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self._cpu: list[float] = []
        self._rss: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process() if _PSUTIL else None

    def __enter__(self) -> "ResourceSampler":
        if self._proc is not None:
            # Prime cpu_percent so the first real sample is meaningful.
            self._proc.cpu_percent(None)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._cpu.append(self._proc.cpu_percent(None))
                self._rss.append(self._proc.memory_info().rss)
            except Exception:
                break

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def stats(self) -> dict | None:
        """Return resource stats, or None when psutil is unavailable."""
        if not _PSUTIL:
            return None
        # Non-zero CPU samples only: idle ticks between async awaits would
        # otherwise drag the mean toward zero and understate real cost.
        cpu = [c for c in self._cpu if c > 0.0]
        peak_rss = max(self._rss) if self._rss else 0
        return {
            "samples": len(self._rss),
            "cpu_percent_mean": round(sum(cpu) / len(cpu), 1) if cpu else 0.0,
            "cpu_percent_peak": round(max(self._cpu), 1) if self._cpu else 0.0,
            "cpu_count": os.cpu_count(),
            "peak_rss_mb": round(peak_rss / 1e6, 1),
        }
