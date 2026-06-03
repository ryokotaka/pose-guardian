"""Resource monitoring helpers for Pi and local development hosts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Deque

import psutil


_THERMAL_ZONE_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
_CURRENT_THROTTLE_MASK = 0x0E  # ARM freq cap, throttling, soft temp limit.


@dataclass(frozen=True)
class MonitorConfig:
    sample_interval_sec: float = 1.0
    fps_window_size: int = 30
    enable_power: bool = True


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: float
    cpu_temp_celsius: float
    cpu_usage_percent: float
    cpu_freq_mhz: float
    memory_used_percent: float
    memory_used_bytes: int
    memory_available_bytes: int
    is_throttled: bool
    throttle_flags: int
    fps: float
    pmic_rail_estimate_watts: float | None


class ResourceMonitor:
    """Sample CPU, memory, FPS, and Pi throttle state.

    The monitor is safe to use on non-Pi hosts: Pi-specific commands simply
    fall back to neutral values when unavailable.
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig()
        if self.config.sample_interval_sec <= 0:
            raise ValueError("sample_interval_sec must be positive")
        if self.config.fps_window_size <= 0:
            raise ValueError("fps_window_size must be positive")

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_times: Deque[float] = deque(maxlen=self.config.fps_window_size)
        self._history: Deque[ResourceSnapshot] = deque(
            maxlen=max(1, int(300 / self.config.sample_interval_sec))
        )
        self._latest_snapshot: ResourceSnapshot | None = None
        self._store_snapshot(self._sample())

    def start(self) -> None:
        """Start background sampling. Calling this twice is harmless."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ResourceMonitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling. Calling this twice is harmless."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.config.sample_interval_sec + 1.0)
        self._thread = None

    def snapshot(self) -> ResourceSnapshot:
        """Return the latest immutable snapshot without forcing a new sample."""
        with self._lock:
            assert self._latest_snapshot is not None
            return self._latest_snapshot

    def record_inference(self, latency_ms: float) -> None:
        """Record one completed inference for rolling FPS calculation."""
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        with self._lock:
            self._frame_times.append(time.monotonic())

    def history(self, seconds: int = 60) -> list[ResourceSnapshot]:
        """Return snapshots sampled within the last ``seconds`` seconds."""
        cutoff = time.monotonic() - max(0, seconds)
        with self._lock:
            return [sample for sample in self._history if sample.timestamp >= cutoff]

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._store_snapshot(self._sample())
            if self._stop_event.wait(self.config.sample_interval_sec):
                break

    def _store_snapshot(self, snapshot: ResourceSnapshot) -> None:
        with self._lock:
            self._latest_snapshot = snapshot
            self._history.append(snapshot)

    def _sample(self) -> ResourceSnapshot:
        memory = psutil.virtual_memory()
        cpu_freq = psutil.cpu_freq()
        throttle_flags = _read_throttle_flags()
        return ResourceSnapshot(
            timestamp=time.monotonic(),
            cpu_temp_celsius=_read_cpu_temp_celsius(),
            cpu_usage_percent=float(psutil.cpu_percent(interval=None)),
            cpu_freq_mhz=float(cpu_freq.current) if cpu_freq is not None else 0.0,
            memory_used_percent=float(memory.percent),
            memory_used_bytes=int(memory.used),
            memory_available_bytes=int(memory.available),
            is_throttled=is_currently_throttled(throttle_flags),
            throttle_flags=throttle_flags,
            fps=self._rolling_fps(),
            pmic_rail_estimate_watts=(
                _read_pmic_rail_estimate_watts()
                if self.config.enable_power
                else None
            ),
        )

    def _rolling_fps(self) -> float:
        with self._lock:
            if len(self._frame_times) < 2:
                return 0.0
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._frame_times) - 1) / elapsed


def read_resources() -> ResourceSnapshot:
    """Read one resource snapshot without starting a background monitor."""
    return ResourceMonitor().snapshot()


def is_currently_throttled(flags: int) -> bool:
    """Return True for current frequency cap/throttle/soft-temp bits."""
    return (flags & _CURRENT_THROTTLE_MASK) != 0


def _read_cpu_temp_celsius() -> float:
    temp = _read_vcgencmd_temp()
    if temp is not None:
        return temp

    temp = _read_sysfs_temp()
    if temp is not None:
        return temp

    temp = _read_psutil_temp()
    if temp is not None:
        return temp

    return 0.0


def _read_vcgencmd_temp() -> float | None:
    output = _run_vcgencmd("measure_temp")
    if output is None:
        return None
    return _parse_measure_temp(output)


def _read_sysfs_temp(path: Path = _THERMAL_ZONE_PATH) -> float | None:
    try:
        return int(path.read_text(encoding="utf-8").strip()) / 1000.0
    except (FileNotFoundError, OSError, ValueError):
        return None


def _read_psutil_temp() -> float | None:
    try:
        sensors = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        return None
    for entries in sensors.values():
        for entry in entries:
            current = getattr(entry, "current", None)
            if current is not None:
                return float(current)
    return None


def _read_throttle_flags() -> int:
    output = _run_vcgencmd("get_throttled")
    if output is None:
        return 0
    return _parse_get_throttled(output)


def _read_pmic_rail_estimate_watts() -> float | None:
    output = _run_vcgencmd("pmic_read_adc")
    if output is None:
        return None
    return _parse_pmic_read_adc(output)


def _run_vcgencmd(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["vcgencmd", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _parse_measure_temp(output: str) -> float | None:
    match = re.search(r"temp=([-+]?\d+(?:\.\d+)?)", output)
    if match is None:
        return None
    return float(match.group(1))


def _parse_get_throttled(output: str) -> int:
    match = re.search(r"throttled=(0x[0-9a-fA-F]+|\d+)", output)
    if match is None:
        return 0
    return int(match.group(1), 0)


def _parse_pmic_read_adc(output: str) -> float | None:
    currents: dict[str, float] = {}
    volts: dict[str, float] = {}
    pattern = re.compile(
        r"^\s*(?P<rail>\S+)\s+"
        r"(?P<kind>current|volt)\(\d+\)="
        r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>[AV])",
        re.MULTILINE,
    )
    for match in pattern.finditer(output):
        rail = _normalize_pmic_rail_name(match.group("rail"))
        value = float(match.group("value"))
        if match.group("kind") == "current":
            currents[rail] = value
        else:
            volts[rail] = value

    watts = sum(currents[rail] * volts[rail] for rail in currents.keys() & volts.keys())
    return watts if watts > 0 else None


def _normalize_pmic_rail_name(rail: str) -> str:
    if rail.endswith("_A") or rail.endswith("_V"):
        return rail[:-2]
    return rail
