"""Fault injection helpers for controlled benchmark scenarios."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from enum import Enum
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import psutil


class FaultScenario(str, Enum):
    NONE = "none"
    CAMERA_DISCONNECT = "camera_disconnect"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_STRESS = "cpu_stress"


@dataclass(frozen=True)
class FaultInjectorConfig:
    memory_hard_cap_percent: float = 92.0
    min_memory_allocation_bytes: int = 1 * 1024 * 1024
    process_shutdown_timeout_sec: float = 2.0


@dataclass(frozen=True)
class RunningFault:
    scenario: FaultScenario
    pid: int
    started_at: float
    duration_sec: float
    command: tuple[str, ...]
    requested_bytes: int | None = None

    @property
    def deadline(self) -> float:
        return self.started_at + self.duration_sec


@dataclass(frozen=True)
class FaultStatus:
    active: bool
    scenarios: tuple[FaultScenario, ...]
    processes: tuple[RunningFault, ...]


PopenFactory = Callable[..., subprocess.Popen[Any]]


def should_inject(frame_index: int, every_n_frames: int | None) -> bool:
    if every_n_frames is None or every_n_frames <= 0:
        return False
    return frame_index > 0 and frame_index % every_n_frames == 0


class FaultInjector:
    """Start short-lived pressure processes and clean them up reliably.

    The pressure runs in child processes so ``clear_all()`` can release it with
    one termination path even when Python's GC behavior is unpredictable.
    """

    def __init__(
        self,
        config: FaultInjectorConfig | None = None,
        *,
        popen_factory: PopenFactory = subprocess.Popen,
        python_executable: str | None = None,
        stress_ng_path: str | None = None,
        register_atexit: bool = True,
    ) -> None:
        self.config = config or FaultInjectorConfig()
        if not 0 < self.config.memory_hard_cap_percent <= 100:
            raise ValueError("memory_hard_cap_percent must be in (0, 100]")
        if self.config.min_memory_allocation_bytes <= 0:
            raise ValueError("min_memory_allocation_bytes must be positive")
        if self.config.process_shutdown_timeout_sec <= 0:
            raise ValueError("process_shutdown_timeout_sec must be positive")

        self._popen_factory = popen_factory
        self._python = python_executable or sys.executable
        self._stress_ng_path = stress_ng_path
        self._processes: list[tuple[subprocess.Popen[Any], RunningFault]] = []
        if register_atexit:
            atexit.register(self.clear_all)

    def __enter__(self) -> FaultInjector:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.clear_all()

    def inject_memory_pressure(
        self,
        *,
        target_percent: float,
        duration_sec: float,
    ) -> RunningFault | None:
        """Allocate memory in a child process until host usage nears target."""
        _validate_duration(duration_sec)
        if target_percent <= 0:
            raise ValueError("target_percent must be positive")
        if target_percent > self.config.memory_hard_cap_percent:
            raise ValueError(
                "target_percent exceeds hard cap "
                f"({target_percent:.1f} > {self.config.memory_hard_cap_percent:.1f})"
            )

        memory = psutil.virtual_memory()
        desired_used = int(memory.total * (target_percent / 100.0))
        bytes_to_allocate = max(
            self.config.min_memory_allocation_bytes,
            desired_used - int(memory.used),
        )
        command = self._memory_pressure_command(bytes_to_allocate, duration_sec)
        return self._start_process(
            scenario=FaultScenario.MEMORY_PRESSURE,
            command=command,
            duration_sec=duration_sec,
            requested_bytes=bytes_to_allocate,
        )

    def inject_cpu_stress(
        self,
        *,
        duration_sec: float,
        num_workers: int | None = None,
    ) -> tuple[RunningFault, ...]:
        """Start CPU pressure using stress-ng when available, else Python loops."""
        _validate_duration(duration_sec)
        workers = num_workers or max(1, (os.cpu_count() or 1) - 1)
        if workers <= 0:
            raise ValueError("num_workers must be positive")

        stress_ng = self._stress_ng_path
        if stress_ng is None:
            stress_ng = shutil.which("stress-ng")

        if stress_ng:
            command = (
                stress_ng,
                "--cpu",
                str(workers),
                "--timeout",
                _format_timeout(duration_sec),
                "--quiet",
            )
            return (
                self._start_process(
                    scenario=FaultScenario.CPU_STRESS,
                    command=command,
                    duration_sec=duration_sec,
                ),
            )

        faults = []
        command = self._cpu_busy_loop_command(duration_sec)
        for _ in range(workers):
            faults.append(
                self._start_process(
                    scenario=FaultScenario.CPU_STRESS,
                    command=command,
                    duration_sec=duration_sec,
                )
            )
        return tuple(faults)

    def status(self) -> FaultStatus:
        self._reap_finished()
        processes = tuple(fault for _process, fault in self._processes)
        return FaultStatus(
            active=bool(processes),
            scenarios=tuple(dict.fromkeys(fault.scenario for fault in processes)),
            processes=processes,
        )

    def is_active(self, scenario: FaultScenario | None = None) -> bool:
        status = self.status()
        if scenario is None:
            return status.active
        return scenario in status.scenarios

    def clear_all(self) -> None:
        """Terminate all child processes, escalating to kill if needed."""
        processes = list(self._processes)
        self._processes.clear()
        for process, _fault in processes:
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=self.config.process_shutdown_timeout_sec)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.config.process_shutdown_timeout_sec)

    def _start_process(
        self,
        *,
        scenario: FaultScenario,
        command: Sequence[str],
        duration_sec: float,
        requested_bytes: int | None = None,
    ) -> RunningFault:
        process = self._popen_factory(
            list(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        fault = RunningFault(
            scenario=scenario,
            pid=int(process.pid),
            started_at=time.monotonic(),
            duration_sec=duration_sec,
            command=tuple(command),
            requested_bytes=requested_bytes,
        )
        self._processes.append((process, fault))
        return fault

    def _reap_finished(self) -> None:
        self._processes = [
            (process, fault)
            for process, fault in self._processes
            if process.poll() is None
        ]

    def _memory_pressure_command(
        self,
        bytes_to_allocate: int,
        duration_sec: float,
    ) -> tuple[str, ...]:
        script = (
            "import sys,time\n"
            "size=int(sys.argv[1]); duration=float(sys.argv[2])\n"
            "chunk=1024*1024; blocks=[]; remaining=size\n"
            "while remaining>0:\n"
            "    n=min(chunk, remaining)\n"
            "    b=bytearray(n)\n"
            "    for i in range(0, n, 4096): b[i]=1\n"
            "    blocks.append(b); remaining-=n\n"
            "time.sleep(duration)\n"
        )
        return (
            self._python,
            "-c",
            script,
            str(int(bytes_to_allocate)),
            str(float(duration_sec)),
        )

    def _cpu_busy_loop_command(self, duration_sec: float) -> tuple[str, ...]:
        script = (
            "import sys,time\n"
            "end=time.monotonic()+float(sys.argv[1]); x=0\n"
            "while time.monotonic()<end:\n"
            "    x=(x+1)%1000003\n"
        )
        return (self._python, "-c", script, str(float(duration_sec)))


def _validate_duration(duration_sec: float) -> None:
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")


def _format_timeout(duration_sec: float) -> str:
    duration = float(duration_sec)
    if duration.is_integer():
        return f"{int(duration)}s"
    return f"{duration:.3f}s"
