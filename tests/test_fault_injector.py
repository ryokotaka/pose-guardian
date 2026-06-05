import pytest

from src import fault_injector
from src.fault_injector import (
    FaultInjector,
    FaultInjectorConfig,
    FaultScenario,
    should_inject,
)


class FakeMemory:
    total = 1_000
    used = 200


class FakeProcess:
    _next_pid = 1000

    def __init__(self, command, stdout=None, stderr=None):
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.terminated = False
        self.killed = False
        self._finished = False

    def poll(self):
        return 0 if self._finished else None

    def terminate(self):
        self.terminated = True
        self._finished = True

    def kill(self):
        self.killed = True
        self._finished = True

    def wait(self, timeout=None):
        if not self._finished:
            raise fault_injector.subprocess.TimeoutExpired(self.command, timeout)
        return 0


class FakePopenFactory:
    def __init__(self):
        self.processes = []

    def __call__(self, command, stdout=None, stderr=None):
        process = FakeProcess(command, stdout=stdout, stderr=stderr)
        self.processes.append(process)
        return process


def test_should_inject_respects_frame_interval() -> None:
    assert should_inject(0, 5) is False
    assert should_inject(4, 5) is False
    assert should_inject(5, 5) is True
    assert should_inject(5, 0) is False
    assert should_inject(5, None) is False


def test_memory_pressure_starts_python_process(monkeypatch) -> None:
    popen = FakePopenFactory()
    monkeypatch.setattr(fault_injector.psutil, "virtual_memory", lambda: FakeMemory())
    injector = FaultInjector(
        FaultInjectorConfig(
            memory_hard_cap_percent=90.0,
            min_memory_allocation_bytes=1,
        ),
        popen_factory=popen,
        python_executable="/fake/python",
        register_atexit=False,
    )

    fault = injector.inject_memory_pressure(target_percent=80.0, duration_sec=3.0)

    assert fault is not None
    assert fault.scenario is FaultScenario.MEMORY_PRESSURE
    assert fault.requested_bytes == 600
    assert popen.processes[0].command[0] == "/fake/python"
    assert popen.processes[0].command[-2:] == ["600", "3.0"]
    assert injector.is_active(FaultScenario.MEMORY_PRESSURE) is True


def test_memory_pressure_caps_unsafe_target(monkeypatch) -> None:
    monkeypatch.setattr(fault_injector.psutil, "virtual_memory", lambda: FakeMemory())
    injector = FaultInjector(
        FaultInjectorConfig(memory_hard_cap_percent=85.0),
        popen_factory=FakePopenFactory(),
        register_atexit=False,
    )

    with pytest.raises(ValueError, match="hard cap"):
        injector.inject_memory_pressure(target_percent=90.0, duration_sec=1.0)


def test_cpu_stress_uses_stress_ng_when_available(monkeypatch) -> None:
    popen = FakePopenFactory()
    monkeypatch.setattr(fault_injector.shutil, "which", lambda name: "/bin/stress-ng")
    injector = FaultInjector(popen_factory=popen, register_atexit=False)

    faults = injector.inject_cpu_stress(duration_sec=2.0, num_workers=3)

    assert len(faults) == 1
    assert faults[0].scenario is FaultScenario.CPU_STRESS
    assert popen.processes[0].command == [
        "/bin/stress-ng",
        "--cpu",
        "3",
        "--timeout",
        "2s",
        "--quiet",
    ]


def test_cpu_stress_falls_back_to_python_workers(monkeypatch) -> None:
    popen = FakePopenFactory()
    monkeypatch.setattr(fault_injector.shutil, "which", lambda name: None)
    injector = FaultInjector(
        popen_factory=popen,
        python_executable="/fake/python",
        register_atexit=False,
    )

    faults = injector.inject_cpu_stress(duration_sec=1.5, num_workers=2)

    assert len(faults) == 2
    assert all(fault.scenario is FaultScenario.CPU_STRESS for fault in faults)
    assert [process.command[0] for process in popen.processes] == [
        "/fake/python",
        "/fake/python",
    ]
    assert [process.command[-1] for process in popen.processes] == ["1.5", "1.5"]


def test_clear_all_terminates_running_processes(monkeypatch) -> None:
    popen = FakePopenFactory()
    monkeypatch.setattr(fault_injector.shutil, "which", lambda name: None)
    injector = FaultInjector(popen_factory=popen, register_atexit=False)
    injector.inject_cpu_stress(duration_sec=5.0, num_workers=2)

    injector.clear_all()

    assert all(process.terminated for process in popen.processes)
    assert injector.status().active is False


def test_context_manager_clears_processes(monkeypatch) -> None:
    popen = FakePopenFactory()
    monkeypatch.setattr(fault_injector.shutil, "which", lambda name: None)

    with FaultInjector(popen_factory=popen, register_atexit=False) as injector:
        injector.inject_cpu_stress(duration_sec=5.0, num_workers=1)

    assert popen.processes[0].terminated is True
