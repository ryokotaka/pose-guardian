# Memory Pressure Smoke Test

This smoke test checks whether the `memory_pressure` fault path can trigger the
controller on a real Raspberry Pi 5 run.

## Conditions

- Device: Raspberry Pi 5 with active cooler attached
- Camera: USB camera via OpenCV
- Controller mode: `controlled`
- Initial model: `thunder`
- Run duration: 70 seconds
- Fault scenario: `memory_pressure`
- Fault timing: starts after 15 seconds, runs for 20 seconds
- Memory target: 82%
- Raw CSV file: `metrics/memory_pressure_smoke.csv` (local only, not committed)

## Summary

| metric | value |
|---|---:|
| Rows | 70 |
| Frames processed | 1044 |
| Model switches | 2 |
| Skipped frames | 0 |
| Force GC actions | 0 |
| Final temperature | 54.3 C |
| Throttle flags | `0x0` |
| Initial memory used | 215 MiB |
| Final memory used | 232 MiB |

## Key Events

| elapsed | event |
|---:|---|
| 17.0 s | `memory_used_percent=84.4 >= threshold=80.0`; controller switched `normal -> degraded` and `thunder -> lightning` |
| 18-37 s | Memory stayed around 83-85%; controller remained `degraded` |
| 38.0 s | Fault ended and memory dropped to about 7.8%; recovery hold timer started |
| End of run | Controller returned to `normal` / `thunder`; `model_switches=2` |

## Interpretation

This test shows that the memory-pressure path works on the Pi at the degraded
threshold: the controller observed high memory usage, switched to the lighter
model, waited for recovery conditions to hold, and returned to Thunder after the
fault cleared.

This smoke test does not prove a memory-pressure performance improvement, and it
does not exercise the `critical` / `force_gc` path. The configured target was
82%, which is above the degraded threshold of 80% but below the critical
threshold of 90%.

