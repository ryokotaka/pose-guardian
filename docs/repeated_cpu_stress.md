# Repeated CPU-Stress Runs

This document summarizes four paired Raspberry Pi 5 CPU-stress runs. The first
pair is the README graph run; the other three pairs were repeated with the same
fault settings to check whether the trend was stable.

## Conditions

- Device: Raspberry Pi 5 with active cooler attached
- Camera: USB camera via OpenCV
- Initial model: `thunder`
- Run duration: 90 seconds per mode
- Fault scenario: `cpu_stress`
- Fault timing: starts after 20 seconds, runs for 30 seconds
- CPU workers: 8
- Raw CSV files are local benchmark artifacts and are not committed

## Per-Run Summary

| run | SLO rows naive | SLO rows controlled | p95 avg naive | p95 avg controlled | inference avg naive | inference avg controlled | FPS naive | FPS controlled | max temp C | throttle rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 15 | 4 | 92.114 | 62.519 | 77.143 | 46.798 | 17.803 | 21.379 | 64.2 | 0 |
| repeat2 | 2 | 4 | 91.002 | 71.657 | 75.331 | 52.846 | 12.159 | 13.129 | 60.9 | 0 |
| repeat3 | 5 | 3 | 93.152 | 55.376 | 76.301 | 43.916 | 12.009 | 13.988 | 62.6 | 0 |
| repeat4 | 13 | 3 | 96.573 | 70.077 | 80.620 | 54.818 | 11.720 | 13.126 | 63.1 | 0 |

## Aggregate

| metric | naive | controlled | result |
|---|---:|---:|---|
| Total SLO rows | 35 | 14 | 60.0% fewer rows |
| Average p95 latency | 93.2 ms | 64.9 ms | lower in 4 of 4 runs |
| Average inference time | 77.3 ms | 49.6 ms | lower in 4 of 4 runs |
| Average FPS | 13.4 | 15.4 | higher in 4 of 4 runs |
| SLO rows improved | - | - | improved in 3 of 4 runs |
| Thermal throttle | 0 rows | 0 rows | throttle-free |

## Interpretation

The repeated runs support the claim that controlled mode improves average
latency, inference time, and FPS under CPU-stress-induced latency pressure.

The SLO-row result is more nuanced: controlled mode reduced total SLO rows from
35 to 14 and improved SLO rows in three of four pairs, but repeat2 had more SLO
rows in controlled mode than in naive mode. The correct claim is aggregate SLO
reduction across these runs, not a guarantee that every individual run reduces
SLO rows.

All runs stayed throttle-free, so this is a CPU-stress / latency-pressure result,
not a thermal-throttling result.

