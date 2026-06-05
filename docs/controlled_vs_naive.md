# Controlled vs Naive (CPU-stress fault)

SLO threshold: `200.0 ms`

## Conditions

- Device: Raspberry Pi 5 with active cooler attached
- Camera: USB camera via OpenCV
- Initial model: `thunder`
- Run duration: 90 seconds per mode
- Fault scenario: `cpu_stress`
- Fault timing: starts after 20 seconds, runs for 30 seconds
- CPU workers: 8
- Raw CSV files are local benchmark artifacts and are not committed

## Summary

| run | rows | duration_s | mode | models | states | actions | p95_avg_ms | p95_max_ms | SLO_rows | SLO_pct | inference_avg_ms | fps_avg | temp_max_c | throttle_rows | switches |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 91 | 90.029 | naive:91 | thunder:91 | normal:91 | none:91 | 92.114 | 205.677 | 15 | 16.667 | 77.143 | 17.803 | 64.2 | 0 | 0 |
| controlled | 90 | 89.044 | controlled:90 | lightning:25, thunder:65 | degraded:25, normal:65 | none:86, switch_to_heavy:2, switch_to_light:2 | 62.519 | 205.315 | 4 | 4.494 | 46.798 | 21.379 | 61.5 | 0 | 4 |

## Interpretation

- SLO violations fell from 15 rows to 4 rows, a 73.3% reduction.
- Average recent p95 latency fell from 92.114 ms to 62.519 ms.
- Average inference time fell from 77.143 ms to 46.798 ms.
- Average FPS improved from 17.803 to 21.379, a 20.1% gain.
- Both runs stayed throttle-free (`throttle_rows=0`).

The controlled run did not eliminate every SLO violation. It reduced the violation count and recovered FPS/inference latency under the same CPU stress. That is the correct claim for this data.

The controlled run switched four times: it returned to Thunder once while the fault was still active, then degraded again. This is a tuning opportunity for a future recovery-policy pass.

## Input CSV

- naive: `metrics/naive_cpu_stress.csv`
- controlled: `metrics/controlled_cpu_stress.csv`

## Plot

- `docs/assets/naive_vs_controlled_cpu_stress.png`

## Notes

- Raw CSV files under `metrics/` are local benchmark outputs and should not be committed.
- A selected plot under `docs/assets/` may be committed when it is used by the README.
- `SLO_rows` counts rows where `recent_latency_p95_ms` is greater than the SLO threshold.
