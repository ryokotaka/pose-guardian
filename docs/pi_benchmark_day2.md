# Pi Benchmark Day 2

Day 2 compares Mac and Raspberry Pi 5 inference latency with the same fixed
reference clips. The clips live under `benchmark/reference_clips/` locally and
must not be committed.

## Procedure

1. Confirm Pi power before measuring:

   ```bash
   vcgencmd measure_temp
   vcgencmd get_throttled
   od -An -tu4 --endian=big /proc/device-tree/chosen/power/max_current
   sudo dmesg --color=never | grep -iE 'under.?voltage|voltage|throttl' | tail -n 10 || true
   ```

2. Copy reference clips to the Pi:

   ```bash
   scp benchmark/reference_clips/clip_*.mp4 ryo@edge-pi.local:~/edge-inference-guardian/benchmark/reference_clips/
   ```

3. Run the benchmark on Mac and Pi:

   ```bash
   .venv/bin/python examples/benchmark_clips.py --csv-output metrics/day2_<host>.csv
   ```

4. Keep CSV files local. Do not commit `metrics/*.csv` or mp4 files.

## Conditions

| Item | Mac | Pi |
|---|---|---|
| Host | martin11.local | edge-pi |
| OS / arch | macOS 26.5.1 / arm64 | Raspberry Pi OS Bookworm / aarch64 |
| Python | 3.11.15 | 3.11.2 |
| Runtime | ai-edge-litert | ai-edge-litert |
| Power | MacBook internal | 5V/5A, `max_current=5000` |
| Throttle before | n/a | `0x0` |
| Throttle after | n/a | `0xe0008` |
| Temperature before/after | n/a | `46.1'C` -> `83.4'C` |
| Cooling note | n/a | No active cooler attached, intentionally measured as passive-cooling baseline |

## Results

First run on 2026-06-03. This is the **passive-cooling baseline**: the Pi was
powered correctly (`max_current=5000`, no undervoltage) but had no active cooler
attached. The run ended with thermal throttle flags, which is part of the result.

| Clip | Model | Mac inf avg/med/p95 ms | Pi inf avg/med/p95 ms | Mac pre avg ms | Pi pre avg ms | Mac FPS | Pi FPS | Pi/Mac avg |
|---|---|---:|---:|---:|---:|---:|---:|---|
| still | Thunder | 3.92 / 3.71 / 4.72 | 35.09 / 34.98 / 35.95 | 1.81 | 1.74 | 168.33 | 26.22 | 9.0x |
| still | Lightning | 1.67 / 1.63 / 1.94 | 8.72 / 8.72 / 8.88 | 1.72 | 1.77 | 280.34 | 85.80 | 5.2x |
| slow | Thunder | 3.70 / 3.63 / 3.98 | 39.64 / 37.47 / 44.25 | 1.81 | 2.20 | 174.92 | 23.11 | 10.7x |
| slow | Lightning | 1.75 / 1.64 / 2.18 | 9.80 / 9.13 / 10.97 | 1.76 | 2.16 | 270.56 | 75.02 | 5.6x |
| fast | Thunder | 3.77 / 3.68 / 4.16 | 42.15 / 44.07 / 44.35 | 1.82 | 2.45 | 172.44 | 21.69 | 11.2x |
| fast | Lightning | 1.69 / 1.62 / 2.10 | 10.14 / 10.85 / 10.96 | 1.73 | 2.30 | 277.08 | 72.37 | 6.0x |

Summary:

| Model | Mac avg inference | Pi avg inference | Pi avg effective FPS | Pi/Mac avg |
|---|---:|---:|---:|---:|
| Thunder | 3.80 ms | 38.96 ms | 23.67 FPS | 10.3x |
| Lightning | 1.70 ms | 9.55 ms | 77.73 FPS | 5.6x |

Thermal flags:

```text
before: temp=46.1'C, throttled=0x0, max_current=5000
after:  temp=83.4'C, throttled=0xe0008, max_current=5000
```

`0xe0008` means the Pi hit the soft temperature limit during or immediately
after the benchmark, and also has historical frequency-cap/throttle/soft-temp
flags. There is no undervoltage bit.

## Interpretation

- Fixed clips make Mac/Pi comparison fair because input frames are identical.
- `run_demo.py` remains a live-camera smoke test. It is not the main benchmark.
- If undervoltage appears, mark the results as power-noisy and repeat after
  fixing power/cabling.
- If thermal throttling appears, record the cooling condition and compare
  passive vs active cooling.
- This first run is **thermal-limited**, not power-noisy. Use it as the
  no-active-cooler baseline. Run the same benchmark again after attaching the
  active cooler to compare passive vs active cooling.
- Lightning already shows enough headroom for real-time inference on Pi. Thunder
  is below 30 FPS in this first run and may be worse under live-camera overhead,
  so Week 3 controller thresholds should treat Lightning as the thermal fallback.
