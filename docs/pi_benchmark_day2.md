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

| Item | Mac | Pi passive | Pi active cooler |
|---|---|---|---|
| Host | martin11.local | edge-pi | edge-pi |
| OS / arch | macOS 26.5.1 / arm64 | Raspberry Pi OS Bookworm / aarch64 | Raspberry Pi OS Bookworm / aarch64 |
| Python | 3.11.15 | 3.11.2 | 3.11.2 |
| Runtime | ai-edge-litert | ai-edge-litert | ai-edge-litert |
| Power | MacBook internal | 5V/5A, `max_current=5000` | 5V/5A, `max_current=5000` |
| Throttle before | n/a | `0x0` | `0x0` |
| Throttle after | n/a | `0xe0008` | `0x0` |
| Temperature before/after | n/a | `46.1'C` -> `83.4'C` | `39.5'C` -> `61.5'C` |
| Cooling note | n/a | No active cooler attached, intentionally measured as passive-cooling baseline | Active cooler attached, `pwm-fan` detected |

## Results

Runs on 2026-06-03. The first Pi run is the **passive-cooling baseline**: the Pi
was powered correctly (`max_current=5000`, no undervoltage) but had no active
cooler attached. The second Pi run uses the same clips and script after
attaching the active cooler.

### Passive-cooling baseline

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

### Active cooler run

| Clip | Model | Pi active inf avg/med/p95 ms | Pi active pre avg ms | Pi active FPS | Passive inf avg ms | Active vs passive |
|---|---|---:|---:|---:|---:|---:|
| still | Thunder | 35.03 / 35.02 / 35.20 | 1.73 | 26.30 | 35.09 | 1.00x |
| still | Lightning | 8.65 / 8.65 / 8.74 | 1.68 | 86.99 | 8.72 | 0.99x |
| slow | Thunder | 34.93 / 34.92 / 35.09 | 1.72 | 26.37 | 39.64 | 0.88x |
| slow | Lightning | 8.65 / 8.65 / 8.73 | 1.67 | 86.53 | 9.80 | 0.88x |
| fast | Thunder | 34.93 / 34.93 / 35.12 | 1.72 | 26.36 | 42.15 | 0.83x |
| fast | Lightning | 8.65 / 8.65 / 8.73 | 1.68 | 86.75 | 10.14 | 0.85x |

Active cooler summary:

| Model | Pi active avg inference | Pi active avg effective FPS | Passive avg inference | Passive avg effective FPS |
|---|---:|---:|---:|---:|
| Thunder | 34.96 ms | 26.35 FPS | 38.96 ms | 23.67 FPS |
| Lightning | 8.65 ms | 86.76 FPS | 9.55 ms | 77.73 FPS |

Thermal flags:

```text
before: temp=39.5'C, throttled=0x0, max_current=5000
after:  temp=61.5'C, throttled=0x0, max_current=5000
```

The active cooler did not make the Pi fundamentally match the Mac. Its main
value is that the fixed benchmark completed without thermal throttle and with a
much lower final temperature.

### Live-camera smoke

After plugging in the Logitech C270, `v4l2-ctl --list-devices` showed the camera
as `/dev/video0` and `/dev/video1`; `/dev/video0` is the capture node. The
headless live-camera smoke test passed:

```text
python examples/run_demo.py --device 0 --no-display --duration 30

finished: elapsed=30.1s  frames=572  last_model=thunder
last_inference_ms=36.9  last_preprocess_ms=1.7
last_avg_conf=0.059  fps_ema=19.86
camera stats: {'frames_read': 574, 'frames_dropped': 0, 'frame_id': 574, 'is_alive': False}
after: temp=57.6'C, throttled=0x0, max_current=5000
```

The low `last_avg_conf` is not a benchmark failure; it depends on what the
camera could see at the end of the smoke test. The purpose of this smoke test is
to prove that the live camera path runs on Pi without crashing.

## Interpretation

- Fixed clips make Mac/Pi comparison fair because input frames are identical.
- `run_demo.py` remains a live-camera smoke test. It is not the main benchmark.
- If undervoltage appears, mark the results as power-noisy and repeat after
  fixing power/cabling.
- If thermal throttling appears, record the cooling condition. The passive run
  did throttle; the active cooler run did not.
- The passive run is **thermal-limited**, not power-noisy. The active cooler run
  demonstrates that cooling can remove the thermal throttle under the same
  fixed-clip workload.
- Lightning shows enough headroom for real-time inference on Pi. Thunder remains
  below 30 FPS even with the active cooler, and the live-camera smoke reached
  about 20 FPS. Week 3 controller thresholds should treat Lightning as the
  thermal/performance fallback.
