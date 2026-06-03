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
| Host | _ | edge-pi |
| OS / arch | _ | Raspberry Pi OS Bookworm / aarch64 |
| Python | _ | 3.11.2 |
| Runtime | ai-edge-litert | ai-edge-litert |
| Power | MacBook internal | 5V/5A, `max_current=5000` |
| Throttle before | n/a | `0x0` |
| Throttle after | n/a | _ |

## Results

Fill this table from `examples/benchmark_clips.py` output.

| Clip | Model | Mac inf avg/med/p95 ms | Pi inf avg/med/p95 ms | Mac pre avg ms | Pi pre avg ms | Mac FPS | Pi FPS | Pi throttle |
|---|---|---:|---:|---:|---:|---:|---:|---|
| still | Thunder | _ | _ | _ | _ | _ | _ | _ |
| still | Lightning | _ | _ | _ | _ | _ | _ | _ |
| slow | Thunder | _ | _ | _ | _ | _ | _ | _ |
| slow | Lightning | _ | _ | _ | _ | _ | _ | _ |
| fast | Thunder | _ | _ | _ | _ | _ | _ | _ |
| fast | Lightning | _ | _ | _ | _ | _ | _ | _ |

## Interpretation

- Fixed clips make Mac/Pi comparison fair because input frames are identical.
- `run_demo.py` remains a live-camera smoke test. It is not the main benchmark.
- If Pi throttling or undervoltage appears, mark the results as power-noisy and
  repeat after fixing power/cabling.
