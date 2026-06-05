# Resource Monitor

`ResourceMonitor` samples resource state on both Raspberry Pi and local
development hosts, while Pi-specific commands fall back safely when unavailable.

## Implemented Surface

`ResourceSnapshot` now contains:

- `timestamp`
- `cpu_temp_celsius`
- `cpu_usage_percent`
- `cpu_freq_mhz`
- `memory_used_percent`
- `memory_used_bytes`
- `memory_available_bytes`
- `is_throttled`
- `throttle_flags`
- `fps`
- `pmic_rail_estimate_watts`

`ResourceMonitor` supports:

- `start()` / `stop()` background sampling
- `snapshot()` for the latest immutable sample
- `record_inference(latency_ms)` for rolling FPS calculation
- `history(seconds)` for recent samples

`read_resources()` remains as a one-shot convenience wrapper.

## Platform Behavior

| Metric | Raspberry Pi | Non-Pi fallback |
|---|---|---|
| CPU temperature | `vcgencmd measure_temp`, then sysfs fallback | psutil sensors if available, otherwise `0.0` |
| CPU usage | `psutil.cpu_percent()` | same |
| CPU frequency | `psutil.cpu_freq()` | same, or `0.0` |
| Memory | `psutil.virtual_memory()` | same |
| Throttle flags | `vcgencmd get_throttled` | `0` |
| PMIC rail estimate | `vcgencmd pmic_read_adc` | `None` |

`is_throttled` checks only current frequency-cap, throttle, and soft-temperature
bits:

```text
is_throttled = (throttle_flags & 0x0E) != 0
```

Historical bits remain available through `throttle_flags` but do not make
`is_throttled` true by themselves.

## Controller Follow-Up

`ResourceController` was updated to read:

- `cpu_usage_percent`
- `memory_used_percent`
- `is_throttled`

This avoids the broken intermediate state where `ResourceSnapshot` is expanded
but the controller still reads the old placeholder fields.

## Verification

Mac test run:

```text
43 passed
```

Pi smoke snapshot:

```text
cpu_temp_celsius=40.6
vcgencmd measure_temp=temp=41.1'C
is_throttled=False
throttle_flags=0x0
pmic_rail_estimate_watts=1.92193239883672
```

Pi monitor/FPS smoke:

```text
fps=19.98
history=4
temp=41.7
throttle=False 0x0
pmic=1.9534818051947898
```

Pi pytest was not run because the Pi virtual environment intentionally does not
install dev-only `pytest`. The direct Pi smoke covers the Pi-specific
`vcgencmd`, PMIC, FPS, and history paths.
