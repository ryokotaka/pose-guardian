# Architecture

[`examples/run_controlled.py`](../examples/run_controlled.py) connects camera
capture, resource monitoring, model selection, and CSV logging.

1. `Camera` captures frames on a background thread. Frame IDs let the main loop
   avoid processing the same frame twice.
2. `ResourceMonitor` samples temperature, CPU and memory use, throttle flags, and
   optional PMIC rail estimates. The loop also supplies frame timings for FPS.
3. For each new frame, `ResourceController` evaluates the latest snapshot and
   recent inference latencies, excluding preprocessing. The default latency window is
   60 frames, with at least 20 samples required for latency-based decisions.
4. The loop applies the action and runs `PoseEstimator`, which keeps both MoveNet
   models loaded. A model switch selects an existing interpreter.
5. The loop writes resource readings, rolling latency, controller state, and
   actions to CSV, once per second by default. The comparison script reads those
   CSVs to produce a Markdown report and optional plot.

In `naive` mode, the same measurements are collected but the initial model stays
fixed. In `controlled` mode, the controller moves between normal, degraded, and
critical states. Pressure can move it directly from normal to critical; recovery
passes through degraded and uses lower thresholds plus a hold timer.

CPU usage is recorded but does not directly trigger the controller. Temperature,
memory, rolling latency, and throttle flags do. The defaults and transition rules
live in [`src/resource_controller.py`](../src/resource_controller.py).

`FaultInjector` supports CPU stress and memory pressure for the controlled-run
script. Camera disconnect injection is not connected to this loop.
`MetricsCollector` is a separate frame counter; CSV export is implemented in the
example scripts, not in that class.
