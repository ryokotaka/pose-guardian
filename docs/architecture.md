# Architecture

The pipeline is evolving from `camera -> pose estimator -> visualizer` into
`camera -> pose estimator -> resource monitor -> resource controller -> metrics`.

Current implemented layers:

- `Camera`: threaded OpenCV capture with fresh-frame IDs.
- `PoseEstimator`: MoveNet Thunder/Lightning wrapper.
- `ResourceMonitor`: temperature, CPU, memory, throttle, FPS, and optional PMIC
  rail estimate snapshots.
