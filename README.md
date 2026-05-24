# Edge Inference Guardian

Edge Inference Guardian is a small control layer for reliable AI inference under camera, latency, and resource constraints.

## Day 1 Status

- Local Python 3.11 virtual environment target: `.venv`
- Core pipeline modules are scaffolded under `src/`
- Benchmark, examples, tests, models, metrics, and docs directories are prepared
- Model files, reference clips, benchmark results, metrics, and local environment files are excluded from Git

## Day 2 Status

- MoveNet Thunder and Lightning TFLite models can be downloaded from the official TF Hub URLs with `models/download_models.sh`
- `examples/sanity_check.py` runs single-image inference with `ai-edge-litert`
- Local output images are written under `tmp/`, which is ignored by Git

```bash
.venv/bin/python -m pip install ai-edge-litert
./models/download_models.sh
curl --fail --location --show-error --output tmp/test.jpg \
  https://images.pexels.com/photos/4384679/pexels-photo-4384679.jpeg
.venv/bin/python examples/sanity_check.py tmp/test.jpg models/movenet_thunder.tflite \
  --output tmp/sanity_check_output_thunder.jpg
.venv/bin/python examples/sanity_check.py tmp/test.jpg models/movenet_lightning.tflite \
  --output tmp/sanity_check_output_lightning.jpg
```
