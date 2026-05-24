#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

cat <<'MSG'
Place the MoveNet TFLite files in this directory:

- movenet_thunder.tflite
- movenet_lightning.tflite

Recommended source: https://www.kaggle.com/models/google/movenet
Do not commit model binaries. They are ignored by .gitignore.
MSG

ls -la ./*.tflite 2>/dev/null || true
