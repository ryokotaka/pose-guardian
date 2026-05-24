#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

THUNDER_URL="https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
LIGHTNING_URL="https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"

download_if_missing() {
  local url="$1"
  local output="$2"

  if [[ -s "$output" ]]; then
    echo "exists: $output"
    return
  fi

  echo "downloading: $output"
  curl --fail --location --show-error --output "$output" "$url"
}

download_if_missing "$THUNDER_URL" "movenet_thunder.tflite"
download_if_missing "$LIGHTNING_URL" "movenet_lightning.tflite"

ls -lh ./*.tflite

cat <<'MSG'

Model binaries are local-only. Do not commit them.
They are ignored by ../.gitignore via models/*.tflite.
MSG
