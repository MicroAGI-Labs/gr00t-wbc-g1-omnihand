#!/usr/bin/env bash
# Daily Thor command for cameras/recording with DEX1 grippers on Orin.
set -euo pipefail
collection_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$collection_root/gear_sonic/scripts/launch_data_collection.py" \
    --hand-backend dex1 --hand-server-host 192.168.123.164 \
    --record-wrist-cameras --data-exporter-frequency 50 "$@"
