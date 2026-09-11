#!/usr/bin/env bash
# Thor cameras/collection/UI with DEX 1 grippers attached to Orin.
set -euo pipefail
collection_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$collection_root/tools/teleop_dex1.sh" \
    --hand-server-host 192.168.123.164 \
    --record-wrist-cameras --data-exporter-frequency 50 "$@"
