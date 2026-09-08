#!/usr/bin/env bash
# Launch the existing teleop dashboard with the measured DEX 1 USB pair.
set -euo pipefail
dex1_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$dex1_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$dex1_root"
exec .venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py \
    --hand-backend dex1 --remote-ui "$@"
