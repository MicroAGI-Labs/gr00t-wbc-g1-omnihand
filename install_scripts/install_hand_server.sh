#!/usr/bin/env bash
# Install only the CPU hand service. No device access or service activation.
set -euo pipefail
hand_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
hand_python="${HAND_SERVER_PYTHON:-python3.10}"
if ! command -v "$hand_python" >/dev/null 2>&1; then
    hand_python="$HOME/.local/share/uv/python/cpython-3.10-linux-aarch64-gnu/bin/python3.10"
fi
if [[ ! -x "$hand_python" ]] && ! command -v "$hand_python" >/dev/null 2>&1; then
    echo 'Set HAND_SERVER_PYTHON to a Python 3.10+ executable.' >&2
    exit 1
fi
"$hand_python" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
"$hand_python" -m venv "$hand_root/.venv_hands"
hand_pip_options=(--only-binary=:all:)
if [[ -n "${HAND_SERVER_WHEELHOUSE:-}" ]]; then
    hand_pip_options+=(--no-index --find-links "$HAND_SERVER_WHEELHOUSE")
fi
"$hand_root/.venv_hands/bin/python" -m pip install "${hand_pip_options[@]}" \
    numpy==1.26.4 pyzmq==27.2.0 msgpack==1.2.2
bash "$hand_root/install_scripts/install_dex1.sh"
cd "$hand_root"
"$hand_root/.venv_hands/bin/python" -m gear_sonic.end_effectors.server --check-only
