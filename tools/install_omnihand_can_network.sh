#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
source_dir="${repo_root}/configs/hosts/thor/systemd-networkd"
target_dir="/etc/systemd/network"
dry_run=0

if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
elif [[ "$#" -ne 0 ]]; then
    echo "Usage: sudo $0 [--dry-run]" >&2
    exit 2
fi
files=(20-omnihand-right.link 20-omnihand-left.link 20-omnihand-right.network 20-omnihand-left.network)
if [[ "${dry_run}" -eq 1 ]]; then
    for filename in "${files[@]}"; do
        echo "would install ${source_dir}/${filename} -> ${target_dir}/${filename}"
    done
    exit 0
fi
if [[ "${EUID}" -ne 0 ]]; then
    echo "Run as root: sudo $0" >&2
    exit 2
fi
for filename in "${files[@]}"; do
    source_path="${source_dir}/${filename}"
    target_path="${target_dir}/${filename}"
    if [[ -f "${target_path}" ]] && ! cmp -s -- "${source_path}" "${target_path}"; then
        backup_path="${target_path}.pre-omnihand"
        if [[ ! -e "${backup_path}" ]]; then
            install -m 0644 -- "${target_path}" "${backup_path}"
            echo "backed up ${target_path} -> ${backup_path}"
        else
            echo "preserving existing backup ${backup_path}"
        fi
    fi
    install -D -m 0644 -- "${source_path}" "${target_path}"
done
udevadm control --reload
networkctl reload
udevadm settle
"${repo_root}/tools/bringup_omnihand_canfd.sh" can10
"${repo_root}/tools/bringup_omnihand_canfd.sh" can11
echo "OmniHand SocketCAN ready: right=can10 left=can11"
