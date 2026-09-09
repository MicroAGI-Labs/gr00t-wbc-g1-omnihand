#!/usr/bin/env bash
# Pair the Thor account with an Orin once; no password is stored.
set -euo pipefail
orin_host="${1:-192.168.123.164}"
if [[ $# -gt 1 || ! "$orin_host" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Usage: bash tools/setup_orin_ssh.sh [ORIN_HOST]" >&2
    exit 2
fi
orin_key="$HOME/.ssh/id_ed25519_sonic_$orin_host"
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
if [[ ! -f "$orin_key" ]]; then
    ssh-keygen -q -t ed25519 -N '' -C "SONIC Thor to $orin_host" -f "$orin_key"
fi
[[ -f "$orin_key.pub" ]] || ssh-keygen -y -f "$orin_key" > "$orin_key.pub"
orin_ssh_options=(-i "$orin_key" -o IdentitiesOnly=yes -o ConnectTimeout=10)
if ! ssh "${orin_ssh_options[@]}" -o BatchMode=yes -- "$orin_host" true 2>/dev/null; then
    echo "Pairing with Orin; enter its password once."
    ssh-copy-id -i "$orin_key.pub" "${orin_ssh_options[@]}" "$orin_host"
fi
ssh "${orin_ssh_options[@]}" -o BatchMode=yes -- "$orin_host" true
echo "Orin paired. Future launches use the dedicated key."
