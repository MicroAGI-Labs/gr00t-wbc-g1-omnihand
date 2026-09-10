#!/usr/bin/env bash
# Pair this Thor account with the Orin once; never store its password.
set -euo pipefail
orin_host="${1:-192.168.123.164}"
if [[ ! "$orin_host" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || [[ $# -gt 1 ]]; then
    echo "Usage: bash tools/setup_orin_ssh.sh [ORIN_IP_OR_HOSTNAME]" >&2
    exit 2
fi
orin_key="$HOME/.ssh/id_ed25519_sonic_$orin_host"
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
if [[ ! -f "$orin_key" ]]; then
    ssh-keygen -q -t ed25519 -N '' -C "SONIC Thor to $orin_host" -f "$orin_key"
fi
if [[ ! -f "$orin_key.pub" ]]; then
    ssh-keygen -y -f "$orin_key" > "$orin_key.pub"
fi
orin_ssh_options=(-i "$orin_key" -o IdentitiesOnly=yes -o ConnectTimeout=10)
if ! ssh "${orin_ssh_options[@]}" -o BatchMode=yes -- "$orin_host" true 2>/dev/null; then
    echo "Pairing with Orin. Enter the Orin account password once when prompted."
    ssh-copy-id -i "$orin_key.pub" -o "IdentityFile=$orin_key" \
        -o IdentitiesOnly=yes -o ConnectTimeout=10 "$orin_host"
fi
ssh "${orin_ssh_options[@]}" -o BatchMode=yes -- "$orin_host" true
echo "Orin paired. Future collection launches will not ask for its password."
