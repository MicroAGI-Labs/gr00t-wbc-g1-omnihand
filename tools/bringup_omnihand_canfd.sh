#!/usr/bin/env bash
set -euo pipefail

interface="${1:-can10}"
if [[ "${EUID}" -ne 0 ]]; then
    echo "Run as root: sudo $0 ${interface}" >&2
    exit 2
fi
if [[ ! "${interface}" =~ ^can(10|11)$ ]]; then
    echo "Refusing unexpected OmniHand interface: ${interface}" >&2
    exit 2
fi
device_root="/sys/class/net/${interface}/device"
driver="$(readlink -f "${device_root}/driver" 2>/dev/null || true)"
if [[ "${driver}" != */gs_usb ]]; then
    echo "Refusing: ${interface} is not attached to gs_usb (${driver:-none})" >&2
    exit 1
fi

ip link set "${interface}" down 2>/dev/null || true
ip link set "${interface}" type can bitrate 1000000 sample-point 0.80 \
    dbitrate 5000000 dsample-point 0.75 fd on
ip link set "${interface}" txqueuelen 1000
ip link set "${interface}" up

mtu="$(<"/sys/class/net/${interface}/mtu")"
flags="$(( $(<"/sys/class/net/${interface}/flags") ))"
if [[ "${mtu}" -ne 72 || $(( flags & 1 )) -eq 0 ]]; then
    echo "CAN-FD admission failed for ${interface}: mtu=${mtu} flags=${flags}" >&2
    exit 1
fi
ip -details -statistics link show "${interface}"
