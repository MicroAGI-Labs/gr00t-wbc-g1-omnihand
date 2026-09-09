#!/usr/bin/env bash
# Build the USB worker without opening a device or changing the system SDK.
set -euo pipefail
dex1_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dex1_sdk="$dex1_root/external_dependencies/dex1_1_service"
dex1_build="$dex1_root/build/dex1"
dex1_commit=2986d26eefa4136d4777493e6fd0b8bac7a4c6ae
mkdir -p "$dex1_build"
if [[ ! -d "$dex1_sdk/.git" ]]; then
    git clone --no-checkout "${DEX1_SDK_SOURCE:-https://github.com/unitreerobotics/dex1_1_service.git}" "$dex1_sdk"
    git -C "$dex1_sdk" checkout --detach "$dex1_commit"
fi
if [[ "$(git -C "$dex1_sdk" rev-parse HEAD)" != "$dex1_commit" ]] ||
   [[ -n "$(git -C "$dex1_sdk" status --porcelain)" ]]; then
    echo "DEX 1 SDK must be clean at $dex1_commit: $dex1_sdk" >&2
    exit 1
fi
dex1_flags=()
case "$(uname -m)" in
    aarch64)
        for dex1_package in libserialport0_0.1.1-3_arm64.deb libserialport-dev_0.1.1-3_arm64.deb; do
            dpkg-deb -x "$dex1_sdk/lib/$dex1_package" "$dex1_build/deps"
        done
        dex1_arch=Arm64
        dex1_flags+=(-I"$dex1_build/deps/usr/include" -L"$dex1_build/deps/usr/lib/aarch64-linux-gnu")
        dex1_rpath="$dex1_sdk/lib:$dex1_build/deps/usr/lib/aarch64-linux-gnu"
        ;;
    x86_64)
        dex1_arch=Linux64
        dex1_rpath="$dex1_sdk/lib"
        # On x86 install libserialport-dev through your system package manager.
        ;;
    *) echo "Unsupported DEX 1 worker architecture" >&2; exit 1 ;;
esac
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
    -I"$dex1_sdk/include" "${dex1_flags[@]}" \
    "$dex1_root/gear_sonic/end_effectors/native/dex1_worker.cpp" \
    -L"$dex1_sdk/lib" -Wl,--disable-new-dtags,-rpath,"$dex1_rpath" \
    -l"UnitreeMotorSDK_$dex1_arch" -lserialport -pthread \
    -o "$dex1_build/dex1_worker.tmp"
"$dex1_build/dex1_worker.tmp" --self-test
mv "$dex1_build/dex1_worker.tmp" "$dex1_build/dex1_worker"
echo "DEX 1 worker ready: $dex1_build/dex1_worker"
