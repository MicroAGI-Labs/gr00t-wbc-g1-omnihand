#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
sdk_dir="${repo_root}/external_dependencies/agillink_omnihand_sdk"
sdk_commit="026740d9fdd8ba32b0605fa702a992b322076f1b"
wheel_rel="linux/aarch64/python/omnihand-1.1.8-cp312-cp312-linux_aarch64.whl"
wheel_sha="3d089768492729d793c5e4b29ef23620a39fd26af27924a2f3ff78ca8f6d93ae"
venv_dir="${repo_root}/.venv_omnihand"

if ! command -v uv >/dev/null; then
    echo "uv is required to create the isolated OmniHand Python 3.12 runtime" >&2
    exit 1
fi
if [[ ! -d "${sdk_dir}/.git" ]]; then
    git clone https://github.com/AgibotTech/agillink_omnihand_sdk.git "${sdk_dir}"
fi
if [[ -n "$(git -C "${sdk_dir}" status --porcelain)" ]]; then
    echo "Refusing to change a locally modified OmniHand SDK checkout: ${sdk_dir}" >&2
    exit 1
fi
git -C "${sdk_dir}" fetch --depth 1 origin "${sdk_commit}"
git -C "${sdk_dir}" checkout --detach "${sdk_commit}"
if [[ "$(git -C "${sdk_dir}" rev-parse HEAD)" != "${sdk_commit}" ]]; then
    echo "OmniHand SDK checkout does not match the admitted commit" >&2
    exit 1
fi

wheel="${sdk_dir}/${wheel_rel}"
printf '%s  %s\n' "${wheel_sha}" "${wheel}" | sha256sum --check --status || {
    echo "OmniHand wheel checksum mismatch: ${wheel}" >&2
    exit 1
}

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    uv venv --python 3.12 "${venv_dir}"
elif ! "${venv_dir}/bin/python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
    echo "Existing ${venv_dir} is not a Python 3.12 environment" >&2
    exit 1
fi
uv pip install --python "${venv_dir}/bin/python" numpy==1.26.4 pyzmq msgpack
uv pip install --python "${venv_dir}/bin/python" --no-deps -e "${repo_root}/gear_sonic"
uv pip install --python "${venv_dir}/bin/python" "${wheel}"
"${venv_dir}/bin/python" -c 'import omnihand; print("OmniHand SDK import OK")'
