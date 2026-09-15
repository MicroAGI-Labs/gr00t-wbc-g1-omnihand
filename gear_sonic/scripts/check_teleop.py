"""Collect teleop recovery evidence without sending control or restart commands.

Uses only Python's standard library. Run from the checkout used by the stack:
  python gear_sonic/scripts/check_teleop.py --hand-server-host 192.168.123.164
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.error
import urllib.request


def fetch_status(base_url, endpoint):
    try:
        with urllib.request.urlopen(base_url.rstrip('/') + endpoint, timeout=3) as response:
            data = json.load(response)
        if not isinstance(data, dict):
            raise ValueError('status response is not an object')
        return data
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {'connected': False, 'error': str(exc)}


def recovery_advice(recording, hands):
    """Report independent failures so recovering one cannot hide another."""
    advice = []
    if not recording.get('connected'):
        advice.append('Recorder offline: inspect the recorder pane and its exit error before restarting it.')
    else:
        finalizer = recording.get('finalizer') or {}
        if finalizer.get('error'):
            advice.append('Local save failed: preserve the dataset and recovery/ files; inspect the finalizer error.')
        elif finalizer.get('at_capacity'):
            advice.append('Local save queue full: wait for pending saves; inspect disk space and writer errors if stuck.')
        sources = recording.get('sources') or {}
        if not sources.get('camera'):
            advice.append('No camera samples: inspect composed_camera_server.service and the camera stream.')
        rates = recording.get('stream_rates') or {}
        if not (rates.get('manager_state') or {}).get('active'):
            advice.append('PICO manager not publishing: inspect its pane for invalid/zero-quaternion tracking; restore headset and controller tracking first.')
        if not sources.get('proprio'):
            advice.append('No robot state: inspect the C++ pane; it may still be waiting for normal SONIC arming/start.')
        if not recording.get('recording_mode_ready'):
            advice.append('Recording mode not active: in default VR3PT, start with A+B+X+Y after tracking is valid; release and re-engage the side grips.')
    connection = hands.get('connection_state')
    if connection == 'fault':
        advice.append('Hand fault is latched: inspect the reported motor/position/health error, resolve it, then use Reconnect hands.')
    elif connection == 'reconnecting':
        advice.append('Hands retrying USB transport: inspect the saved kernel log and adapter identities. If USB is present but motor replies are absent, check motor power and adapter-to-motor cabling.')
    elif not hands.get('connected'):
        advice.append('Hands unavailable or stale: inspect the remote service, SSH connection, and hand-state error before requesting reconnect.')
    elif hands.get('input_stale'):
        advice.append('Hand hardware is connected and holding; PICO intent is absent/stale. Restore tracking and use the normal grip/trigger re-engagement sequence.')
    return advice


def run_command(args, *, cwd=None, input_text=None):
    try:
        result = subprocess.run(args, cwd=cwd, input=input_text, text=True,
                                capture_output=True, timeout=20)
        return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'returncode': None, 'error': str(exc)}


USB_PROBE = r'''
import datetime, json, pathlib
root = pathlib.Path('/sys/bus/usb/devices')
devices = []
def read(path):
    try: return path.read_text().strip()
    except OSError: return None
for port in pathlib.Path('/dev/serial/by-id').glob('*'):
    if not any(serial in port.name for serial in ('FTBQ776H', 'FTBWJBC1')): continue
    tty = port.resolve().name
    node = pathlib.Path('/sys/class/tty') / tty / 'device'
    parents = []
    for parent in node.resolve().parents:
        if not (parent / 'idVendor').exists(): continue
        parents.append({'path': str(parent), **{name: read(parent / name) for name in
            ('idVendor', 'idProduct', 'product', 'serial', 'speed', 'power/control', 'power/runtime_status')}})
    devices.append({'port': str(port), 'tty': tty, 'usb_ancestors': parents})
print(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'boot_id': read(pathlib.Path('/proc/sys/kernel/random/boot_id')),
                  'uptime': read(pathlib.Path('/proc/uptime')), 'serial_devices': devices}, indent=2))
'''


def collect(output, *, base_url, hand_server_host=None, ssh_key=None, duration=5):
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    samples = []
    deadline = time.monotonic() + duration
    while True:
        samples.append({'utc': datetime.now(timezone.utc).isoformat(),
                        'recording': fetch_status(base_url, '/recording/status'),
                        'hands': fetch_status(base_url, '/hands/status')})
        if time.monotonic() >= deadline:
            break
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    (output / 'status.json').write_text(json.dumps(samples, indent=2))
    commands = {
        'checkout': ['git', 'status', '--short', '--branch'],
        'revision': ['git', 'rev-parse', 'HEAD'],
        'panes': ['tmux', 'list-panes', '-a', '-F', '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} command=#{pane_current_command}'],
        'disk': ['df', '-h', str(repo), str(Path.home())],
        'camera-service': ['journalctl', '--utc', '-u', 'composed_camera_server.service', '-n', '80', '--no-pager'],
    }
    for pane in range(5):
        commands[f'pane-{pane}'] = ['tmux', 'capture-pane', '-p', '-t', f'sonic_data_collection:0.{pane}', '-S', '-100']
    for name, command in commands.items():
        (output / f'{name}.json').write_text(json.dumps(run_command(command, cwd=repo), indent=2))
    if hand_server_host:
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5']
        if ssh_key:
            ssh += ['-i', str(ssh_key), '-o', 'IdentitiesOnly=yes']
        ssh += ['--', hand_server_host]
        remote = {
            'usb-topology': ('python3 -', USB_PROBE),
            'usb-kernel': (shlex.join(['journalctl', '--utc', '-k', '-b', '--since=-30min', '-n', '500', '--no-pager', '-o', 'short-iso']), None),
        }
        for name, (command, stdin) in remote.items():
            (output / f'{name}.json').write_text(json.dumps(run_command(ssh + [command], input_text=stdin), indent=2))
    latest = samples[-1]
    advice = recovery_advice(latest['recording'], latest['hands'])
    (output / 'summary.txt').write_text('\n'.join(advice or ['No listed recovery blocker in this snapshot; this does not validate physical motion.']) + '\n')
    return advice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--hand-server-host', help='Optional SSH target for read-only USB topology/kernel logs')
    parser.add_argument('--ssh-key', type=Path)
    parser.add_argument('--duration', type=float, default=5)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not 0 <= args.duration <= 60:
        parser.error('--duration must be between 0 and 60 seconds')
    key = args.ssh_key
    if key is None and args.hand_server_host == '192.168.123.164':
        candidate = Path.home() / '.ssh/id_ed25519_sonic_192.168.123.164'
        if candidate.is_file():
            key = candidate
    output = args.output_dir or Path('outputs/teleop-diagnostics') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    advice = collect(output, base_url=args.url, hand_server_host=args.hand_server_host,
                     ssh_key=key, duration=args.duration)
    print(f'Diagnostics saved to {output.resolve()}')
    for item in advice:
        print(f'- {item}')
    if not advice:
        print('No listed recovery blocker in this snapshot; physical motion is not tested.')


if __name__ == '__main__':
    main()
