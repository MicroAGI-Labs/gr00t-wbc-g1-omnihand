"""
All-in-one tmux launcher for SONIC data collection.

Starts the full data collection stack in one visible tmux dashboard. With
``--sim --hand-backend omnihand`` the dashboard has all six component panes:

    Window 0 — data_collection:
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 1: Teleop        │
    ├───────────────────────┼───────────────────────┤
    │ Pane 2: Data Exporter │ Pane 3: Camera Viewer │
    ├───────────────────────┼───────────────────────┤
    │ Pane 4: MuJoCo Sim    │ Pane 5: OmniHand      │
    └───────────────────────┴───────────────────────┘

The simulator and hand panes are included only when their corresponding
options are enabled. Real-hardware runs also include a read-only camera-server
log pane; systemd remains the owner of the camera process.

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_pico.sh            -> .venv_teleop
        bash install_scripts/install_data_collection.sh -> .venv_data_collection
    - gear_sonic_deploy built (see docs)
    - For sim: .venv_sim must exist (see install instructions)

Usage (from repo root — no venv activation needed):
    python gear_sonic/scripts/launch_data_collection.py                          # real robot (default)
    python gear_sonic/scripts/launch_data_collection.py --sim                    # MuJoCo sim
    python gear_sonic/scripts/launch_data_collection.py --no-camera-viewer       # skip viewer
    # In-process CloudXR / DeviceIO:
    python gear_sonic/scripts/launch_data_collection.py --pico-input-source isaac-teleop
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Literal


def _bootstrap_venv():
    """Re-exec with the .venv_data_collection Python if tyro is not available."""
    try:
        import tyro  # noqa: F401

        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_data_collection" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: tyro is not installed and .venv_data_collection not found.\n"
            "  Run: bash install_scripts/install_data_collection.sh"
        )
        sys.exit(1)

    print(f"Re-launching with {venv_python} ...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()

# Worktrees may reuse installed environments whose editable package points to
# another checkout. All panes must import the source selected by this launcher.
_SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _SOURCE_ROOT)
os.environ["PYTHONPATH"] = _SOURCE_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import tyro  # noqa: E402

from gear_sonic.utils.data_collection.hub_config import DEFAULT_TASK_PROMPT  # noqa: E402


def _get_local_ip() -> str:
    """Best-effort detection of the PC's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


@dataclass
class DataCollectionLaunchConfig:
    """CLI config for the all-in-one data collection tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim (deploy.sh sim) instead of real robot."""

    # C++ deploy options
    deploy_input_type: str = "zmq_manager"
    """Input type for the C++ deploy (zmq_manager, keyboard, etc.)."""

    deploy_zmq_host: str = "localhost"
    """ZMQ host for the C++ deploy to listen on."""

    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh (e.g., 'policy/checkpoints/my_model/model_step_100000').
    Leave empty to use the deploy.sh default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh. Leave empty for default."""

    hand_backend: Literal["dex3", "omnihand", "dex1", "none"] = "dex3"
    """Hand owner. OmniHand uses the external controller in sim and hardware."""

    sender_time_recording: bool = False
    """Opt in to producer-time recording; requires a dataset with synchronization metadata."""

    synchronization_delay: float = 0.1
    synchronization_wait_timeout: float = 0.25
    hand_clock_port: int = 5574

    hand_server_host: str | None = None
    """Launch DEX 1 hands on this host over SSH; omit to launch locally."""

    start_hand_server: bool = True
    """Start remote hands in the dashboard; disable to use an existing service."""

    hand_server_repo: str = "/home/unitree/gr00t-wbc-g1-omnihand"
    """Prepared checkout on the remote hand host (SSH uses your normal user/config)."""

    omnihand_close_scale: float = 1.0
    """Fraction of the calibrated O10 closing range used for hardware motion."""

    omnihand_sim_close_scale: float = 1.0
    """Fraction of the O10 closed pose used by the Atlas MuJoCo model."""

    omnihand_transition_duration: float = 0.2
    """Seconds for OmniHand to open or close; lower values respond faster."""

    dex1_transition_duration: float = 1.35
    """Seconds per DEX 1 stroke (1.35–30) for locally or SSH-launched hands."""

    check_only: bool = False
    """Check launch prerequisites and print the hand command without starting services."""

    omnihand_left_interface: str = "can11"
    """Serial-bound SocketCAN interface for the physical left O10."""

    omnihand_right_interface: str = "can10"
    """Serial-bound SocketCAN interface for the physical right O10."""

    hand_intent_port: int = 5569
    """Dedicated latest-only hand-intent ZMQ port."""

    hand_state_port: int = 5570
    """External-hand state port shared by the exporter and browser UI."""

    hand_control_port: int = 5572
    """Browser UI command port used to request a clean hand-worker restart."""

    teleop_control_port: int = 5573
    """Browser UI command port used to request a smooth return to idle."""

    # Teleop streamer options
    pico_manager: bool = True
    """Run pico_manager_thread_server with --manager flag."""

    body_control_mode: Literal[
        "vr3pt-slow-planner", "ik-upper-slow-planner", "full-smpl"
    ] = "vr3pt-slow-planner"
    """PICO body tracking and locomotion ownership used by data collection."""

    pico_input_source: str = "xrt"
    """Teleop input source for pico_manager_thread_server.py (xrt or isaac-teleop)."""

    pico_vis_vr3pt: bool = False
    """Enable VR 3-point visualization on the teleop streamer."""

    pico_vis_smpl: bool = False
    """Enable SMPL visualization on the teleop streamer."""

    pico_waist_tracking: bool = False
    """Enable waist tracking on the teleop streamer."""

    idle_base_transition_duration: float = 2.0
    """Seconds for smooth arm motion into and out of the teleop alignment pose."""

    disable_vr_motion_limiter: bool = True
    """Bypass VR motion filtering; --no-disable-vr-motion-limiter enables it. Pico stale-input hold stays active."""

    vr_max_speed: float = 0.5
    """Operating Cartesian translation speed limit in m/s (hard ceiling = 1.5x)."""

    vr_max_acceleration: float = 0.9
    """Operating Cartesian translation acceleration in m/s^2 (hard ceiling = 1.5x)."""

    vr_max_jerk: float = 9.0
    """Fixed Cartesian translation jerk ceiling in m/s^3."""

    vr_max_angular_jerk_deg: float = 3600.0
    """Fixed Cartesian angular jerk ceiling in degrees/s^3."""

    vr_motion_log_dir: str | None = None
    """Optional directory for full-rate before/after VR motion traces."""

    # Data exporter options
    task_prompt: str = DEFAULT_TASK_PROMPT
    """Language task prompt for the data exporter."""

    dataset_name: str | None = None
    """Omit to create a new timestamped dataset; set explicitly to resume one."""

    data_exporter_frequency: int = 50
    """Data collection frequency (Hz) for the data exporter."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist) in the dataset."""

    record_zed_stereo: bool = False
    """Record ZED left-eye RGB and float32 depth in the dataset."""

    text_to_speech: bool = True
    """Enable voice feedback via espeak (data exporter)."""

    # Camera viewer
    camera_viewer: bool = True
    """Start the camera viewer pane."""

    camera_host: str = "localhost"
    """Camera server host (shared by data exporter and viewer)."""

    camera_port: int = 5555
    """Stable camera endpoint: MuJoCo in simulation, ZED on hardware."""

    manage_camera_service: bool = True
    """Stop the ZED system service for simulation and start it for hardware."""

    camera_server_logs: bool = True
    """Show the systemd camera-server log in a read-only hardware pane."""

    sim_elastic_band: bool = False
    """Suspend the simulated robot with MuJoCo's virtual elastic band."""

    remote_ui: bool = False
    """Serve the camera and recorder UI through an SSH port forward.

    In simulation this also runs MuJoCo headlessly.
    """

    remote_ui_port: int = 8080
    """Loopback-only HTTP port used by the browser UI."""


SESSION_NAME = "sonic_data_collection"
DASHBOARD_WINDOW = f"{SESSION_NAME}:data_collection"
DASHBOARD_PANE = f"{DASHBOARD_WINDOW}.0"
CORE_PANE_COUNT = 4
SIM_PANE = 4


def _check_prerequisites(config: DataCollectionLaunchConfig):
    """Verify that required tools and venvs exist."""
    errors = []
    if config.sender_time_recording:
        if config.camera_host not in {"localhost", "127.0.0.1", "::1"}:
            errors.append("sender-time recording currently requires the camera publisher on this host")
        if not all(math.isfinite(v) and v > 0 for v in
                   (config.synchronization_delay, config.synchronization_wait_timeout)):
            errors.append("synchronization delay and wait timeout must be positive finite seconds")
        if not 1 <= config.hand_clock_port <= 65535 or config.hand_clock_port in (
            config.hand_intent_port, config.hand_state_port, config.hand_control_port,
        ):
            errors.append("hand clock port must be within 1..65535 and distinct from other hand ports")

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    if config.remote_ui_port == config.camera_port:
        errors.append("--remote-ui-port and --camera-port must be different")
    if not 1 <= config.remote_ui_port <= 65535:
        errors.append("--remote-ui-port must be between 1 and 65535")
    if not 1 <= config.camera_port <= 65535:
        errors.append("--camera-port must be between 1 and 65535")

    repo_root = Path(__file__).resolve().parent.parent.parent

    teleop_python = repo_root / ".venv_teleop" / "bin" / "python"
    if not (repo_root / ".venv_teleop" / "bin" / "activate").exists():
        errors.append(".venv_teleop not found. Run: bash install_scripts/install_pico.sh")
    elif config.body_control_mode == "ik-upper-slow-planner":
        ik_import = subprocess.run(
            [
                str(teleop_python),
                "-c",
                "import pink, qpsolvers; assert qpsolvers.available_solvers",
            ],
            capture_output=True,
            text=True,
        )
        if ik_import.returncode != 0:
            errors.append(
                "Upper-body IK dependencies are missing from .venv_teleop. "
                "Run: bash install_scripts/install_pico.sh"
            )

    if not (repo_root / ".venv_data_collection" / "bin" / "activate").exists():
        errors.append(".venv_data_collection not found. Run: bash install_scripts/install_data_collection.sh")

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not (deploy_dir / "deploy.sh").exists():
        errors.append(
            f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. Ensure the deploy directory is set up."
        )

    if config.sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(".venv_sim not found. Set up the simulation venv first (see install instructions).")

    if config.hand_backend == "omnihand" and not config.sim:
        if not (repo_root / ".venv_omnihand" / "bin" / "activate").exists():
            errors.append(".venv_omnihand not found. Run: bash install_scripts/install_omnihand.sh")
    if config.hand_backend == "omnihand" and config.sim:
        scene = repo_root / "gear_sonic/data/robot_model/model_data/g1_omnihand/scene_49dof.xml"
        mesh = (
            repo_root / "gear_sonic/data/robot_model/model_data/g1_omnihand/"
            "omnihand_description/assets/meshes/l_palm.STL"
        )
        if not scene.is_file() or not mesh.is_file() or mesh.stat().st_size < 1000:
            errors.append("Atlas OmniHand MuJoCo assets are unavailable or still LFS pointers. Run: git lfs pull")

    if config.hand_server_host is not None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.hand_server_host):
            errors.append("--hand-server-host must be an IPv4 address or hostname")
        if config.start_hand_server and not shutil.which("ssh"):
            errors.append("ssh is required to launch the remote hand server")
        if config.hand_backend != "dex1" or config.sim:
            errors.append("--hand-server-host requires physical --hand-backend dex1")

    if config.hand_backend == "dex1":
        from gear_sonic.end_effectors.backends.dex1 import USB_PORTS

        if config.sim:
            errors.append("DEX 1 currently supports physical USB grippers only; no DEX 1 MuJoCo scene is configured")
        if config.hand_server_host is None:
            worker = repo_root / "build/dex1/dex1_worker"
            if not os.access(worker, os.X_OK):
                errors.append("DEX 1 worker missing. Run: bash install_scripts/install_dex1.sh")
            for side, port in USB_PORTS.items():
                if not os.access(port, os.R_OK | os.W_OK):
                    errors.append(f"DEX 1 {side} serial adapter unavailable or inaccessible: {port}")
        if not math.isfinite(config.dex1_transition_duration) or not 1.35 <= config.dex1_transition_duration <= 30:
            errors.append("--dex1-transition-duration must be between 1.35 and 30 seconds")

    if not 0.0 <= config.omnihand_close_scale <= 1.0:
        errors.append("--omnihand-close-scale must be between zero and one")
    if not 0.0 <= config.omnihand_sim_close_scale <= 1.0:
        errors.append("--omnihand-sim-close-scale must be between zero and one")
    if config.omnihand_transition_duration <= 0.0:
        errors.append("--omnihand-transition-duration must be positive")
    hand_ports = {
        "--hand-intent-port": config.hand_intent_port,
        "--hand-state-port": config.hand_state_port,
        "--hand-control-port": config.hand_control_port,
        "--teleop-control-port": config.teleop_control_port,
    }
    for name, port in hand_ports.items():
        if not 1 <= port <= 65535:
            errors.append(f"{name} must be between 1 and 65535")
    reserved_ports = [config.camera_port, config.remote_ui_port, *hand_ports.values()]
    if len(reserved_ports) != len(set(reserved_ports)):
        errors.append("camera, remote UI, hand, and teleop ZMQ ports must all be different")

    if config.pico_input_source not in {"xrt", "isaac-teleop"}:
        errors.append("--pico-input-source must be one of: xrt, isaac-teleop")
    if config.idle_base_transition_duration <= 0.0:
        errors.append("--idle-base-transition-duration must be positive")
    for name in ("vr_max_speed", "vr_max_acceleration", "vr_max_jerk", "vr_max_angular_jerk_deg"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive and finite")

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    """Kill any existing tmux session with our name."""
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _camera_port_is_listening(port: int) -> bool:
    """Return whether a local process currently owns the camera endpoint."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _switch_camera_source(config: DataCollectionLaunchConfig) -> None:
    """Give the stable camera port to MuJoCo or the physical ZED service."""
    if not config.manage_camera_service:
        return

    service = "composed_camera_server.service"
    action = "stop" if config.sim else "start"
    source = "MuJoCo" if config.sim else "ZED"
    currently_active = subprocess.run(
        ["systemctl", "is-active", "--quiet", service],
        capture_output=True,
    ).returncode == 0
    expected_listening = not config.sim
    if (
        currently_active == (action == "start")
        and _camera_port_is_listening(config.camera_port) == expected_listening
    ):
        print(f"Camera source already set to {source} on port {config.camera_port}.")
        return
    if currently_active != (action == "start"):
        print(f"Switching camera source to {source} on port {config.camera_port}...")
        result = subprocess.run(["sudo", "-n", "systemctl", action, service])
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not {action} {service} without a password prompt. "
                f"Run 'sudo systemctl {action} {service}' once, then launch again."
            )
    else:
        print(f"Waiting for camera port {config.camera_port} to {'be ready' if expected_listening else 'be free'}...")

    deadline = time.monotonic() + (10.0 if config.sim else 60.0)
    while time.monotonic() < deadline:
        if _camera_port_is_listening(config.camera_port) == expected_listening:
            return
        time.sleep(0.25)

    state = "become free" if config.sim else "start listening"
    raise RuntimeError(
        f"Camera port {config.camera_port} did not {state}. "
        f"Check camera availability and service logs: journalctl -u {service} -n 50 --no-pager"
    )


def _launch_hands(config: DataCollectionLaunchConfig) -> bool:
    return config.hand_backend in {"omnihand", "dex1"} and (
        config.hand_server_host is None or config.start_hand_server
    )


def _remote_hand_ssh_args(config: DataCollectionLaunchConfig) -> list[str]:
    args = [
        "ssh", "-tt", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=3",
    ]
    key = Path.home() / ".ssh" / f"id_ed25519_sonic_{config.hand_server_host}"
    if key.is_file():
        args += ["-i", str(key), "-o", "IdentitiesOnly=yes"]
    return args


def _check_remote_hand_connection(config: DataCollectionLaunchConfig) -> None:
    """Check authentication before replacing a dashboard or starting services."""
    if config.hand_server_host is None or not config.start_hand_server:
        return
    args = _remote_hand_ssh_args(config)
    args[1] = "-T"
    try:
        result = subprocess.run(
            [*args, "--", config.hand_server_host, "true"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit("Orin connection timed out; check the robot network.") from None
    if result.returncode != 0:
        setup = shlex.join(["bash", "tools/setup_orin_ssh.sh", config.hand_server_host])
        raise SystemExit(
            f"Cannot connect to hand host {config.hand_server_host} without a password.\n"
            f"One-time pairing: {setup}\n"
            f"SSH: {result.stderr.strip()}"
        )


def _remote_hand_command(config: DataCollectionLaunchConfig) -> str:
    """Keep the remote process attached to the hand pane, including shutdown."""
    command = (
        f"cd {shlex.quote(config.hand_server_repo)} && "
        # SSH_CONNECTION supplies Thor's address as seen by the hand host.
        'exec .venv_hands/bin/python -m gear_sonic.end_effectors.server '
        '--teleop-host "${SSH_CONNECTION%% *}" '
        f"--intent-port {config.hand_intent_port} --state-port {config.hand_state_port} "
        f"--control-port {config.hand_control_port} "
        f"--dex1-transition-duration {config.dex1_transition_duration} --enable-command"
    )
    if config.sender_time_recording:
        command += f" --clock-port {config.hand_clock_port}"
    return shlex.join([
        *_remote_hand_ssh_args(config), "--", str(config.hand_server_host), command,
    ])


def _hand_pane(config: DataCollectionLaunchConfig) -> int:
    """Return the OmniHand pane index for the selected launch configuration."""
    return CORE_PANE_COUNT + int(config.sim)


def _camera_server_log_pane(config: DataCollectionLaunchConfig) -> int:
    """Return the hardware camera log pane after optional sim/hand panes."""
    return CORE_PANE_COUNT + int(config.sim) + int(_launch_hands(config))


def _create_tmux_session(config: DataCollectionLaunchConfig):
    """Create one tiled dashboard containing every requested component."""
    # Create detached session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )

    # Enable mouse support (click panes, scroll, resize)
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )

    # Bind Ctrl+\ to kill the entire session (no prefix needed)
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )

    # Rename default window
    subprocess.run(["tmux", "rename-window", "-t", SESSION_NAME, "data_collection"], check=True)
    subprocess.run(
        ["tmux", "set-window-option", "-t", DASHBOARD_WINDOW, "pane-base-index", "0"],
        check=True,
    )

    pane_count = (
        CORE_PANE_COUNT
        + int(config.sim)
        + int(_launch_hands(config))
        + int(not config.sim and config.camera_server_logs)
    )
    for _ in range(1, pane_count):
        subprocess.run(
            ["tmux", "split-window", "-d", "-t", DASHBOARD_WINDOW],
            check=True,
        )
        # Without rebalancing here, every split targets the still-active pane 0
        # and eventually fails with "no space for new pane" on small terminals.
        subprocess.run(["tmux", "select-layout", "-t", DASHBOARD_WINDOW, "tiled"], check=True)

    # Let all pane shells finish initialization (.bashrc, conda, etc.)
    time.sleep(5)


def _select_dashboard():
    """Make the data-collection dashboard the visible tmux view."""
    subprocess.run(
        ["tmux", "select-window", "-t", DASHBOARD_WINDOW],
        check=True,
    )
    subprocess.run(
        ["tmux", "select-pane", "-t", DASHBOARD_PANE],
        check=True,
    )


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0):
    """Send a command string to a tmux pane."""
    target = f"{DASHBOARD_WINDOW}.{pane_index}"
    # An existing tmux server retains its old environment. Pass the source
    # checkout explicitly even when the launcher reused an installed venv.
    cmd = f"export PYTHONPATH={shlex.quote(_SOURCE_ROOT)} && {cmd}"

    subprocess.run(
        ["tmux", "send-keys", "-t", target, cmd, "C-m"],
    )
    time.sleep(wait)


def _check_pane_alive(pane_index: int) -> bool:
    """Check if a tmux pane's process is still running."""
    target = f"{DASHBOARD_WINDOW}.{pane_index}"
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def _hand_worker_command(config: DataCollectionLaunchConfig) -> tuple[str, str]:
    """Return the environment and complete external-hand worker command."""
    if config.hand_backend == "dex1":
        return ".venv_data_collection", shlex.join([
            "python", "-m", "gear_sonic.end_effectors.controller", "run",
            "--backend", "dex1", "--sides", "both", "--enable-command",
            "--intent-endpoint", f"tcp://localhost:{config.hand_intent_port}",
            "--state-endpoint", f"tcp://*:{config.hand_state_port}",
            "--dex1-transition-duration", str(config.dex1_transition_duration),
        ])
    return (".venv_sim" if config.sim else ".venv_omnihand"), shlex.join([
        "python", "-m", "gear_sonic.end_effectors.controller", "run",
        "--backend", "sim" if config.sim else "omnihand", "--sides", "both",
        "--intent-endpoint", f"tcp://localhost:{config.hand_intent_port}",
        "--state-endpoint", f"tcp://*:{config.hand_state_port}",
        "--left-interface", config.omnihand_left_interface,
        "--right-interface", config.omnihand_right_interface,
        "--close-scale", str(config.omnihand_sim_close_scale if config.sim else config.omnihand_close_scale),
        "--transition-duration", str(config.omnihand_transition_duration),
        *([] if config.sim else ["--enable-command"]),
    ])


def main(config: DataCollectionLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent

    _check_prerequisites(config)
    if config.check_only:
        print("Launch prerequisites passed; no services started.")
        if config.hand_server_host is not None:
            if config.start_hand_server:
                print(_remote_hand_command(config))
            else:
                print(f"Using existing hand server at {config.hand_server_host}:{config.hand_state_port}")
        elif _launch_hands(config):
            print(_hand_worker_command(config)[1])
        return

    _check_remote_hand_connection(config)

    print("=" * 60)
    print("  SONIC Data Collection Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  Task prompt:     {config.task_prompt}")
    print(f"  Dataset name:    {config.dataset_name or '(auto)'}")
    print(f"  Deploy input:    {config.deploy_input_type}")
    print(f"  Teleop input:    {config.pico_input_source}")
    print(f"  Body control:    {config.body_control_mode}")
    print(f"  VR limiter:      {'DISABLED' if config.disable_vr_motion_limiter else 'enabled'}")
    if not config.disable_vr_motion_limiter:
        print(f"  VR speed limit:  {config.vr_max_speed:g} m/s (hard ceiling {1.5 * config.vr_max_speed:g} m/s)")
        print(f"  VR acceleration: {config.vr_max_acceleration:g} m/s^2 (hard ceiling {1.5 * config.vr_max_acceleration:g} m/s^2)")
        print(f"  VR jerk limits:  {config.vr_max_jerk:g} m/s^3, {config.vr_max_angular_jerk_deg:g} deg/s^3")
    if config.vr_motion_log_dir:
        print(f"  VR motion logs:  {config.vr_motion_log_dir}")
    print(f"  Hand backend:    {config.hand_backend}")
    print(f"  Hand server:     {config.hand_server_host or 'local (managed by launcher)'}")
    if config.deploy_checkpoint:
        print(f"  Checkpoint:      {config.deploy_checkpoint}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  DC frequency:    {config.data_exporter_frequency} Hz")
    viewer_mode = "Browser" if config.remote_ui else ("Native" if config.camera_viewer else "No")
    print(f"  Camera viewer:   {viewer_mode}")
    print(f"  Wrist cameras:   {'Yes' if config.record_wrist_cameras else 'No'}")
    print(f"  Text-to-speech:  {'Yes' if config.text_to_speech else 'No'}")
    print(f"  PC IP (for PICO): {_get_local_ip()}")
    print(f"  Teleop vis:      vr3pt={config.pico_vis_vr3pt} smpl={config.pico_vis_smpl}")
    print("=" * 60)

    _kill_existing_session()
    _switch_camera_source(config)
    _create_tmux_session(config)
    print(f"Created tmux session: {SESSION_NAME}")

    # --- Pane 4 (sim only): MuJoCo Simulator ---
    if config.sim:
        python_prefix = "MUJOCO_GL=egl python" if config.remote_ui else "python"
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"{python_prefix} gear_sonic/scripts/run_sim_loop.py "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {config.camera_port}"
        )
        if config.remote_ui:
            sim_cmd += " --no-enable-onscreen --stream-camera third_person"
        if not config.sim_elastic_band:
            sim_cmd += " --no-enable-elastic-band"
        if config.hand_backend == "omnihand":
            sim_cmd += " --external-hand-control"
        print(f"Starting MuJoCo simulator (pane {SIM_PANE})...")
        _send_to_pane(SIM_PANE, sim_cmd, wait=3.0)

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {repo_root / 'gear_sonic_deploy'} && "
        f"./deploy.sh "
        f"--yes "
        f"--input-type {config.deploy_input_type} "
        f"--zmq-host {config.deploy_zmq_host} "
    )
    hand_control = {
        "dex3": "legacy-dex3",
        "omnihand": "external",
        "dex1": "external",
        "none": "none",
    }[config.hand_backend]
    deploy_cmd += f"--hand-control {hand_control} "
    if config.deploy_checkpoint:
        deploy_cmd += f"--cp {config.deploy_checkpoint} "
    if config.deploy_obs_config:
        deploy_cmd += f"--obs-config {config.deploy_obs_config} "
    if config.deploy_planner:
        deploy_cmd += f"--planner {config.deploy_planner} "
    if config.deploy_motion_data:
        deploy_cmd += f"--motion-data {config.deploy_motion_data} "
    if config.deploy_output_type:
        deploy_cmd += f"--output-type {config.deploy_output_type} "
    deploy_cmd += deploy_mode

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(0, deploy_cmd, wait=3.0)

    if not _check_pane_alive(0):
        print("WARNING: C++ deploy pane may have failed to start.")

    # --- Pane 1 (top-right): Teleop Streamer ---
    pico_teleop_mode = {
        "vr3pt-slow-planner": "vr3pt",
        "ik-upper-slow-planner": "ik-upper",
        "full-smpl": "pose",
    }[config.body_control_mode]
    required_stream_mode = {
        "vr3pt-slow-planner": 5,
        "ik-upper-slow-planner": 6,
        "full-smpl": 1,
    }[config.body_control_mode]
    pico_process_cmd = (
        "python gear_sonic/scripts/pico_manager_thread_server.py "
        f"--input-source {config.pico_input_source} "
        f"--hand-intent-port {config.hand_intent_port} "
        f"--teleop-control-port {config.teleop_control_port} "
        f"--teleop-mode {pico_teleop_mode} "
        f"--idle-base-transition-duration {config.idle_base_transition_duration} "
        f"--vr-max-speed {config.vr_max_speed} "
        f"--vr-max-acceleration {config.vr_max_acceleration} "
        f"--vr-max-jerk {config.vr_max_jerk} "
        f"--vr-max-angular-jerk-deg {config.vr_max_angular_jerk_deg} "
        "--initial-locomotion-mode slow_walk"
    )
    if config.pico_manager:
        pico_process_cmd += " --manager"
    if config.disable_vr_motion_limiter:
        pico_process_cmd += " --disable-vr-motion-limiter"
    else:
        pico_process_cmd += " --enable-vr-motion-limiter"
    if config.vr_motion_log_dir:
        pico_process_cmd += f" --vr-motion-log-dir {shlex.quote(config.vr_motion_log_dir)}"
    if config.pico_vis_vr3pt:
        pico_process_cmd += " --vis_vr3pt"
    if config.pico_vis_smpl:
        pico_process_cmd += " --vis_smpl"
    if config.pico_waist_tracking:
        pico_process_cmd += " --waist_tracking"

    # The vendor XRT extension is native code and can abort the Python worker on
    # a broken connection. Keep pane 1 alive and restart only unexpected exits;
    # a clean exit or Ctrl-C still stops the supervisor.
    pico_cmd = (
        f"cd {repo_root} && "
        "source .venv_teleop/bin/activate && "
        "while true; do "
        f"{pico_process_cmd}; "
        "status=$?; "
        'if [ "$status" -eq 0 ] || [ "$status" -eq 130 ]; then break; fi; '
        'echo "[PICO supervisor] Worker exited with status $status; restarting in 2s"; '
        "sleep 2; "
        "done"
    )

    print("Starting teleop streamer (pane 1)...")
    _send_to_pane(1, pico_cmd, wait=2.0)

    # --- Dedicated hand controller pane, local or SSH. ---
    if _launch_hands(config):
        hand_pane = _hand_pane(config)
        if config.hand_server_host is not None:
            hand_cmd = _remote_hand_command(config)
        else:
            hand_venv, hand_worker_cmd = _hand_worker_command(config)
            # The supervisor owns the restart command outside the native SDK
            # process, so the UI can recover even if a vendor call is wedged.
            hand_cmd = (
                f"cd {repo_root} && source {hand_venv}/bin/activate && "
                "python -m gear_sonic.end_effectors.supervisor "
                f"--control-endpoint tcp://localhost:{config.hand_control_port} -- "
                f"{hand_worker_cmd}"
            )
        print(f"Starting {config.hand_backend} controller (pane {hand_pane})...")
        _send_to_pane(hand_pane, hand_cmd)

    # --- Read-only system camera service log (hardware only) ---
    if not config.sim and config.camera_server_logs:
        camera_log_pane = _camera_server_log_pane(config)
        camera_log_cmd = (
            "journalctl -u composed_camera_server.service "
            "--follow --lines 100 --no-pager"
        )
        print(f"Following camera server logs (pane {camera_log_pane})...")
        _send_to_pane(camera_log_pane, camera_log_cmd)

    # --- Pane 3 (middle-right): Native or browser camera viewer ---
    if config.remote_ui:
        viewer_cmd = (
            f"cd {repo_root} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/run_camera_web_viewer.py "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port} "
            f"--http-port {config.remote_ui_port} "
            f"--hand-state-port {config.hand_state_port} "
            f"--hand-state-host {shlex.quote(config.hand_server_host or 'localhost')} "
            f"--hand-control-port {config.hand_control_port} "
            f"--teleop-control-port {config.teleop_control_port}"
        )
        if config.hand_backend in {"omnihand", "dex1"}:
            viewer_cmd += " --enable-hand-controls"
        print(f"Starting browser viewer on loopback port {config.remote_ui_port} (pane 3)...")
        _send_to_pane(3, viewer_cmd, wait=2.0)
    elif config.camera_viewer:
        viewer_cmd = (
            f"cd {repo_root} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/run_camera_viewer.py "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port}"
        )
        print("Starting camera viewer (pane 3)...")
        _send_to_pane(3, viewer_cmd, wait=2.0)

    # --- Pane 2 (middle-left): Data Exporter ---
    exporter_cmd = (
        f"cd {repo_root} && "
        f"source .venv_data_collection/bin/activate && "
        f"python gear_sonic/scripts/run_data_exporter.py "
        f"--task-prompt '{config.task_prompt}' "
        f"--data-collection-frequency {config.data_exporter_frequency} "
        f"--required-stream-mode {required_stream_mode} "
        f"--camera-host {config.camera_host} "
        f"--camera-port {config.camera_port} "
        f"--hand-state-port {config.hand_state_port} "
        f"--hand-state-host {shlex.quote(config.hand_server_host or 'localhost')}"
    )
    if config.sender_time_recording:
        exporter_cmd += (
            f" --sender-time-recording --synchronization-delay {config.synchronization_delay}"
            f" --synchronization-wait-timeout {config.synchronization_wait_timeout}"
            f" --hand-clock-port {config.hand_clock_port}"
        )
    if config.dataset_name:
        exporter_cmd += f" --dataset-name '{config.dataset_name}'"
    if config.record_wrist_cameras:
        exporter_cmd += " --record-wrist-cameras"
    if config.record_zed_stereo:
        exporter_cmd += " --record-zed-stereo"
    if config.remote_ui:
        exporter_cmd += " --require-hub-upload"
    if not config.text_to_speech:
        exporter_cmd += " --no-text-to-speech"

    print("Starting data exporter (pane 2)...")
    _send_to_pane(2, exporter_cmd, wait=1.0)

    # Focus deploy for live controller diagnostics.
    _select_dashboard()

    print()
    print("=" * 60)
    print("  All components launched!")
    print()
    print(f"  tmux session: {SESSION_NAME}")
    print()
    print("  Window 'data_collection' (all panels visible):")
    print("    Pane 0: C++ Deploy  <-- you are here")
    print("    Pane 1: Teleop Streamer")
    print("    Pane 2: Data Exporter")
    if config.remote_ui:
        print(f"    Pane 3: Browser UI on 127.0.0.1:{config.remote_ui_port}")
        print()
        print("  On your computer, open another terminal and run:")
        print(f"    ssh -N -L {config.remote_ui_port}:127.0.0.1:{config.remote_ui_port} unitree@<robot-host>")
        print(f"  Then open: http://127.0.0.1:{config.remote_ui_port}")
    elif config.camera_viewer:
        print("    Pane 3: Camera Viewer")
    if config.sim:
        print(f"    Pane {SIM_PANE}: MuJoCo Simulator")
    if _launch_hands(config):
        print(f"    Pane {_hand_pane(config)}: {config.hand_backend} controller")
    elif config.hand_server_host is not None:
        print(f"    Hands: existing service on {config.hand_server_host}")
    if not config.sim and config.camera_server_logs:
        print(
            f"    Pane {_camera_server_log_pane(config)}: "
            "Camera server logs (read-only)"
        )
    print()
    print("  deploy.sh was started with non-interactive confirmation.")
    print()
    print("  Controls:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    # Attach to the session
    try:
        if os.environ.get("TMUX"):
            subprocess.run(
                ["tmux", "switch-client", "-t", DASHBOARD_WINDOW],
                check=True,
            )
            return
        subprocess.run(
            ["tmux", "attach-session", "-t", DASHBOARD_WINDOW],
            check=True,
        )
    except KeyboardInterrupt:
        pass

    # After detach/exit, offer cleanup
    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(sig, frame):
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    config = tyro.cli(DataCollectionLaunchConfig)
    main(config)
