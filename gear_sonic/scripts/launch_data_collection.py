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
options are enabled.

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

import tyro  # noqa: E402


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

    hand_server_host: str | None = None
    """Run physical DEX1 hands on a remote Orin host over SSH."""

    start_hand_server: bool = True
    """Start the remote hand service from the dashboard when configured."""

    hand_server_repo: str = "/home/unitree/gr00t-wbc-g1-omnihand"

    dex1_transition_duration: float = 1.5
    """Seconds per DEX1 stroke; supported range is 1.35–30."""

    check_only: bool = False
    """Validate prerequisites and print commands without starting tmux."""

    hand_intent_port: int = 5569
    hand_state_port: int = 5570
    hand_control_port: int = 5572
    teleop_control_port: int = 5573

    omnihand_close_scale: float = 0.35
    """Fraction of the provisional O10 closed pose admitted for hardware motion."""

    omnihand_sim_close_scale: float = 1.0
    """Fraction of the O10 closed pose used by the Atlas MuJoCo model."""

    omnihand_transition_duration: float = 1.0
    """Seconds for either simulated or physical OmniHand to open or close."""

    omnihand_left_interface: str = "can11"
    """Serial-bound SocketCAN interface for the physical left O10."""

    omnihand_right_interface: str = "can10"
    """Serial-bound SocketCAN interface for the physical right O10."""

    # Teleop streamer options
    pico_manager: bool = True
    """Run pico_manager_thread_server with --manager flag."""

    pico_input_source: str = "xrt"
    """Teleop input source for pico_manager_thread_server.py (xrt or isaac-teleop)."""

    teleop_mode: Literal["pose", "vr3pt"] = "pose"
    """VR3PT uses staged arm alignment; pose keeps the full-body controls."""

    pico_vis_vr3pt: bool = False
    """Enable VR 3-point visualization on the teleop streamer."""

    pico_vis_smpl: bool = False
    """Enable SMPL visualization on the teleop streamer."""

    pico_waist_tracking: bool = False
    """Enable waist tracking on the teleop streamer."""

    # Data exporter options
    task_prompt: str = "demo"
    """Language task prompt for the data exporter."""

    require_hub_upload: bool = False
    """Require browser dataset selection before recording (use with --remote-ui)."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate from timestamp."""

    data_exporter_frequency: int = 50
    """Data collection frequency (Hz) for the data exporter."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist) in the dataset."""

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

    sim_elastic_band: bool = False
    """Suspend the simulated robot with MuJoCo's virtual elastic band."""

    remote_ui: bool = False
    """Run MuJoCo headlessly and serve a browser UI through an SSH port forward."""

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

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    if config.remote_ui and not config.sim:
        errors.append("--remote-ui is only supported with --sim")
    if config.remote_ui_port == config.camera_port:
        errors.append("--remote-ui-port and --camera-port must be different")
    if not 1 <= config.remote_ui_port <= 65535:
        errors.append("--remote-ui-port must be between 1 and 65535")
    if not 1 <= config.camera_port <= 65535:
        errors.append("--camera-port must be between 1 and 65535")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_teleop" / "bin" / "activate").exists():
        errors.append(".venv_teleop not found. Run: bash install_scripts/install_pico.sh")

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
        if config.hand_backend != "dex1" or config.sim:
            errors.append("--hand-server-host requires physical --hand-backend dex1")
        if config.start_hand_server and not shutil.which("ssh"):
            errors.append("ssh is required to start the remote hand server")

    if config.hand_backend == "dex1":
        if config.sim:
            errors.append("DEX1 currently supports physical USB grippers only")
        if config.hand_server_host is None:
            worker = repo_root / "build/dex1/dex1_worker"
            if not os.access(worker, os.X_OK):
                errors.append("DEX1 worker missing. Run: bash install_scripts/install_dex1.sh")
        if not math.isfinite(config.dex1_transition_duration) or not 1.35 <= config.dex1_transition_duration <= 30:
            errors.append("--dex1-transition-duration must be between 1.35 and 30 seconds")

    hand_ports = [
        config.hand_intent_port,
        config.hand_state_port,
        config.hand_control_port,
        config.teleop_control_port,
    ]
    if any(not 1 <= port <= 65535 for port in hand_ports):
        errors.append("hand and teleop control ports must be between 1 and 65535")
    if len(set([config.camera_port, config.remote_ui_port, *hand_ports])) != 2 + len(hand_ports):
        errors.append("camera, remote UI, hand, and teleop ZMQ ports must all be different")

    if not 0.0 <= config.omnihand_close_scale <= 1.0:
        errors.append("--omnihand-close-scale must be between zero and one")
    if not 0.0 <= config.omnihand_sim_close_scale <= 1.0:
        errors.append("--omnihand-sim-close-scale must be between zero and one")
    if config.omnihand_transition_duration <= 0.0:
        errors.append("--omnihand-transition-duration must be positive")

    if config.pico_input_source not in {"xrt", "isaac-teleop"}:
        errors.append("--pico-input-source must be one of: xrt, isaac-teleop")
    if config.teleop_mode != "pose" and not config.pico_manager:
        errors.append("--teleop-mode vr3pt requires the PICO manager")

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
    print(f"Switching camera source to {source} on port {config.camera_port}...")
    result = subprocess.run(["sudo", "systemctl", action, service])
    if result.returncode != 0:
        raise RuntimeError(f"Could not {action} {service}")

    deadline = time.monotonic() + (10.0 if config.sim else 60.0)
    expected_listening = not config.sim
    while time.monotonic() < deadline:
        if _camera_port_is_listening(config.camera_port) == expected_listening:
            return
        time.sleep(0.25)

    state = "become free" if config.sim else "start listening"
    raise RuntimeError(f"Camera port {config.camera_port} did not {state} after service {action}")


def _hand_pane(config: DataCollectionLaunchConfig) -> int:
    """Return the OmniHand pane index for the selected launch configuration."""
    return CORE_PANE_COUNT + int(config.sim)


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

    pane_count = CORE_PANE_COUNT + int(config.sim) + int(_launch_hands(config))
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


def _launch_hands(config: DataCollectionLaunchConfig) -> bool:
    return config.hand_backend in {"omnihand", "dex1"} and (
        config.hand_server_host is None or config.start_hand_server
    )


def _hand_worker_command(config: DataCollectionLaunchConfig) -> tuple[str, str]:
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


def _remote_hand_command(config: DataCollectionLaunchConfig) -> str:
    command = (
        f"cd {shlex.quote(config.hand_server_repo)} && "
        "exec .venv_hands/bin/python -m gear_sonic.end_effectors.server "
        '--teleop-host "${SSH_CONNECTION%% *}" '
        f"--intent-port {config.hand_intent_port} --state-port {config.hand_state_port} "
        f"--control-port {config.hand_control_port} "
        f"--dex1-transition-duration {config.dex1_transition_duration} --enable-command"
    )
    return shlex.join(
        [
            "ssh", "-tt", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=2",
            "-o", "ServerAliveCountMax=3", "--", config.hand_server_host, command,
        ]
    )


def main(config: DataCollectionLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent

    _check_prerequisites(config)
    if config.check_only:
        print("Launch prerequisites passed; no services started.")
        if config.hand_server_host is not None:
            if config.start_hand_server:
                print(_remote_hand_command(config))
            else:
                print(
                    f"Using existing hand server at "
                    f"{config.hand_server_host}:{config.hand_state_port}"
                )
        elif _launch_hands(config):
            print(_hand_worker_command(config)[1])
        return

    print("=" * 60)
    print("  SONIC Data Collection Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  Task prompt:     {config.task_prompt}")
    print(f"  Dataset name:    {config.dataset_name or '(auto)'}")
    print(f"  Deploy input:    {config.deploy_input_type}")
    print(f"  Teleop input:    {config.pico_input_source}")
    print(f"  Hand backend:    {config.hand_backend}")
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

    if config.hand_backend in {"omnihand", "dex1"} and not config.sim:
        acknowledgement = input("Physical OmniHand control may move both hands. Type OMNIHAND to continue: ")
        if acknowledgement != "OMNIHAND":
            print("OmniHand launch cancelled; no hardware commands were enabled.")
            return

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
        if config.hand_backend in {"omnihand", "dex1"}:
            sim_cmd += " --external-hand-control"
        print(f"Starting MuJoCo simulator (pane {SIM_PANE})...")
        _send_to_pane(SIM_PANE, sim_cmd, wait=3.0)

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {repo_root / 'gear_sonic_deploy'} && "
        f"./deploy.sh "
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
    pico_process_cmd = (
        "python gear_sonic/scripts/pico_manager_thread_server.py "
        f"--input-source {config.pico_input_source}"
    )
    if config.pico_manager:
        pico_process_cmd += f" --manager --teleop-mode {config.teleop_mode}"
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

    # --- Dedicated hand controller pane (OmniHand only) ---
    if _launch_hands(config):
        hand_pane = _hand_pane(config)
        if config.hand_server_host is not None:
            hand_cmd = _remote_hand_command(config)
        else:
            hand_venv, hand_worker_cmd = _hand_worker_command(config)
            hand_cmd = (
                f"cd {repo_root} && source {hand_venv}/bin/activate && "
                "python -m gear_sonic.end_effectors.supervisor "
                f"--control-endpoint tcp://localhost:{config.hand_control_port} -- {hand_worker_cmd}"
            )
        print(f"Starting {config.hand_backend} controller (pane {hand_pane})...")
        _send_to_pane(hand_pane, hand_cmd)

    # --- Pane 3 (middle-right): Native or browser camera viewer ---
    if config.remote_ui:
        viewer_cmd = (
            f"cd {repo_root} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/run_camera_web_viewer.py "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port} "
            f"--http-port {config.remote_ui_port}"
        )
        if config.hand_backend in {"omnihand", "dex1"}:
            viewer_cmd += " --hand-controls"
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
        f"--camera-host {config.camera_host} "
        f"--camera-port {config.camera_port}"
    )
    if config.dataset_name:
        exporter_cmd += f" --dataset-name '{config.dataset_name}'"
    if config.record_wrist_cameras:
        exporter_cmd += " --record-wrist-cameras"
    if config.require_hub_upload:
        exporter_cmd += " --require-hub-upload"
    if not config.text_to_speech:
        exporter_cmd += " --no-text-to-speech"

    print("Starting data exporter (pane 2)...")
    _send_to_pane(2, exporter_cmd, wait=1.0)

    # Focus deploy, which may be waiting for confirmation or a sudo password.
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
    print()
    print("  ** deploy.sh (pane 0) is waiting for confirmation —")
    print("     click on pane 0 and press Enter to proceed **")
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
