from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.end_effectors import server
from gear_sonic.end_effectors.backends.sim import SimHandBackend
from gear_sonic.end_effectors.controller import SafeHandController
from gear_sonic.end_effectors.profiles import DEX1
from gear_sonic.scripts import launch_data_collection as launcher
from gear_sonic.scripts.run_data_exporter import _capture_reproducibility_metadata


def mock_launcher(monkeypatch):
    commands = {}
    for name in ("_check_prerequisites", "_switch_camera_source", "_create_tmux_session"):
        monkeypatch.setattr(launcher, name, lambda config: None)
    for name in ("_kill_existing_session", "_select_dashboard"):
        monkeypatch.setattr(launcher, name, lambda: None)
    monkeypatch.setattr(launcher, "_send_to_pane", lambda pane, cmd, **kw: commands.update({pane: cmd}))
    monkeypatch.setattr(launcher, "_check_pane_alive", lambda pane: True)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kw: SimpleNamespace(returncode=0))
    return commands


@pytest.mark.parametrize("host,start", [(None, True), ("192.168.123.164", True), ("orin", False)])
def test_full_launcher_routes_hands_without_changing_body_or_camera(monkeypatch, host, start):
    commands = mock_launcher(monkeypatch)
    config = launcher.DataCollectionLaunchConfig(
        hand_backend="dex1", remote_ui=True, hand_server_host=host, start_hand_server=start
    )
    launcher.main(config)
    assert "--hand-control external" in commands[0]
    assert "--hand-intent-port 5569" in commands[1]
    for pane in (2, 3):
        argv = shlex.split(commands[pane])
        assert argv[argv.index("--hand-state-host") + 1] == (host or "localhost")
        assert argv[argv.index("--camera-host") + 1] == "localhost"
    assert "--enable-hand-controls" in commands[3]
    if host and not start:
        assert len(commands) == 5
        assert commands[4].startswith("journalctl")
        assert not any("end_effectors.supervisor" in cmd for cmd in commands.values())
    elif host:
        assert shlex.split(commands[4])[:2] == ["ssh", "-tt"]
        assert commands[5].startswith("journalctl")
    else:
        assert "end_effectors.supervisor" in commands[4]
        assert "tcp://localhost:5569" in commands[4]
        assert commands[5].startswith("journalctl")


@pytest.mark.parametrize("host,passes", [(None, False), ("192.168.123.164", True)])
def test_usb_and_native_worker_prerequisites_apply_only_to_local_hands(monkeypatch, host, passes):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(launcher.os, "access", lambda *args: False)
    config = launcher.DataCollectionLaunchConfig(hand_backend="dex1", hand_server_host=host)
    if passes:
        launcher._check_prerequisites(config)
    else:
        with pytest.raises(SystemExit):
            launcher._check_prerequisites(config)


def test_remote_check_only_never_prints_or_starts_a_local_worker(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "_check_prerequisites", lambda config: None)
    monkeypatch.setattr(launcher, "_hand_worker_command", lambda config: pytest.fail("local worker requested"))
    launcher.main(
        launcher.DataCollectionLaunchConfig(
            hand_backend="dex1",
            hand_server_host="orin",
            start_hand_server=False,
            check_only=True,
        )
    )
    assert "Using existing hand server at orin:5570" in capsys.readouterr().out


def test_remote_command_quotes_checkout_and_forwards_configuration(monkeypatch, capsys):
    config = launcher.DataCollectionLaunchConfig(
        hand_backend="dex1", hand_server_host="orin", hand_server_repo="/tmp/a b'$(false)",
        hand_intent_port=6001, hand_state_port=6002, hand_control_port=6003,
        dex1_transition_duration=2.0, check_only=True,
    )
    monkeypatch.setattr(launcher, "_check_prerequisites", lambda config: None)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: pytest.fail("check-only started SSH"))
    launcher.main(config)
    command = capsys.readouterr().out.splitlines()[-1]
    ssh = shlex.split(command)
    assert ssh[-2] == "orin"
    remote = shlex.split(ssh[-1])
    assert remote[1] == config.hand_server_repo
    assert remote[remote.index("--teleop-host") + 1] == "${SSH_CONNECTION%% *}"
    assert remote[remote.index("--intent-port") + 1] == "6001"
    assert remote[remote.index("--state-port") + 1] == "6002"
    assert remote[remote.index("--control-port") + 1] == "6003"
    assert remote[remote.index("--dex1-transition-duration") + 1] == "2.0"
    assert "BatchMode=yes" in ssh


def test_remote_launch_uses_paired_key_without_changing_other_ssh_config(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = launcher.DataCollectionLaunchConfig(hand_backend="dex1", hand_server_host="orin")
    assert "-i" not in shlex.split(launcher._remote_hand_command(config))
    key = tmp_path / ".ssh" / "id_ed25519_sonic_orin"
    key.parent.mkdir()
    key.write_text("test placeholder")
    args = shlex.split(launcher._remote_hand_command(config))
    assert args[args.index("-i") + 1] == str(key)
    assert "IdentitiesOnly=yes" in args


def test_failed_authentication_preserves_running_stack(monkeypatch):
    monkeypatch.setattr(launcher, "_check_prerequisites", lambda config: None)
    for name in ("_kill_existing_session", "_switch_camera_source", "_create_tmux_session"):
        monkeypatch.setattr(launcher, name, lambda *args: pytest.fail("existing stack changed"))
    commands = []

    def reject(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=255, stderr="Permission denied (publickey,password).")

    monkeypatch.setattr(launcher.subprocess, "run", reject)
    with pytest.raises(SystemExit, match="One-time pairing: bash tools/setup_orin_ssh.sh orin"):
        launcher.main(launcher.DataCollectionLaunchConfig(hand_backend="dex1", hand_server_host="orin"))
    assert len(commands) == 1
    assert commands[0][-1] == "true"
    assert "BatchMode=yes" in commands[0]
    assert "-T" in commands[0]


def test_remote_authentication_timeout_is_reported_without_launching(monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(launcher.subprocess, "run", timeout)
    with pytest.raises(SystemExit, match="connection timed out"):
        launcher._check_remote_hand_connection(
            launcher.DataCollectionLaunchConfig(hand_backend="dex1", hand_server_host="orin")
        )


@pytest.mark.parametrize("host,start", [(None, True), ("orin", False)])
def test_local_and_existing_hand_servers_skip_ssh_authentication(monkeypatch, host, start):
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected SSH"))
    launcher._check_remote_hand_connection(
        launcher.DataCollectionLaunchConfig(hand_backend="dex1", hand_server_host=host, start_hand_server=start)
    )


@pytest.mark.parametrize(
    "options",
    [
        {"hand_server_host": "tcp://orin:5570"},
        {"hand_server_host": ""},
        {"hand_server_host": "orin", "hand_backend": "omnihand"},
        {"hand_server_host": "orin", "sim": True},
    ],
)
def test_launcher_rejects_invalid_remote_configuration(monkeypatch, options):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(launcher.os, "access", lambda *args: True)
    config = launcher.DataCollectionLaunchConfig(**{"hand_backend": "dex1", **options})
    with pytest.raises(SystemExit):
        launcher._check_prerequisites(config)


def test_standalone_service_routes_intent_state_and_reconnect(monkeypatch):
    args = server.build_parser().parse_args(["--teleop-host", "192.168.123.166", "--enable-command"])
    command = server.worker_command(args)
    assert command[command.index("--intent-endpoint") + 1] == "tcp://192.168.123.166:5569"
    assert command[command.index("--state-endpoint") + 1] == "tcp://0.0.0.0:5570"
    assert command[command.index("--frequency") + 1] == "50"
    assert "--enable-command" in command
    calls = []
    monkeypatch.setattr(server.os, "access", lambda *args: True)
    monkeypatch.setattr(
        server,
        "HandWorkerSupervisor",
        lambda cmd, **kw: SimpleNamespace(
            run=lambda: calls.append((cmd, kw)) or 0,
        ),
    )
    assert server.main(["--teleop-host", "192.168.123.166", "--enable-command"]) == 0
    assert calls[0][1]["control_endpoint"] == "tcp://192.168.123.166:5572"


def test_service_check_is_only_the_offline_native_self_test(monkeypatch):
    commands = []
    monkeypatch.setattr(server.os, "access", lambda *args: True)
    monkeypatch.setattr(
        server.subprocess, "run", lambda cmd, **kw: commands.append(cmd) or SimpleNamespace(returncode=0)
    )
    assert server.main(["--check-only", "--dex1-worker", "/tmp/fake-worker"]) == 0
    assert commands == [["/tmp/fake-worker", "--self-test"]]
    with pytest.raises(SystemExit):
        server.main([])


@pytest.mark.parametrize(
    "args", [["--intent-port", "5570"], ["--state-port", "70000"], ["--dex1-transition-duration", "nan"]]
)
def test_service_rejects_invalid_configuration(args):
    with pytest.raises(ValueError):
        server.worker_command(server.build_parser().parse_args(args))


@pytest.mark.parametrize("source_clock", [1, 9_000_000_000_000_000])
def test_hand_watchdog_uses_receiver_clock_despite_different_source_clock(source_clock):
    device = SimHandBackend(DEX1.left)
    controller = SafeHandController(DEX1, {"left": device}, backend_name="sim", clock=lambda: 10.0)
    intent = {
        "sequence": 10,
        "monotonic_ns": source_clock,
        "hold": False,
        "left": {"valid": True, "closed": True},
    }
    assert controller.accept_intent(intent, now=10.0)
    assert not controller.step(now=10.02)["input_stale"]
    stale = controller.step(now=10.6)
    assert stale["input_stale"]
    before = device.read_positions()
    assert not controller.accept_intent(intent, now=10.7)  # replay cannot refresh the watchdog
    controller.step(now=10.7)
    np.testing.assert_array_equal(device.read_positions(), before)
    controller.close()


@pytest.mark.parametrize("host,scope", [("localhost", "single_linux_host"), ("192.168.123.164", "per_host")])
def test_recording_metadata_identifies_cross_host_clock_domains(host, scope):
    metadata = _capture_reproducibility_metadata({}, 50, DEX1, hand_state_host=host)
    assert metadata["capture_monotonic_clock_scope"] == scope
    assert metadata["hand_state_host"] == host
    assert metadata["hand_capture_clock_domains"]["hand_intent_received_monotonic_ns"] == "hand_server"
    assert metadata["hand_capture_clock_domains"]["hand_state_received_monotonic_ns"] == "recorder_host"
