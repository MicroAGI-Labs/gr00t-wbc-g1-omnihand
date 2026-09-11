from types import SimpleNamespace

import pytest

from gear_sonic.scripts import launch_data_collection as launcher


@pytest.mark.parametrize(
    "sim,active,listening,action",
    [
        (False, True, [True], None),
        (False, True, [False, False, True], None),
        (False, False, [False, True], "start"),
        (True, False, [False], None),
        (True, False, [True, True, False], None),
        (True, True, [True, False], "stop"),
    ],
)
def test_camera_service_only_changes_state_when_needed(monkeypatch, sim, active, listening, action):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0 if command[0] == "sudo" or active else 3)

    readiness = iter(listening)
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher, "_camera_port_is_listening", lambda port: next(readiness))
    monkeypatch.setattr(launcher.time, "sleep", lambda delay: None)

    launcher._switch_camera_source(launcher.DataCollectionLaunchConfig(sim=sim))

    assert calls[0] == ["systemctl", "is-active", "--quiet", "composed_camera_server.service"]
    assert calls[1:] == (
        [["sudo", "-n", "systemctl", action, "composed_camera_server.service"]] if action else []
    )


def test_running_camera_without_stream_reports_readiness_failure_without_sudo(monkeypatch):
    def run(command, **kwargs):
        assert command == ["systemctl", "is-active", "--quiet", "composed_camera_server.service"]
        return SimpleNamespace(returncode=0)

    ticks = iter((0.0, 0.0, 61.0))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher, "_camera_port_is_listening", lambda port: False)
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(launcher.time, "sleep", lambda delay: None)

    with pytest.raises(RuntimeError, match="Camera port 5555 did not start listening.*journalctl"):
        launcher._switch_camera_source(launcher.DataCollectionLaunchConfig())


@pytest.mark.parametrize("record_cameras", [True, False])
def test_launcher_keeps_direct_vr_controls_inside_restart_loop(monkeypatch, record_cameras):
    import shlex

    commands = {}
    for name in ("_check_prerequisites", "_check_remote_hand_connection", "_kill_existing_session",
                 "_switch_camera_source", "_create_tmux_session", "_select_dashboard"):
        monkeypatch.setattr(launcher, name, lambda *args: None)
    monkeypatch.setattr(launcher, "_get_local_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(launcher, "_check_pane_alive", lambda *args: True)
    monkeypatch.setattr(launcher, "_send_to_pane", lambda pane, command, **kwargs: commands.update({pane: command}))
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    config = launcher.DataCollectionLaunchConfig(
        hand_backend="none", camera_viewer=False, camera_server_logs=False,
    )
    if not record_cameras:
        config.record_wrist_cameras = False
        config.record_zed_stereo = False
        config.max_episode_duration_s = 0.0
    launcher.main(config)
    lexer = shlex.shlex(commands[1], posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    args = list(lexer)
    for flag, expected in (("--teleop-mode", "vr3pt"), ("--initial-locomotion-mode", "slow_walk"),
                           ("--teleop-control-port", str(config.teleop_control_port))):
        assert args.index("do") < args.index(flag) < args.index("done")
        assert args[args.index(flag) + 1] == expected
    assert "--manager" in args
    assert not any("limiter" in arg or arg.startswith("--vr-max-") for arg in args)
    from gear_sonic.scripts.run_data_exporter import SonicDataExporterConfig
    import tyro

    exporter_args = shlex.split(commands[2].split("run_data_exporter.py ", 1)[1])
    exporter_config = tyro.cli(SonicDataExporterConfig, args=exporter_args)
    assert exporter_config.record_wrist_cameras is record_cameras
    assert exporter_config.record_zed_stereo is record_cameras
    assert exporter_config.max_episode_duration_s == (240.0 if record_cameras else 0.0)
