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


@pytest.mark.parametrize("speed", [None, 0.5])
def test_launcher_forwards_jerk_and_trace_settings_inside_restart_loop(monkeypatch, speed):
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
        vr_max_angular_jerk_deg=7200,
        vr_motion_log_dir="/tmp/VR trials literal $path", disable_vr_motion_limiter=False,
    )
    if speed is not None:
        config.vr_max_speed = speed
        config.vr_max_acceleration = 0.75
        config.vr_max_jerk = 12
    launcher.main(config)
    lexer = shlex.shlex(commands[1], posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    args = list(lexer)
    assert float(args[args.index("--vr-max-speed") + 1]) == (0.35 if speed is None else speed)
    assert args.index("do") < args.index("--vr-max-speed") < args.index("done")
    assert float(args[args.index("--vr-max-acceleration") + 1]) == (0.9 if speed is None else 0.75)
    assert args.index("do") < args.index("--vr-max-acceleration") < args.index("done")
    assert float(args[args.index("--vr-max-jerk") + 1]) == (9 if speed is None else 12)
    assert args[args.index("--vr-max-angular-jerk-deg") + 1] == "7200"
    assert args[args.index("--vr-motion-log-dir") + 1] == config.vr_motion_log_dir
    assert args.index("do") < args.index("--vr-max-jerk") < args.index("done")
    assert "--disable-vr-motion-limiter" not in args
