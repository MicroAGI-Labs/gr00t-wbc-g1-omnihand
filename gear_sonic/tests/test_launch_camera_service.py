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
