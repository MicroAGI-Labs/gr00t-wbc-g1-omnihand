from types import SimpleNamespace

import pytest

from gear_sonic.end_effectors import server


def test_worker_command_routes_remote_intent_and_state():
    args = server.build_parser().parse_args(["--teleop-host", "orin", "--enable-command"])
    command = server.worker_command(args)
    assert "tcp://orin:5569" in command
    assert "tcp://0.0.0.0:5570" in command
    assert "--enable-command" in command


@pytest.mark.parametrize("args", [["--intent-port", "5570"], ["--state-port", "70000"]])
def test_worker_command_rejects_invalid_ports(args):
    with pytest.raises(ValueError):
        server.worker_command(server.build_parser().parse_args(args))


def test_check_only_runs_only_worker_self_test(monkeypatch):
    calls = []
    monkeypatch.setattr(server.os, "access", lambda *args: True)
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    assert server.main(["--check-only", "--dex1-worker", "/tmp/fake-worker"]) == 0
    assert calls == [["/tmp/fake-worker", "--self-test"]]


def test_physical_server_requires_explicit_enable_command(monkeypatch):
    monkeypatch.setattr(server.os, "access", lambda *args: True)
    with pytest.raises(SystemExit):
        server.main(["--dex1-worker", "/tmp/fake-worker"])
