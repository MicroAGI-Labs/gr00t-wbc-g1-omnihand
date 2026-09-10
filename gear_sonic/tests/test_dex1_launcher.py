import pytest

from gear_sonic.scripts import launch_data_collection as launcher


def test_dex1_local_worker_command():
    config = launcher.DataCollectionLaunchConfig(hand_backend="dex1")
    environment, command = launcher._hand_worker_command(config)
    assert environment == ".venv_data_collection"
    assert "--backend dex1" in command
    assert "--enable-command" in command


def test_remote_dex1_command_quotes_repo_and_uses_host_address():
    config = launcher.DataCollectionLaunchConfig(
        hand_backend="dex1", hand_server_host="orin", hand_server_repo="/tmp/a b"
    )
    command = launcher._remote_hand_command(config)
    assert "ssh" in command
    assert "orin" in command
    assert "/tmp/a b" in command
    assert "--enable-command" in command


@pytest.mark.parametrize(
    "host,backend,sim",
    [("bad host", "dex1", False), ("orin", "omnihand", False), ("orin", "dex1", True)],
)
def test_remote_configuration_is_rejected(monkeypatch, host, backend, sim):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(launcher.os, "access", lambda *args: True)
    config = launcher.DataCollectionLaunchConfig(
        hand_backend=backend, hand_server_host=host, sim=sim
    )
    with pytest.raises(SystemExit):
        launcher._check_prerequisites(config)


def test_check_only_does_not_create_tmux(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "_check_prerequisites", lambda config: None)
    monkeypatch.setattr(launcher, "_create_tmux_session", lambda config: pytest.fail("tmux started"))
    launcher.main(launcher.DataCollectionLaunchConfig(hand_backend="dex1", check_only=True))
    assert "no services started" in capsys.readouterr().out
