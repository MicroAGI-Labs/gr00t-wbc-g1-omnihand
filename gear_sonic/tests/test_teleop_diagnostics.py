"""Recovery guidance must distinguish hardware, input, and recorder failures."""
import json

import pytest

from gear_sonic.scripts import check_teleop


def ready():
    return {'connected': True, 'sources': {'camera': True, 'proprio': True},
            'recording_mode_ready': True, 'stream_rates': {'manager_state': {'active': True}}}


def test_connected_hardware_without_input_is_not_diagnosed_as_usb_failure():
    messages = check_teleop.recovery_advice(ready(), {'connected': True, 'input_stale': True})
    assert len(messages) == 1
    assert 'hardware is connected and holding' in messages[0]
    assert 'Reconnect hands' not in messages[0]


@pytest.mark.parametrize('state,text', [('fault', 'latched'), ('reconnecting', 'retrying USB'),
                                        ('offline', 'unavailable'), ('stale', 'unavailable')])
def test_hand_failure_states_have_distinct_recovery(state, text):
    messages = check_teleop.recovery_advice(ready(), {'connected': False, 'connection_state': state})
    assert len(messages) == 1
    assert text in messages[0]


def test_offline_recorder_does_not_hide_a_hand_fault():
    messages = check_teleop.recovery_advice({'connected': False}, {'connection_state': 'fault'})
    assert len(messages) == 2
    assert 'Recorder offline' in messages[0]
    assert 'latched' in messages[1]


def test_save_failure_and_missing_tracking_both_reported():
    recording = ready()
    recording['finalizer'] = {'error': 'disk full'}
    recording['stream_rates'] = {}
    messages = check_teleop.recovery_advice(recording, {'connected': True})
    assert any('preserve the dataset' in message for message in messages)
    assert any('PICO manager not publishing' in message for message in messages)


def test_capture_survives_offline_services_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(check_teleop, 'fetch_status', lambda *args: {'connected': False})
    commands = []
    def command(args, **kwargs):
        commands.append(args)
        return {'returncode': 1, 'stderr': 'service unavailable'}
    monkeypatch.setattr(check_teleop, 'run_command', command)
    output = tmp_path / 'diagnostics'
    messages = check_teleop.collect(output, base_url='http://localhost:8080', duration=0)
    assert len(messages) == 2
    assert json.loads((output / 'status.json').read_text())[0]['recording']['connected'] is False
    assert (output / 'summary.txt').is_file()
    assert all(args[0] in {'git', 'tmux', 'df', 'journalctl'} for args in commands)
    assert not any('send-keys' in args or 'respawn-pane' in args for args in commands)
    with pytest.raises(FileExistsError):
        check_teleop.collect(output, base_url='http://localhost:8080', duration=0)
