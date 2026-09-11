from types import SimpleNamespace

import pytest

from gear_sonic.scripts import run_data_exporter as recorder
from gear_sonic.utils.data_collection.episode_state import EpisodeState


@pytest.fixture(params=[120.0, 240.0])
def recording(monkeypatch, request):
    now = [100.0]
    key = [None]
    monkeypatch.setattr(recorder.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(recorder.time, "monotonic_ns", lambda: int(now[0] * 1e9))
    c = object.__new__(recorder.GrootDataCollector)
    c._episode_state = EpisodeState()
    c.max_episode_duration_s = request.param
    c._episode_started_at = c._episode_stopped_at = None
    c._keyboard_listener = SimpleNamespace(read_msg=lambda: key.pop(0) if key else None)
    c._manager_toggle_dc = c._manager_toggle_da = False
    c._manager_discard_reason = None
    c.current_stream_mode = c.required_stream_mode = 5
    c.require_hub_upload = False
    c._sender_sync = None
    c._episode_input_errors = set()
    c.sonic_timing_monitor = SimpleNamespace(reset=lambda: None)
    c._print_and_say = lambda *args, **kwargs: None
    events = []
    c._set_recording_audio_event = events.append
    c.data_exporter = SimpleNamespace(episode_buffer={"episode_index": 0, "size": 0}, video_writers={})
    def detach(*, advance_index=True):
        completed = c.data_exporter.episode_buffer
        c.data_exporter.episode_buffer = {"episode_index": completed["episode_index"] + int(advance_index), "size": 0}
        return completed, {}
    c.data_exporter.detach_episode = detach
    jobs = []
    c.episode_finalizer = SimpleNamespace(can_record=lambda: True, enqueue=lambda **job: jobs.append(job))
    c._episode_validation = lambda: {"passed": True, "errors": []}
    key[:] = ["c"]
    c._check_recording_commands()
    assert c._episode_started_at == 100.0
    c.data_exporter.episode_buffer["size"] = 10
    return c, now, key, events, jobs


@pytest.mark.parametrize("save_at_boundary", [False, True])
def test_elapsed_limit_discards_once_even_during_pause_or_missing_frames(recording, save_at_boundary):
    c, now, key, events, jobs = recording
    # A paused teleop and only ten collected frames must not extend the deadline.
    c.current_stream_mode = 3
    now[0] = 100.0 + c.max_episode_duration_s - 0.01
    c._check_recording_commands()
    assert c._episode_state.get_state() == "recording" and not jobs
    now[0] = 100.0 + c.max_episode_duration_s
    c._manager_toggle_dc = save_at_boundary
    c._check_recording_commands()
    assert c._episode_state.get_state() == "idle"
    assert c.current_stream_mode == 3
    assert c._episode_started_at is None
    assert events == ["start", "discard"]
    assert jobs[0]["success"] is False
    assert jobs[0]["delete"] is True
    assert c.data_exporter.episode_buffer["episode_index"] == 0
    assert jobs[0]["episode_buffer"]["size"] == 10
    assert jobs[0]["validation"] == {"passed": False, "errors": ["episode_duration_limit"],
                                      "failure_reason": "episode_duration_limit",
                                      "episode_duration_s": c.max_episode_duration_s,
                                      "max_episode_duration_s": c.max_episode_duration_s}
    assert "recording limit; removing its temporary videos" in c._recording_message
    assert "failed validation" not in c._recording_message
    c._check_recording_commands()
    assert len(jobs) == 1 and c._episode_state.get_state() == "idle"


def test_operator_can_accept_an_episode_below_limit(recording):
    c, now, key, events, jobs = recording
    now[0] = 100.0 + c.max_episode_duration_s - 0.01
    key[:] = ["c"]
    c._check_recording_commands()
    assert jobs[0]["success"] is True
    assert events == ["start", "saved"]


def test_sender_drain_does_not_count_as_recording_time(recording):
    c, now, key, events, jobs = recording
    stops = []
    c._sender_sync = SimpleNamespace(stop=stops.append, reset=lambda: None)
    now[0] = 100.0 + c.max_episode_duration_s - 0.01
    key[:] = ["c"]
    c._check_recording_commands()
    assert c._episode_state.get_state() == "need_to_save"
    now[0] = 100.0 + c.max_episode_duration_s + 10.0
    assert c._episode_elapsed_s() == pytest.approx(c.max_episode_duration_s - 0.01)
    c._finish_recording(save=True, discard_reason="operator_discarded")
    assert jobs[0]["success"] is True
    assert events == ["start", "saved"]


def test_limit_can_be_disabled(recording):
    c, now, key, events, jobs = recording
    c.max_episode_duration_s = 0
    now[0] = 1000.0
    c._check_recording_commands()
    assert not jobs and c._episode_state.get_state() == "recording"


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf")])
def test_bad_duration_limit_is_rejected_before_opening_robot_or_dataset(limit):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        recorder.main(recorder.SonicDataExporterConfig(max_episode_duration_s=limit))
