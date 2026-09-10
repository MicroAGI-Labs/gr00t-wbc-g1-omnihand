"""Browser and recorder ownership of dataset configuration."""

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gear_sonic.scripts.run_camera_web_viewer import CameraWebViewerConfig, RecorderControlHub
from gear_sonic.scripts.run_data_exporter import GrootDataCollector
from gear_sonic.utils.data_collection.hub_config import DATASET_CONFIG_PREFIX
from gear_sonic.utils.data_collection.episode_state import EpisodeState

CONFIG = {"repo_id": "org/dataset", "prompt": "pick up the cup", "private": True}


def test_browser_validates_and_queues_configuration(monkeypatch):
    import gear_sonic.scripts.run_camera_web_viewer as viewer

    api = Mock()
    api.list_repo_files.return_value = []
    monkeypatch.setattr(viewer, "HfApi", lambda: api)
    hub = RecorderControlHub(CameraWebViewerConfig(hf_namespace="org"))
    hub._status = {"recording": False, "saving": False, "total_episodes": 0}
    hub._status_received_at = time.monotonic()
    hub.configure_dataset("dataset", CONFIG["prompt"], CONFIG["private"])
    assert json.loads(hub._commands.get_nowait()[len(DATASET_CONFIG_PREFIX):]) == CONFIG
    api.create_repo.assert_called_once_with("org/dataset", repo_type="dataset", private=True, exist_ok=True)
    for name, prompt, private in (("../dataset", "task", True), ("dataset", "", True), ("dataset", "task", "false")):
        with pytest.raises(ValueError):
            hub.configure_dataset(name, prompt, private)
    assert hub._commands.empty()


@pytest.mark.parametrize("busy", [None, "recording", "finalizing", "saved"])
def test_collector_owns_configuration_lock(busy):
    exporter = SimpleNamespace(
        task="demo", meta=SimpleNamespace(info={"total_episodes": 0}),
        episode_buffer={"size": 0},
    )
    configured = []

    def configure(**config):
        configured.append(config)
        exporter.task = config["prompt"]

    collector = GrootDataCollector.__new__(GrootDataCollector)
    collector.data_exporter = exporter
    collector.hub_uploader = SimpleNamespace(configure=configure)
    collector._episode_state = EpisodeState()
    if busy == "recording":
        collector._episode_state.change_state()
    if busy == "saved":
        exporter.meta.info["total_episodes"] = 1
    collector.episode_finalizer = SimpleNamespace(status=lambda: {"pending": int(busy == "finalizing"), "finalizing": False})
    collector._keyboard_listener = SimpleNamespace(read_msg=lambda: DATASET_CONFIG_PREFIX + json.dumps(CONFIG))
    collector._check_recording_commands()
    assert bool(configured) == (busy is None)
    assert exporter.task == ("demo" if busy else CONFIG["prompt"])
