from collections import deque

from gear_sonic.scripts.run_camera_web_viewer import CameraFrameHub


def test_camera_rates_are_reported_per_stream():
    hub = object.__new__(CameraFrameHub)
    hub._camera_samples = {
        "left_wrist": deque([(0.0, 1.0), (0.02, 2.0), (0.04, 3.0)]),
        "right_wrist": deque([(0.0, 1.0)]),
    }
    assert hub._camera_rates() == {"left_wrist": 50.0}
