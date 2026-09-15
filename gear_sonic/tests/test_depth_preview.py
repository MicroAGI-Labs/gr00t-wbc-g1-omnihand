import numpy as np
import pytest

from gear_sonic.camera.depth_preview import camera_images_with_depth_preview


def test_sdk_depth_view_is_preserved_despite_changing_raw_depth():
    sdk = np.full((2, 3, 3), [220, 50, 10], dtype=np.uint8)
    images = {"ego_view_depth": sdk}
    for depth in (
        np.array([[500, 750, 1000], [1200, 1500, 2000]], dtype=np.float32),
        np.array([[500, 750, 1000], [1200, 1500, 20000]], dtype=np.float32),
    ):
        original = depth.copy()
        result = camera_images_with_depth_preview(images, {"ego_view_depth": depth})
        np.testing.assert_array_equal(result["ego_view_depth"], sdk)
        np.testing.assert_array_equal(depth, original)
        assert result is not images
    assert images["ego_view_depth"] is sdk


def test_raw_depth_fallback_is_rgb_and_marks_missing_depth_black():
    images = {"ego_view": np.zeros((2, 5, 3), dtype=np.uint8)}
    depth = np.array([[1, 2, 3, 4, 5], [6, np.nan, np.inf, -np.inf, 0]], dtype=np.float32)
    result = camera_images_with_depth_preview(images, {"ego_view_depth": depth})
    preview = result["ego_view_depth"]
    assert preview.dtype == np.uint8
    assert preview.shape == (2, 5, 3)
    # The far end of Turbo is red, so RGB channel zero must exceed blue.
    assert preview[1, 0, 0] > preview[1, 0, 2]
    np.testing.assert_array_equal(preview[1, 1:], 0)
    assert "ego_view_depth" not in images


@pytest.mark.parametrize("depth", [np.zeros((2, 3)), np.full((2, 3), np.nan)])
def test_no_depth_image_is_invented_without_valid_measurements(depth):
    assert camera_images_with_depth_preview({}, {"ego_view_depth": depth}) == {}
