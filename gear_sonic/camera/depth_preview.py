"""Depth video rendering shared by the browser UI and dataset recorder."""

import cv2
import numpy as np


def colorize_depth(depth: np.ndarray) -> np.ndarray | None:
    """Preserve the browser's display-only depth colors and per-frame scaling."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return None
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return None
    low, high = np.percentile(values[valid], [2, 98])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized[~valid] = 0.0
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def camera_images_with_depth_preview(images: dict, depths: dict) -> dict:
    """Use the UI preview when available, otherwise keep the camera's video."""
    images = dict(images)
    depth = depths.get("ego_view_depth")
    if depth is not None:
        preview = colorize_depth(depth)
        if preview is not None:
            images["ego_view_depth"] = preview
    return images
