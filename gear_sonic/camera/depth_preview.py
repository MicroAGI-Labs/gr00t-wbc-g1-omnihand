"""Depth video rendering shared by the browser UI and dataset recorder."""

import cv2
import numpy as np


def colorize_depth(depth: np.ndarray) -> np.ndarray | None:
    """Build a display-only RGB fallback when the camera supplies no depth view."""
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
    preview = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def camera_images_with_depth_preview(images: dict, depths: dict) -> dict:
    """Preserve the camera's rendered depth view; colorize raw depth only as fallback."""
    images = dict(images)
    # ZED VIEW.DEPTH already provides the SDK visualization. Replacing it with
    # per-frame percentile colors discards that rendering and changes the color
    # scale as the scene or far-depth outliers change.
    if images.get("ego_view_depth") is not None:
        return images
    depth = depths.get("ego_view_depth")
    if depth is not None:
        preview = colorize_depth(depth)
        if preview is not None:
            images["ego_view_depth"] = preview
    return images
