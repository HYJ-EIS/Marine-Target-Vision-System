"""OpenCV utilities for handling file paths with non-ASCII characters (e.g. Chinese) on Windows."""

import cv2
import numpy as np


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image from a path that may contain non-ASCII characters."""
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def imwrite_unicode(filepath: str, img: np.ndarray, params: list | None = None) -> bool:
    """Write an image to a path that may contain non-ASCII characters."""
    try:
        ext = '.' + filepath.rsplit('.', 1)[-1] if '.' in filepath else '.jpg'
        if params is not None:
            result, buf = cv2.imencode(ext, img, params)
        else:
            result, buf = cv2.imencode(ext, img)
        if result:
            buf.tofile(filepath)
            return True
        return False
    except Exception:
        return False
