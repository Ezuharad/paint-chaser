# 2025 Steven Chiacchira
"""Utility class for capturing screen regions."""
import numpy as np
from PIL import ImageGrab


class ScreenCamera:
    """Utility class for capturing screen regions."""

    def __init__(self, region: dict[str, int]) -> None:
        """
        Construct a new `ScreenCamera`.

        :param: region: the region to capture, expressed as a :py:type:`dict` with the
        following keys:
            - "left": the leftmost pixel of the region to capture, measured from the
            left of the screen.
            - "top": the topmost pixel of the region to capture, measured from the top
            of the screen.
            - "width": the width of the screen capture.
            - "height": the height of the screen capture.
        """
        self._region = (
            region["left"],
            region["top"],
            region["left"] + region["width"],
            region["top"] + region["height"],
        )

    def get_capture(self) -> np.typing.NDArray:
        """
        Take a capture of the defined region using :py:func:`PIL.ImageGrab.grab`.

        :returns: a :py:type:`numpy.typing.NDArray` representing the screen region as a
        three channel (RGB) image.
        """
        capture = ImageGrab.grab(self._region)
        capture = np.array(capture)[:, :, :3]

        return capture
