# 2025 Steven Chiacchira + Claude Sonnet 4
"""
:py:class:`gymnasium.Env` representing a game of paint chase from UFO 50.

Requires UFO 50 to be running in a specified screen region to work.
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import pynput
import torch
import torchvision

from paint_chaser.environment.ScreenCamera import ScreenCamera
from paint_chaser.environment.UFO50PaintChaseController import UFO50PaintChaseController
from paint_chaser.environment.UFO50PaintChaseTile import UFO50PaintChaseTile


class UFO50PaintChaseEnv(gym.Env):
    """
    :py:class:`gymnasium.Env` representing paint chase.

    UFO 50 must be running in a specified screen region to work.
    """

    BOARD_DIM = (9, 15)
    BAD_COLOR = np.array([209, 15, 76])
    GOOD_COLOR = np.array([0, 106, 180])
    METER_COLOR = np.array([254, 112, 0])
    MENU_BORDER_COLOR = np.array([255, 255, 255])
    ARRAY_TO_TENSOR_TRANSFORM = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.0, 1.0),
            torchvision.transforms.Resize((90, 160)),
        ],
    )
    BLOCK = UFO50PaintChaseTile.BLOCK

    @classmethod
    def _approx_equal_tensor(
        cls,
        a: torch.Tensor,
        b: float | torch.Tensor,
        eps: float = 0.001,
    ) -> torch.Tensor:
        """Return `True` if `a` is equal to `b` within `eps` tolerance."""
        is_approx_equal = torch.abs(a - b) <= eps
        return is_approx_equal

    @classmethod
    def _capture_indicates_round_progressing(cls, capture: np.typing.NDArray) -> bool:
        """
        Indicate if `capture` indicates the round is progressing.

        `True` if both of the following conditions are met in paint chase as indicated
        by `capture`:
            - The round timer (meter) is not empty. Detected via orange pixels at the
            bottom of the timer.
            - The end of round scoring menu is not shown. Detected via the lower white
            border of the scoring meter.
        and `False` otherwise.

        :param capture: a non-transformed image of the game screen.

        :returns: `True` if `capture` indicates the round is progressing,
        and `False` otherwise.
        """
        meter_final_pixel_color = capture[185, 370]
        has_time_left = (meter_final_pixel_color == cls.METER_COLOR).all()

        menu_border_pixels = capture[110, 103:303]
        end_menu_shown = (menu_border_pixels == cls.MENU_BORDER_COLOR).all()

        return has_time_left and not end_menu_shown

    def _transform_capture_to_game_state(
        self,
        capture: np.typing.NDArray,
    ) -> torch.Tensor:
        """
        Transform `capture` to a game state :py:class`torch.Tensor`.

        Applies the following transforms to `capture` in order:
            1. Converts `capture` from a :py:class:`numpy.typing.NDArray`
            to a :py:class:`torch.Tensor`.
                a. Converts `capture`'s dtype to the torch default.
                b. Rescales `capture` from [0, 255] to [0.0, 1.0]
            2. Cuts the round timer (meter) from `capture`.
            3. Pads `capture` with green obstacle blocks.
        and returns the result.

        :param capture: the capture to apply the transform to.

        :returns: the transformed capture.
        """
        transformed = self.ARRAY_TO_TENSOR_TRANSFORM(capture)
        meter_clipped = transformed[:, :, :150]

        tile_padded = self._frame.clone()
        tile_padded[:, 10:100, 10:160] = meter_clipped

        return tile_padded

    def __init__(
        self,
        screen_region: dict[str, int],
    ) -> None:
        """
        Construct a new :py:class:`gymnasium.Env` for paint chase from UFO 50.

        :param screen_region: the region of the screen in which paint chase is running.
        Note that this region should not include any window decorations, and just the
        game itself.
        """
        self._camera = ScreenCamera(screen_region)
        self._controller = UFO50PaintChaseController()
        self._frame = torch.tile(
            self.BLOCK.value,
            (self.BOARD_DIM[0] + 2, self.BOARD_DIM[1] + 2),
        )

    def reset(
        self,
        *_: dict[str, Any] | None,
        ) -> tuple[object, dict]:
        """
        Reset the environment for a new episode.

        Performs the following actions repeatedly until the game resets:
            1. Attempts to reset paint chaser using the game controller.
            2. Takes a capture of the game screen.
            3. Checks the capture for game progress.

        Once the game is reset, the most recent capture from (2) will be transformed to
        a game state and returned as a tuple element.

        :returns: a tuple containing a game state and an empty :py:type:`dict`.
        """
        finished_reset = False
        while not finished_reset:
            self._controller.reset()
            for _reset_attempt in range(5):
                capture = self._camera.get_capture()
                if self._capture_indicates_round_progressing(capture):
                    finished_reset = True
                    break
                time.sleep(0.2)

        capture = self._transform_capture_to_game_state(capture)

        return capture, {}

    def step(
            self,
            action: int,
            ) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, Any]]:
        """
        Perform an action as specified by `action` in paint chase.

        Requires the paint chase window to be selected.
        Performs the following in order:
            1. Performs `action` as specified by the following map:
                0 -> Up arrow
                1 -> Down arrow
                2 -> Left arrow
                3 -> Right arrow
            2. Takes a capture of the game area.
            3. Computes the number of good pixels and the number of bad pixels.
            3. Determines if the game has ended from the capture from (2).
            4. Transforms the capture from (2) into a game state tensor.
            5. Returns the tuple (game_state_tensor, reward, finished, False, {}).

        :param action: an integer representing the action to take.

        :returns: the tuple (game_state_tensor, [#good_pixels, #bad_pixels], finished, False, {})
        """
        match action:
            case 0:
                self._controller.tap(pynput.keyboard.Key.up)
            case 1:
                self._controller.tap(pynput.keyboard.Key.down)
            case 2:
                self._controller.tap(pynput.keyboard.Key.left)
            case 3:
                self._controller.tap(pynput.keyboard.Key.right)

        capture = self._camera.get_capture()

        n_good_pixels = (capture == self.GOOD_COLOR).sum()
        n_bad_pixels = (capture == self.BAD_COLOR).sum()

        # n_good_pixels - self._bad_pixel_penalty  n_bad_pixels
        reward = torch.tensor([n_good_pixels, n_bad_pixels])
        finished = not self._capture_indicates_round_progressing(capture)

        capture = self._transform_capture_to_game_state(capture)

        return capture, reward, finished, False, {}

    def render(self) -> None:
        """
        Unimplemented method. Do not call.

        :raises NotImplementedError: do not call this method.
        """
        raise NotImplementedError
