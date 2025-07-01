# 2025 Steven Chiacchira
"""Utility class for controlling paint chase from UFO 50."""

import time

from pynput import keyboard


class UFO50PaintChaseController:
    """Controller for paint chase from UFO 50."""

    def __init__(self) -> None:
        """
        Construct a new `UFO50PaintChaseController`.

        Note that for controls to be registered by paint chase, UFO 50 must be in focus
        on the desktop.
        """
        self._keyboard = keyboard.Controller()

    def tap(self, key: str | keyboard.Key) -> None:
        """
        Immediately press and releas `key` on the keyboard.

        :param key: the key to press.
        """
        self._keyboard.press(key)
        self._keyboard.release(key)

    def reset(self) -> None:
        """
        Attempt to reset paint chase using the escape menu.

        Due to timing issues with dropped inputs, this method takes at least

        5.2 seconds to return.
        """
        to_register_wait_sec = 0.1
        to_reset_wait_sec = 0.8
        to_start_game_wait_sec = 1.8
        countdown_wait_sec = 2.1

        self.tap(keyboard.Key.esc)
        time.sleep(to_register_wait_sec)
        self.tap(keyboard.Key.down)
        time.sleep(to_register_wait_sec)
        self.tap("z")
        time.sleep(to_register_wait_sec)
        self.tap("z")
        time.sleep(to_reset_wait_sec)
        self.tap("z")
        time.sleep(to_start_game_wait_sec)
        self.tap("z")
        time.sleep(to_register_wait_sec)
        self.tap("z")
        time.sleep(to_register_wait_sec)
        self.tap("z")
        time.sleep(countdown_wait_sec)

    def step(self) -> None:
        """
        Presses the `z` key in order to escape score menu popups from blowouts.

        ..deprecated:: 1.0
            No longer used.
        """
        self.tap("z")
