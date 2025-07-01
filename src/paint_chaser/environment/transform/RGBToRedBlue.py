# 2025 Steven Chiacchira + Claude Sonnet 4
"""Custom class for transforming an RGB image to one red-blue channel."""

import torch
from torch import nn


class RGBToRedBlue(nn.Module):
    """
    Transform that converts RGB images to a single channel representing red vs blue.

    Converts RGB tensor to single channel where:
        - 1 Represents red-ness (more red than blue).
        - -1 Represents blue-ness (more blue than red).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """
        Construct a new `RGBToRedBlue` transform.

        :param eps: a small value used to prevent dividing by zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply transform to `image`.

        The following formula is used to calculate the red blue score:
        `r / r + b + eps`

        :param image: a RGB tensor of shape (..., 3, H, W) with values
        in [0, 1] or [0, 255].

        :returns: a single channel tensor of shape (..., 1, H, W) with values in [0, 1].
        """
        r = image[..., 0:1, :, :]
        b = image[..., 2:3, :, :]

        denominator = r + b + self.eps
        redblue_score = r / denominator

        return redblue_score

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps})"
