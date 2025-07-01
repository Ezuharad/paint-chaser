# 2025 Steven Chiacchira (with help from ChatGPT and Claude Sonnet 4)
"""Neural network for playing UFO50"""

import torch
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """
    Stack of two convolutional layers with ReLU activations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        negative_slope: float,
    ) -> None:
        """
        Construct a new `DoubleConv`.

        :param in_channels: number of channels expected in input.
        :param out_channels: number of channels to output.
        :param kernel_size: size of convolutional kernel to use.
        :param negative_slope: negative slope to use for LeakyReLU activation.
        """
        super().__init__()
        self._double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding="same"
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding="same"
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._double_conv(x)


class ResBlock(nn.Module):
    """
    Residual block with LeakyReLU activations.
    """

    def __init__(self, n_features: int, negative_slope: float) -> None:
        """
        Construct a new `ResBlock`.

        :param n_features: number of features in the input and output.
        :param negative_slope: negative slope to use for LeakyReLU activation.
        """
        super().__init__()
        self._block = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(negative_slope),
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        combined = residual + self._block(x)

        return combined


class ResidualAgent(nn.Module):
    def __init__(
        self,
        negative_slope: float = 1e-2,
    ) -> None:
        """
        Construct a new `ResidualAgent`.

        Contains bot a CNN spine capable of feature extraction, as well as a residual network for logic and decision-making.
        :param negative_slope: negative slope to use for LeakyReLU.
        """
        super().__init__()
        self._spine = nn.Sequential(
            DoubleConv(3, 32, 3, negative_slope),
            nn.MaxPool2d(2, 2),
            DoubleConv(32, 32, 3, negative_slope),
            nn.MaxPool2d(2, 2),
            DoubleConv(32, 64, 3, negative_slope),
            nn.MaxPool2d(2, 2),
            DoubleConv(64, 64, 3, negative_slope),
            nn.MaxPool2d(2, 2),
            DoubleConv(64, 64, 3, negative_slope),
            nn.MaxPool2d(2, 2),
        )

        self._flatten = nn.Flatten()

        self._mlp = nn.Sequential(
            ResBlock(960, negative_slope),
            nn.Linear(960, 4),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._spine(x)
        x = self._flatten(x)
        x = self._mlp(x)

        return x
