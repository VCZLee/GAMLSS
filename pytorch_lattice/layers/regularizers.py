"""Regularizers for calibrated modeling layers."""

import torch
import torch.nn as nn


class LaplacianRegularizer(nn.Module):
    """Laplacian regularizer for PWL calibration layer.

    Calibrator Laplacian regularization penalizes the change in the calibration
    output. It is defined to be:

    `l1 * ||delta||_1 + l2 * ||delta||_2^2`

    where `delta` is the segment heights of the PWL calibrator.
    """

    def __init__(
        self,
        l1: float = 0.0,
        l2: float = 0.0,
        is_cyclic: bool = False,
        regularize_bias: bool = False,
    ):
        """Initializes an instance of `LaplacianRegularizer`.

        Args:
            l1: l1 regularization amount.
            l2: l2 regularization amount.
            is_cyclic: Whether the first and last keypoints should take the same
                output value.
            regularize_bias: Whether to also apply the regularization penalty to
                the bias term (the first weight). Useful for pushing the entire
                function towards a flat line at 0.
        """
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.is_cyclic = is_cyclic
        self.regularize_bias = regularize_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns regularization loss.

        Args:
            x: Tensor of shape: `(k, 1)` which represents weights of PWL
                calibration layer. First row of weights is bias term. All remaining
                represent delta in y-value compare to previous point (segment heights).
        """
        if not self.l1 and not self.l2:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)

        heights = x[1:]
        if self.is_cyclic:
            # Need to add such last height to make all heights to sum up to 0.0 in
            # order to make calibrator cyclic.
            last_height = -torch.sum(heights, dim=0, keepdim=True)
            heights = torch.cat([heights, last_height], dim=0)

        loss = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        if self.regularize_bias:
            bias = x[0:1]
            if self.l1:
                loss = loss + self.l1 * torch.sum(torch.abs(bias))
            if self.l2:
                loss = loss + self.l2 * torch.sum(torch.square(bias))

        if self.l1:
            loss = loss + self.l1 * torch.sum(torch.abs(heights))
        if self.l2:
            loss = loss + self.l2 * torch.sum(torch.square(heights))

        return loss


class HessianRegularizer(nn.Module):
    """Hessian regularizer for PWL calibration layer.

    Calibrator hessian regularizer penalizes the change in slopes of linear
    pieces. It is defined to be:

    `l1 * ||nonlinearity||_1 + l2 * ||nonlinearity||_2^2`

    where `nonlinearity` is the change in segment heights.
    """

    def __init__(self, l1: float = 0.0, l2: float = 0.0, is_cyclic: bool = False):
        """Initializes an instance of `HessianRegularizer`.

        Args:
            l1: l1 regularization amount.
            l2: l2 regularization amount.
            is_cyclic: Whether the first and last keypoints should take the same
                output value.
        """
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.is_cyclic = is_cyclic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns regularization loss.

        Args:
            x: Tensor of shape: `(k, 1)` which represents weights of PWL
                calibration layer. First row of weights is bias term. All remaining
                represent delta in y-value compare to previous point (segment heights).
        """
        if not self.l1 and not self.l2:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)

        if self.is_cyclic:
            heights = x[1:]
            last_height = -torch.sum(heights, dim=0, keepdim=True)
            heights = torch.cat([heights, last_height, heights[0:1]], dim=0)
            nonlinearity = heights[1:] - heights[:-1]
        else:
            nonlinearity = x[2:] - x[1:-1]

        loss = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        if self.l1:
            loss = loss + self.l1 * torch.sum(torch.abs(nonlinearity))
        if self.l2:
            loss = loss + self.l2 * torch.sum(torch.square(nonlinearity))

        return loss


class WrinkleRegularizer(nn.Module):
    """Wrinkle regularizer for PWL calibration layer.

    Calibrator wrinkle regularization penalizes the change in the second
    derivative. It is defined to be:

    `l1 * ||third_derivative||_1 + l2 * ||third_derivative||_2^2`

    where `third_derivative` is the change in nonlinearity.
    """

    def __init__(self, l1: float = 0.0, l2: float = 0.0, is_cyclic: bool = False):
        """Initializes an instance of `WrinkleRegularizer`.

        Args:
            l1: l1 regularization amount.
            l2: l2 regularization amount.
            is_cyclic: Whether the first and last keypoints should take the same
                output value.
        """
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.is_cyclic = is_cyclic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns regularization loss.

        Args:
            x: Tensor of shape: `(k, 1)` which represents weights of PWL
                calibration layer. First row of weights is bias term. All remaining
                represent delta in y-value compare to previous point (segment heights).
        """
        if not self.l1 and not self.l2:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        if x.shape[0] < 3:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)

        if self.is_cyclic:
            heights = x[1:]
            last_height = -torch.sum(heights, dim=0, keepdim=True)
            heights = torch.cat(
                [heights, last_height, heights[0:1], heights[1:2]], dim=0
            )
            nonlinearity = heights[1:] - heights[:-1]
        else:
            nonlinearity = x[2:] - x[1:-1]
        wrinkleness = nonlinearity[1:] - nonlinearity[:-1]

        loss = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        if self.l1:
            loss = loss + self.l1 * torch.sum(torch.abs(wrinkleness))
        if self.l2:
            loss = loss + self.l2 * torch.sum(torch.square(wrinkleness))

        return loss


class L1L2Regularizer(nn.Module):
    """L1/L2 regularizer for layer weights.

    This regularizer penalizes the absolute and/or squared values of the weights
    directly. It is defined to be:

    `l1 * ||x||_1 + l2 * ||x||_2^2`

    where `x` are the weights of the layer.
    """

    def __init__(self, l1: float = 0.0, l2: float = 0.0):
        """Initializes an instance of `L1L2Regularizer`.

        Args:
            l1: l1 regularization amount.
            l2: l2 regularization amount.
        """
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns regularization loss.

        Args:
            x: Tensor representing weights of a layer.
        """
        if not self.l1 and not self.l2:
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)

        loss = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        if self.l1:
            loss = loss + self.l1 * torch.sum(torch.abs(x))
        if self.l2:
            loss = loss + self.l2 * torch.sum(torch.square(x))

        return loss
