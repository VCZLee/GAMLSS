"""Tests for regularizers."""

import torch
from pytorch_lattice.layers.regularizers import (
    LaplacianRegularizer,
    HessianRegularizer,
    WrinkleRegularizer,
)


def test_laplacian_regularizer():
    """Tests LaplacianRegularizer."""
    # x = [bias, h1, h2, h3]
    x = torch.tensor([[1.0], [0.5], [0.2], [-0.1]]).double()

    # l1 only
    reg = LaplacianRegularizer(l1=1.0, l2=0.0)
    # loss = |0.5| + |0.2| + |-0.1| = 0.8
    assert torch.allclose(reg(x), torch.tensor(0.8).double())

    # l2 only
    reg = LaplacianRegularizer(l1=0.0, l2=1.0)
    # loss = 0.5^2 + 0.2^2 + (-0.1)^2 = 0.25 + 0.04 + 0.01 = 0.3
    assert torch.allclose(reg(x), torch.tensor(0.3).double())

    # is_cyclic
    reg = LaplacianRegularizer(l1=1.0, l2=0.0, is_cyclic=True)
    # heights = [0.5, 0.2, -0.1]
    # last_height = -(0.5 + 0.2 - 0.1) = -0.6
    # loss = |0.5| + |0.2| + |-0.1| + |-0.6| = 1.4
    assert torch.allclose(reg(x), torch.tensor(1.4).double())

    # L1 + L2 + is_cyclic
    reg = LaplacianRegularizer(l1=1.0, l2=0.5, is_cyclic=True)
    # heights = [0.5, 0.2, -0.1, -0.6]
    # L1 = 0.5 + 0.2 + 0.1 + 0.6 = 1.4
    # L2 = 0.5 * (0.5^2 + 0.2^2 + (-0.1)^2 + (-0.6)^2)
    # L2 = 0.5 * (0.25 + 0.04 + 0.01 + 0.36) = 0.5 * 0.66 = 0.33
    # Total = 1.4 + 0.33 = 1.73
    assert torch.allclose(reg(x), torch.tensor(1.73).double())


def test_hessian_regularizer():
    """Tests HessianRegularizer."""
    # x = [bias, h1, h2, h3]
    x = torch.tensor([[1.0], [0.5], [0.2], [-0.1]]).double()

    # l1 only
    reg = HessianRegularizer(l1=1.0, l2=0.0)
    # nonlinearity = [h2-h1, h3-h2] = [0.2-0.5, -0.1-0.2] = [-0.3, -0.3]
    # loss = |-0.3| + |-0.3| = 0.6
    assert torch.allclose(reg(x), torch.tensor(0.6).double())

    # is_cyclic
    reg = HessianRegularizer(l1=1.0, l2=0.0, is_cyclic=True)
    # heights = [0.5, 0.2, -0.1, -0.6, 0.5] (including implicit last and wrap around)
    # nonlinearity = [0.2-0.5, -0.1-0.2, -0.6-(-0.1), 0.5-(-0.6)] = [-0.3, -0.3, -0.5, 1.1]
    # loss = 0.3 + 0.3 + 0.5 + 1.1 = 2.2
    assert torch.allclose(reg(x), torch.tensor(2.2).double())

    # L1 + L2 + is_cyclic
    reg = HessianRegularizer(l1=1.0, l2=0.5, is_cyclic=True)
    # L1 = 2.2
    # L2 = 0.5 * (0.09 + 0.09 + 0.25 + 1.21) = 0.5 * 1.64 = 0.82
    # Total = 2.2 + 0.82 = 3.02
    assert torch.allclose(reg(x), torch.tensor(3.02).double())


def test_wrinkle_regularizer():
    """Tests WrinkleRegularizer."""
    # x = [bias, h1, h2, h3, h4]
    x = torch.tensor([[1.0], [0.5], [0.2], [-0.1], [0.3]]).double()

    # l1 only
    reg = WrinkleRegularizer(l1=1.0, l2=0.0)
    # nonlinearity = [h2-h1, h3-h2, h4-h3] = [-0.3, -0.3, 0.4]
    # wrinkleness = [-0.3 - (-0.3), 0.4 - (-0.3)] = [0.0, 0.7]
    # loss = 0.0 + 0.7 = 0.7
    assert torch.allclose(reg(x), torch.tensor(0.7).double())

    # L1 + L2 + is_cyclic
    reg = WrinkleRegularizer(l1=1.0, l2=0.5, is_cyclic=True)
    # heights = [0.5, 0.2, -0.1, 0.3, -0.9, 0.5, 0.2]
    # nonlinearity = [-0.3, -0.3, 0.4, -1.2, 1.4, -0.3]
    # wrinkleness = [0.0, 0.7, -1.6, 2.6, -1.7]
    # L1 = 0.0 + 0.7 + 1.6 + 2.6 + 1.7 = 6.6
    # L2 = 0.5 * (0.0 + 0.49 + 2.56 + 6.76 + 2.89) = 0.5 * 12.7 = 6.35
    # Total = 6.6 + 6.35 = 12.95
    assert torch.allclose(reg(x), torch.tensor(12.95).double())
