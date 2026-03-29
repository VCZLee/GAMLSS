"""Tests for regularizers."""

import torch
from pytorch_lattice.layers.regularizers import (
    LaplacianRegularizer,
    HessianRegularizer,
    WrinkleRegularizer,
    L1L2Regularizer,
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

    # cyclic only
    reg = LaplacianRegularizer(l1=1.0, l2=0.0, is_cyclic=True)
    # heights = [0.5, 0.2, -0.1]
    # last_height = -(0.5 + 0.2 - 0.1) = -0.6
    # loss = |0.5| + |0.2| + |-0.1| + |-0.6| = 1.4
    assert torch.allclose(reg(x), torch.tensor(1.4).double())

    # l1 + l2 + cyclic
    reg = LaplacianRegularizer(l1=1.0, l2=1.0, is_cyclic=True)
    # heights = [0.5, 0.2, -0.1, -0.6]
    # L1 = 0.5 + 0.2 + 0.1 + 0.6 = 1.4
    # L2 = 0.5^2 + 0.2^2 + (-0.1)^2 + (-0.6)^2 = 0.25 + 0.04 + 0.01 + 0.36 = 0.66
    # Total = 1.4 + 0.66 = 2.06
    assert torch.allclose(reg(x), torch.tensor(2.06).double())

    # l1 only + regularize_bias
    reg = LaplacianRegularizer(l1=1.0, l2=0.0, regularize_bias=True)
    # heights = [0.5, 0.2, -0.1] -> loss = 0.8
    # bias = [1.0] -> loss = 1.0
    # Total = 1.8
    assert torch.allclose(reg(x), torch.tensor(1.8).double())

    # l2 only + regularize_bias
    reg = LaplacianRegularizer(l1=0.0, l2=1.0, regularize_bias=True)
    # heights L2 = 0.3
    # bias L2 = 1.0
    # Total = 1.3
    assert torch.allclose(reg(x), torch.tensor(1.3).double())

    # l1 + l2 + regularize_bias
    reg = LaplacianRegularizer(l1=1.0, l2=1.0, regularize_bias=True)
    # heights L1 + L2 = 0.8 + 0.3 = 1.1
    # bias L1 + L2 = 1.0 + 1.0 = 2.0
    # Total = 3.1
    assert torch.allclose(reg(x), torch.tensor(3.1).double())

    # l1 + l2 + is_cyclic + regularize_bias
    reg = LaplacianRegularizer(l1=1.0, l2=1.0, is_cyclic=True, regularize_bias=True)
    # heights L1 + L2 (cyclic) = 1.4 + 0.66 = 2.06
    # bias L1 + L2 = 1.0 + 1.0 = 2.0
    # Total = 2.06 + 2.0 = 4.06
    assert torch.allclose(reg(x), torch.tensor(4.06).double())


def test_hessian_regularizer():
    """Tests HessianRegularizer."""
    # x = [bias, h1, h2, h3]
    x = torch.tensor([[1.0], [0.5], [0.2], [-0.1]]).double()

    # l1 only
    reg = HessianRegularizer(l1=1.0, l2=0.0)
    # nonlinearity = [h2-h1, h3-h2] = [0.2-0.5, -0.1-0.2] = [-0.3, -0.3]
    # loss = |-0.3| + |-0.3| = 0.6
    assert torch.allclose(reg(x), torch.tensor(0.6).double())

    # l2 only
    reg = HessianRegularizer(l1=0.0, l2=1.0)
    # loss = (-0.3)^2 + (-0.3)^2 = 0.09 + 0.09 = 0.18
    assert torch.allclose(reg(x), torch.tensor(0.18).double())

    # cyclic only
    reg = HessianRegularizer(l1=1.0, l2=0.0, is_cyclic=True)
    # heights = [0.5, 0.2, -0.1, -0.6, 0.5] (including implicit last and wrap around)
    # nonlinearity = [0.2-0.5, -0.1-0.2, -0.6-(-0.1), 0.5-(-0.6)] = [-0.3, -0.3, -0.5, 1.1]
    # loss = 0.3 + 0.3 + 0.5 + 1.1 = 2.2
    assert torch.allclose(reg(x), torch.tensor(2.2).double())

    # l1 + l2 + cyclic
    reg = HessianRegularizer(l1=1.0, l2=1.0, is_cyclic=True)
    # L1 = 2.2
    # L2 = 0.09 + 0.09 + 0.25 + 1.21 = 1.64
    # Total = 2.2 + 1.64 = 3.84
    assert torch.allclose(reg(x), torch.tensor(3.84).double())


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

    # l2 only
    reg = WrinkleRegularizer(l1=0.0, l2=1.0)
    # loss = 0.0^2 + 0.7^2 = 0.49
    assert torch.allclose(reg(x), torch.tensor(0.49).double())

    # cyclic only
    reg = WrinkleRegularizer(l1=1.0, l2=0.0, is_cyclic=True)
    # heights = [0.5, 0.2, -0.1, 0.3, -0.9, 0.5, 0.2]
    # nonlinearity = [-0.3, -0.3, 0.4, -1.2, 1.4, -0.3]
    # wrinkleness = [0.0, 0.7, -1.6, 2.6, -1.7]
    # loss = 0.0 + 0.7 + 1.6 + 2.6 + 1.7 = 6.6
    assert torch.allclose(reg(x), torch.tensor(6.6).double())

    # l1 + l2 + cyclic
    reg = WrinkleRegularizer(l1=1.0, l2=1.0, is_cyclic=True)
    # L1 = 6.6
    # L2 = 0.0^2 + 0.7^2 + (-1.6)^2 + 2.6^2 + (-1.7)^2 = 0.0 + 0.49 + 2.56 + 6.76 + 2.89 = 12.7
    # Total = 6.6 + 12.7 = 19.3
    assert torch.allclose(reg(x), torch.tensor(19.3).double())


def test_l1_l2_regularizer():
    """Tests L1L2Regularizer."""
    x = torch.tensor([[1.0], [-0.5], [2.0]]).double()

    # l1 only
    reg = L1L2Regularizer(l1=1.0, l2=0.0)
    # loss = |1.0| + |-0.5| + |2.0| = 3.5
    assert torch.allclose(reg(x), torch.tensor(3.5).double())

    # l2 only
    reg = L1L2Regularizer(l1=0.0, l2=1.0)
    # loss = 1.0^2 + (-0.5)^2 + 2.0^2 = 1.0 + 0.25 + 4.0 = 5.25
    assert torch.allclose(reg(x), torch.tensor(5.25).double())

    # l1 + l2
    reg = L1L2Regularizer(l1=1.0, l2=1.0)
    # loss = 3.5 + 5.25 = 8.75
    assert torch.allclose(reg(x), torch.tensor(8.75).double())

    # zero regularization
    reg = L1L2Regularizer(l1=0.0, l2=0.0)
    assert torch.allclose(reg(x), torch.tensor(0.0).double())
