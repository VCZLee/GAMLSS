"""Tests for plotting utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytorch_lattice.plots import calibrator, linear_coefficients


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.plot")
@patch("matplotlib.pyplot.bar")
@patch("matplotlib.pyplot.title")
def test_calibrator_numerical(mock_title, mock_bar, mock_plot, mock_show):
    """Tests calibrator plot for a numerical feature."""
    mock_model = MagicMock()
    mock_calibrator = MagicMock()

    # Setup mock data
    mock_calibrator.keypoints_inputs.return_value.numpy.return_value = np.array([0, 1])
    mock_calibrator.keypoints_outputs.return_value.numpy.return_value = np.array([0, 1])
    mock_model.calibrators = {"feature_a": mock_calibrator}

    # We mock isinstance to force the numerical (plt.plot) path
    with patch("pytorch_lattice.plots.isinstance", return_value=False):
        calibrator(mock_model, "feature_a")

    mock_plot.assert_called_once()
    mock_show.assert_called_once()
    mock_bar.assert_not_called()
    mock_title.assert_called_with("Calibrator: feature_a")


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.bar")
@patch("matplotlib.pyplot.xticks")
def test_calibrator_categorical(mock_xticks, mock_bar, mock_show):
    """Tests calibrator plot for a categorical feature."""
    mock_model = MagicMock()
    mock_calibrator = MagicMock()
    mock_feature = MagicMock()

    # Setup mock data
    mock_calibrator.keypoints_inputs.return_value.numpy.return_value = np.array([0, 1])
    mock_calibrator.keypoints_outputs.return_value.numpy.return_value = np.array(
        [0.1, 0.2]
    )
    mock_model.calibrators = {"cat_feature": mock_calibrator}

    mock_feature.feature_name = "cat_feature"
    mock_feature.categories = ["A", "B"]
    mock_model.features = [mock_feature]

    # Mock isinstance to force the categorical (plt.bar) path
    # We need to handle two different isinstance checks in the function
    from pytorch_lattice.layers import CategoricalCalibrator
    from pytorch_lattice.models.features import CategoricalFeature

    def side_effect(obj, cls):
        if cls == CategoricalCalibrator:
            return True
        if cls == CategoricalFeature:
            return True
        return False

    with patch("pytorch_lattice.plots.isinstance", side_effect=side_effect):
        calibrator(mock_model, "cat_feature")

    mock_bar.assert_called_once()
    mock_show.assert_called_once()
    mock_xticks.assert_called_with(rotation=45)


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.bar")
def test_linear_coefficients(mock_bar, mock_show):
    """Tests linear_coefficients plot."""
    mock_model = MagicMock()
    mock_feature = MagicMock()
    mock_feature.feature_name = "feature_a"
    mock_model.features = [mock_feature]
    mock_model.linear.kernel.detach.return_value.numpy.return_value = np.array([[0.5]])
    mock_model.use_bias = True
    mock_model.linear.bias.detach.return_value.numpy.return_value = np.array([0.1])

    # Ensure it passes the isinstance(model, CalibratedLinear) check

    with patch("pytorch_lattice.plots.isinstance", return_value=True):
        linear_coefficients(mock_model)

    mock_bar.assert_called_once()
    mock_show.assert_called_once()


def test_calibrator_invalid_feature():
    """Tests that calibrator raises ValueError for unknown feature."""
    mock_model = MagicMock()
    mock_model.calibrators = {}
    with pytest.raises(ValueError, match="Feature unknown not found"):
        calibrator(mock_model, "unknown")


def test_linear_coefficients_invalid_model():
    """Tests that linear_coefficients raises ValueError for wrong model type."""
    mock_model = MagicMock()
    with patch("pytorch_lattice.plots.isinstance", return_value=False):
        with pytest.raises(ValueError, match="Model must be a `CalibratedLinear`"):
            linear_coefficients(mock_model)
