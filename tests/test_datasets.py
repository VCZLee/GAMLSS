"""Tests for dataset loading utilities."""

from unittest.mock import patch

import numpy as np
import pandas as pd
from pytorch_lattice.datasets import adult, heart


@patch("pandas.read_csv")
def test_heart(mock_read_csv):
    """Tests the heart dataset loading function."""
    # We use a mocked toy dataset because the current URLs in the codebase
    # are returning 404. This allows us to test the logic (like popping the target)
    # without needing a working internet connection or the real files.
    mock_df = pd.DataFrame(
        {"age": [63, 67], "sex": [1, 1], "cp": [3, 4], "target": [0, 1]}
    )
    mock_read_csv.return_value = mock_df

    X, y = heart()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert "target" not in X.columns
    assert len(X) == 2
    assert np.array_equal(y, np.array([0, 1]))


@patch("pandas.read_csv")
def test_adult(mock_read_csv):
    """Tests the adult dataset loading function."""
    mock_df = pd.DataFrame(
        {
            "age": [39, 50],
            "workclass": ["State-gov", "Self-emp-not-inc"],
            "label": [0, 1],
        }
    )
    mock_read_csv.return_value = mock_df

    X, y = adult()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert "label" not in X.columns
    assert len(X) == 2
    assert np.array_equal(y, np.array([0, 1]))
