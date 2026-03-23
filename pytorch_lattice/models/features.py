"""Feature objects for use in models.

To construct a model, create the feature configurations and pass them
in to the model constructor.
"""

import logging
from typing import Optional, Union

import numpy as np

from ..enums import InputKeypointsInit, Monotonicity


class NumericalFeature:
    """Feature configuration for numerical features.

    Attributes:
        All: `__init__` arguments.
        input_keypoints: The input keypoints used for this feature's calibrator. These
            keypoints will be initialized using the given `data` under the desired
            `input_keypoints_init` scheme.
    """

    def __init__(
        self,
        feature_name: str,
        data: np.ndarray,
        num_keypoints: int = 5,
        input_keypoints_init: InputKeypointsInit = InputKeypointsInit.QUANTILES,
        missing_input_value: Optional[float] = None,
        monotonicity: Optional[Monotonicity] = None,
        is_cyclic: bool = False,
        projection_iterations: int = 8,
        laplacian_l1: float = 0.0,
        laplacian_l2: float = 0.0,
        hessian_l1: float = 0.0,
        hessian_l2: float = 0.0,
        wrinkle_l1: float = 0.0,
        wrinkle_l2: float = 0.0,
    ) -> None:
        """Initializes a `NumericalFeatureConfig` instance.

        Args:
            feature_name: The name of the feature. This should match the header for the
                column in the dataset representing this feature.
            data: Numpy array of float-valued data used for calculating keypoint inputs
                and initializing keypoint outputs.
            num_keypoints: The number of keypoints used by the underlying piece-wise
                linear function of a NumericalCalibrator. There will be
                `num_keypoints - 1` total segments.
            input_keypoints_init: The scheme to use for initializing the input
                keypoints. See `InputKeypointsInit` for more details.
            missing_input_value: If provided, this feature's calibrator will learn to
                map all instances of this missing input value to a learned output value.
            monotonicity: Monotonicity constraint for this feature, if any.
            is_cyclic: Whether the output for the last keypoint should be identical to
                the output for the first keypoint.
            projection_iterations: Number of times to run Dykstra's projection
                algorithm when applying constraints.
            laplacian_l1: L1 regularization amount for the Laplacian regularizer.
            laplacian_l2: L2 regularization amount for the Laplacian regularizer.
            hessian_l1: L1 regularization amount for the Hessian regularizer.
            hessian_l2: L2 regularization amount for the Hessian regularizer.
            wrinkle_l1: L1 regularization amount for the Wrinkle regularizer.
            wrinkle_l2: L2 regularization amount for the Wrinkle regularizer.

        Raises:
            ValueError: If `data` contains NaN values.
            ValueError: If `input_keypoints_init` is invalid.
        """
        self.feature_name = feature_name

        if np.isnan(data).any():
            raise ValueError("Data contains NaN values.")

        self.data = data
        self.num_keypoints = num_keypoints
        self.input_keypoints_init = input_keypoints_init
        self.missing_input_value = missing_input_value
        self.monotonicity = monotonicity
        self.is_cyclic = is_cyclic
        self.projection_iterations = projection_iterations
        self.laplacian_l1 = laplacian_l1
        self.laplacian_l2 = laplacian_l2
        self.hessian_l1 = hessian_l1
        self.hessian_l2 = hessian_l2
        self.wrinkle_l1 = wrinkle_l1
        self.wrinkle_l2 = wrinkle_l2

        sorted_unique_values = np.unique(data)

        if input_keypoints_init == InputKeypointsInit.QUANTILES:
            if sorted_unique_values.size < num_keypoints:
                logging.info(
                    "Observed fewer unique values for feature %s than %d desired "
                    "keypoints. Using the observed %d unique values as keypoints.",
                    feature_name,
                    num_keypoints,
                    sorted_unique_values.size,
                )
                self.input_keypoints = sorted_unique_values
            else:
                quantiles = np.linspace(0.0, 1.0, num=num_keypoints)
                self.input_keypoints = np.quantile(
                    sorted_unique_values, quantiles, method="nearest"
                )
        elif input_keypoints_init == InputKeypointsInit.UNIFORM:
            self.input_keypoints = np.linspace(
                sorted_unique_values[0], sorted_unique_values[-1], num=num_keypoints
            )
        else:
            raise ValueError(f"Unknown input keypoints init: {input_keypoints_init}")


class CategoricalFeature:
    """Feature configuration for categorical features.

    Attributes:
        All: `__init__` arguments.
        category_indices: A dictionary mapping string categories to their index.
        monotonicity_index_pairs: A conversion of `monotonicity_pairs` from string
            categories to category indices. Only available if `monotonicity_pairs` are
            provided.
    """

    def __init__(
        self,
        feature_name: str,
        categories: Union[list[int], list[str]],
        missing_input_value: Optional[float] = None,
        monotonicity_pairs: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        """Initializes a `CategoricalFeatureConfig` instance.

        Args:
            feature_name: The name of the feature. This should match the header for the
                column in the dataset representing this feature.
            categories: The categories that should be used for this feature. Any
                categories not contained will be considered missing or unknown. If you
                expect to have such missing categories, make sure to
            missing_input_value: If provided, this feature's calibrator will learn to
                map all instances of this missing input value to a learned output value.
            monotonicity_pairs: List of pairs of categories `(category_a, category_b)`
                indicating that the calibrator output for `category_b` should be greater
                than or equal to that of `category_a`.
        """
        self.feature_name = feature_name
        self.categories = categories
        self.missing_input_value = missing_input_value
        self.monotonicity_pairs = monotonicity_pairs

        self.category_indices = {category: i for i, category in enumerate(categories)}
        self.monotonicity_index_pairs = [
            (self.category_indices[a], self.category_indices[b])
            for a, b in monotonicity_pairs or []
        ]
