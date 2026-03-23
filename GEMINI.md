# PyTorch Lattice (GAMLSS)

A PyTorch implementation of lattice modeling techniques, focusing on interpretability and shape constraints (e.g., monotonicity). This project provides "glassbox" models that are more transparent than typical deep neural networks.

## Project Overview

- **Purpose**: To provide a library for building and training calibrated models (linear or lattice-based) with domain-knowledge constraints. This library is a partial port of TensorFlow Lattice to PyTorch.
- **GAMLSS Goal**: This fork is specifically used to help create a **GAMLSS (Generalized Additive Models for Location, Scale, and Shape)** implementation in Python. The objective is a GAMLSS model where each term is a function (either a PWL/numeric calibration or a categorical calibration) of an input variable.

## Porting Reference & Mappings

To aid in porting, original TensorFlow Lattice (TFL) source files are stored in the `reference/` directory.

### Concept Mappings

| TensorFlow Lattice (TFL) | PyTorch Lattice (PTL) | Description |
| :--- | :--- | :--- |
| `pwl_calibration_layer.py` | `numerical_calibrator.py` | PWL/Numeric calibration layer. |
| `categorical_calibration_layer.py` | `categorical_calibrator.py` | Categorical calibration layer. |

### Key Reference Files
- **PWL/Numeric Calibration**: `reference/pwl_calibration_layer.py`
- **Categorical Calibration**: `reference/categorical_calibration_layer.py`
- **Tests**: `reference/pwl_calibration_test.py`, `reference/categorical_calibration_test.py`

## Core Architecture
  - `FeatureConfig`: Configuration for individual features (keypoints, monotonicity, etc.).
  - `Layers`: Custom PyTorch layers for numerical and categorical calibration.
  - `Models`: High-level GAMLSS-focused model architectures (in development).
- **Main Technologies**: PyTorch, Pandas, NumPy, Matplotlib, Pydantic, uv.

## Building and Running

This project uses **uv** for dependency management.

### Environment Management
- **uv**: All dependencies and virtual environments are managed by `uv`. Use `uv sync` to install and `uv run <command>` to execute within the environment.
- **Python Version**: Target Python version is 3.12.

### Setup
```powershell
# Install dependencies
uv sync

# Run tests or commands within environment
uv run pytest
```

### PowerShell Support
When using PowerShell 5.1 (default on Windows), do not use the `&&` operator to chain commands. Instead, use `;` as a statement separator:
```powershell
uv sync; uv run pytest
```

## Source Control and Branches

- **main**: The primary development branch.
- **ci/cd-config**: Dedicated to "housekeeping" tasks. This includes CI/CD workflows, project-wide system configurations, and general documentation. Prefer pushing these changes here—even if discovered while working on another branch—unless doing so adds significant complexity or risk.
- **gh-pages**: Automatically managed by GitHub Actions (`docs.yml`) for hosting documentation. Do not manually merge into or modify this branch.
- **Feature Branches**: Use descriptive names (e.g., `feat/`, `fix/`). **Feature branches must include both the implementation code and the corresponding tests required for verification.**

### Key Commands
- **Testing**: `pytest`
- **Linting**: `ruff check .`
- **Formatting**: `ruff format .`
- **Type Checking**: `mypy .`
- **Documentation**: `mkdocs serve` or `mkdocs build`

## Development Conventions

- **Code Style**: Adheres to `ruff`'s default rules (configured in `pyproject.toml`). Uses double quotes for strings and 88-character line length.
- **Type Safety**: Uses `mypy` for static type checking. Pydantic is used for configuration validation (`FeatureConfig`, `ModelConfigs`).
- **Testing**: Uses `pytest`. Tests are located in the `tests/` directory and follow a structure mirroring the source code.
- **Documentation**: Uses `mkdocs` with the `material` theme and `mkdocstrings` for API documentation.
- **Constraints**: Constraints (like monotonicity) are applied after each optimization step using `model.apply_constraints()`.
- **Jupyter Notebooks**: Only read `.ipynb` files when strictly necessary. Convert the notebook to a temporary text file (e.g., using `jupyter nbconvert --to python`) before reading, lowering context bloat, and delete the temporary file immediately after.

## Directory Structure

- `pytorch_lattice/`: Core library code.
  - `layers/`: Implementation of calibration layers.
  - `models/`: High-level model architectures.
  - `utils/`: Data processing and model utility functions.
- `tests/`: Comprehensive test suite.
- `examples/`: Example scripts demonstrating library usage.
- `notebooks/`: Jupyter notebooks for data exploration and research.
- `docs/`: Source files for documentation.
