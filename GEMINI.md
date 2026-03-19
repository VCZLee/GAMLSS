# PyTorch Lattice (GAMLSS)

A PyTorch implementation of lattice modeling techniques, focusing on interpretability and shape constraints (e.g., monotonicity). This project provides "glassbox" models that are more transparent than typical deep neural networks.

## Project Overview

- **Purpose**: To provide a library for building and training calibrated models (linear or lattice-based) with domain-knowledge constraints.
- **Core Architecture**:
  - `Classifier`: High-level API for training models on tabular data.
  - `FeatureConfig`: Configuration for individual features (keypoints, monotonicity, etc.).
  - `Layers`: Custom PyTorch layers for numerical/categorical calibration and lattice interpolation.
  - `Models`: `CalibratedLinear` and `CalibratedLattice` which compose calibrators and a linear/lattice layer.
- **Main Technologies**: PyTorch, Pandas, NumPy, Matplotlib, Pydantic, uv.

## Building and Running

This project uses **uv** for dependency management.

### Setup
```powershell
# Install dependencies
uv sync

# Run tests or commands within environment
uv run pytest
```

### Key Commands
- **Testing**: `pytest`
- **Linting**: `ruff check .`
- **Formatting**: `ruff format .`
- **Type Checking**: `mypy .`
- **Documentation**: `mkdocs serve`

## Development Conventions

- **Code Style**: Adheres to `ruff`'s default rules (configured in `pyproject.toml`). Uses double quotes for strings and 88-character line length.
- **Type Safety**: Uses `mypy` for static type checking. Pydantic is used for configuration validation (`FeatureConfig`, `ModelConfigs`).
- **Testing**: Uses `pytest`. Tests are located in the `tests/` directory and follow a structure mirroring the source code.
- **Documentation**: Uses `mkdocs` with the `material` theme and `mkdocstrings` for API documentation.
- **Constraints**: Constraints (like monotonicity) are applied after each optimization step using `model.apply_constraints()`.

## Directory Structure

- `pytorch_lattice/`: Core library code.
  - `layers/`: Implementation of calibration and lattice layers.
  - `models/`: High-level model architectures.
  - `utils/`: Data processing and model utility functions.
- `tests/`: Comprehensive test suite.
- `examples/`: Example scripts demonstrating library usage.
- `docs/`: Source files for documentation.
