# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PAMIQ Vision Experiments  for self-supervised visual representation learning. The project extends the PAMIQ framework's agent-environment paradigm to vision tasks, where an agent collects image observations and learns representations without labels.

## Development Commands

```bash
# Environment setup
make venv              # Create venv and install all dependencies
uv sync --all-extras   # Sync dependencies (after pyproject.toml changes)

# Development workflow
make format            # Run pre-commit hooks (Ruff formatting)
make test              # Run pytest with coverage
make type              # Run Pyright type checking
make run               # Complete workflow: format → test → type

# Docker development (supports GPU)
ENABLE_GPU=true make docker-up    # Start containers with GPU
make docker-attach                # Attach to dev container
make docker-down                  # Stop containers
```

## Architecture Overview

### PAMIQ Integration Pattern

The codebase follows PAMIQ's agent-environment-trainer paradigm:

```
ImageEnvironment → ImageCollectingAgent → Buffer → JEPATrainer → Models
```

- **Environment** generates observations (images)
- **Agent** collects observations into named buffers
- **Trainer** samples from buffers to train models
- **Models** are managed by trainer and updated iteratively

### JEPA Components

The JEPA implementation consists of interconnected models:

1. **Context Encoder** (`src/exp/models/jepa/context_encoder.py`): Processes masked images
2. **Target Encoder** (`src/exp/models/jepa/target_encoder.py`): Processes full images (EMA-updated)
3. **Predictor** (`src/exp/models/jepa/predictor.py`): Predicts target representations from context
4. **LightWeightDecoder** (`src/exp/models/jepa/decoder.py`): Optional image reconstruction

Key architectural pattern: All models use shared components from `src/exp/models/components/` (transformers, patchifiers, embeddings) enabling consistent patch-based processing.

### Configuration System

Uses Hydra for hierarchical configuration:

- Base configs in `configs/` define component defaults
- Override with CLI: `python -m exp +experiment=jepa`
- Access in code: `@hydra.main(config_path="configs", config_name="base")`

### Type System

Project uses custom type aliases for clarity:

```python
from exp.types import size_2d  # Tuple[int, int] for image/patch sizes
```

String enums define constants:

```python
class BufferNames(StrEnum):
    OBSERVATION = "observation"

class ModelNames(StrEnum):
    CONTEXT_ENCODER = "context_encoder"
```

## Testing Strategy

```bash
# Run all tests
make test

# Run specific test file
pytest tests/exp/models/jepa/test_encoder.py

# Run with specific markers
pytest -m "not slow"

# Debug test
pytest tests/exp/test_example.py::test_name -vvs
```

Tests use parametrized fixtures for different configurations. When adding features, mirror the source structure in `tests/` with `test_` prefix.

## Key Patterns

1. **Device Management**: Models automatically handle device placement via `model.to(device)` in constructors
2. **Batch Processing**: All models expect batched inputs `(B, C, H, W)` for images
3. **Modular Components**: Reuse transformers, embeddings, and patchifiers across models

## Docker GPU Development

The Docker environment includes NVIDIA GPU support and Claude Code CLI:

```bash
# Ensure NVIDIA Docker runtime is installed
# Set ENABLE_GPU=true for GPU access
ENABLE_GPU=true make docker-up
```

Inside container, all development tools are pre-configured with mounted workspace.
