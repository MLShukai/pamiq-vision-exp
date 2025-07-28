# Static Metrics Usage Guide

## Overview

The static metrics tool evaluates trained models on datasets to compute performance metrics.
This is useful for post-training analysis and validation.

## Quick Start

```bash
# Basic usage - evaluate a state on CIFAR-100
python static_metrics.py saved_state=outputs/2025-01-01/12-34-56/states/state.pt

# Use different dataset
python static_metrics.py saved_state=/path/to/state.pt dataset=cifar10

# Add tags for Aim UI organization
python static_metrics.py saved_state=/path/to/state.pt tags=[validation,final]
```

## Required Parameter

- `saved_state`: Path to the saved state file (usually ends with `.pt`)

## Optional Parameters

- `dataset`: Which dataset to use (default: cifar100)
- `metrics`: Which metrics to compute (default: jepa)
- `tags`: List of tags for Aim UI organization
