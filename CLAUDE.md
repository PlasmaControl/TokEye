# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TokEye is a Python application for automatic classification and localization of fluctuating signals in spectrograms, primarily for plasma physics (tokamak diagnostics). It uses deep learning (PyTorch) with a transformer-based U-Net architecture and provides a Gradio web interface for interactive analysis and annotation.

## Commands

```bash
# Install dependencies (requires uv)
uv sync            # core deps only
uv sync --dev      # include dev tools (pytest, ruff, etc.)
uv sync --group train  # include training deps (lightning, h5py, etc.)

# Run the web app
python -m TokEye.app              # starts on localhost:7860
python -m TokEye.app --port 8888  # custom port
python -m TokEye.app --share      # public Gradio link

# Lint
uv run ruff check .

# Test
uv run pytest                # all tests
uv run pytest tests/test_basic.py  # single file
uv run pytest -k "test_name"       # single test by name
```

## Architecture

Source code lives in `src/tokeye/` (installed as `tokeye` package via `uv_build`).

### Models (`models/`)
Three model families, each with a `model_*.py` and `config_*.py`:
- **big_tf_unet** — primary transformer U-Net for spectrogram segmentation
- **ae_tf_maskrcnn** — alternative Mask R-CNN approach
- **ae_tf_boxrcnn** — alternative Box R-CNN approach

Shared building blocks in `models/modules/`: `unet.py` (base U-Net), `nn.py` (layers), `bsn.py` (boundary segmentation network).

Model I/O: input `(B, 1, H, W)` → output `(B, 2, H, W)` where channel 0 = coherent activity, channel 1 = transient activity.

### App (`app/`)
Gradio web interface launched via `python -m TokEye.app`. Three tabs:
- **Analyze** (`app/analyze/`) — load signals, compute STFT spectrograms, run inference
- **Annotate** (`app/tabs/annotate.py`) — manual labeling interface
- **Utilities** (`app/tabs/utilities.py`) — miscellaneous tools

Inference pipeline: `app/processing/` handles model loading, tiled inference for large spectrograms, and post-processing.

### Training (`training/`)
Multi-step data pipelines (step_0 through step_7) for preparing training data from raw signals. Two regimes: `big_tf_unet/` (original) and `big_tf_unet_multiscale/` (enhanced). Uses PyTorch Lightning.

### Extra (`extra/`)
Domain-specific utilities: DIII-D tokamak helpers (`extra/D3D/`), evaluation tools for DCLDE marine mammal dataset (`extra/eval/silbidopy/`).

## Code Style

- Ruff for linting (config in `ruff.toml`): line length 88, pathlib over os.path (`PTH`), `from __future__ import annotations` enforced (`FA`)
- Python >= 3.13 required
- Pre-trained model weights (`.pt`, `.ckpt`) and data files (`.npy`) are gitignored — stored in `model/` and `data/` directories locally, hosted on Hugging Face

## CI

GitHub Actions runs on push/PR to main: `uv run ruff check .` then `uv run pytest` across Python 3.10–3.14.
