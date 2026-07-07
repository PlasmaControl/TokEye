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
tokeye app                   # starts on localhost:7860 (or: python -m tokeye.app)
tokeye app --port 8888       # custom port
tokeye app --share           # public Gradio link

# Headless CLI (no gradio import; safe for HPC/CI)
tokeye run "shots/*.npy" --output-dir results  # batch inference
tokeye download big_tf_unet                    # pre-fetch weights, print cache path
tokeye example                                 # write a synthetic demo signal

# Mode-analysis suite
tokeye modespec modes.yaml       # classic Mirnov mode-number analysis (vendored pymodespec)
tokeye elmspec "shots/*.npy"     # ELM events from the transient channel
tokeye alfvenspec "shots/*.npy"  # Alfvén-eigenmode boxes/masks (ae_tf_maskrcnn)
tokeye eigspec                   # interactive modal ID / SSI (vendored eigspec port)
tokeye modesearch                # mode database — design stage, prints the plan

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
- **big_tf_unet** — primary transformer U-Net for spectrogram segmentation (HF: `nc1/big_tf_unet`)
- **ae_tf_maskrcnn** — Mask R-CNN instance detector, used by `tokeye alfvenspec` (HF: `nc1/ae_tf_maskrcnn`)
- **ae_tf_boxrcnn** — alternative Box R-CNN approach (not registered, no weights)

`hub.MODEL_REGISTRY` order is load-bearing: `_build_from_state_dict` probes specs in insertion order, so `big_tf_unet` must stay first. `ModelSpec.repo_id` overrides `DEFAULT_REPO_ID` per model.

Shared building blocks in `models/modules/`: `unet.py` (base U-Net), `nn.py` (layers), `bsn.py` (boundary segmentation network).

Model I/O: input `(B, 1, H, W)` → output `(B, 2, H, W)` where channel 0 = coherent activity, channel 1 = transient activity.

### App (`app/`)
Gradio web interface launched via `tokeye app` (console script) or `python -m tokeye.app`. Three tabs:
- **Analyze** (`app/analyze/`) — load signals, compute STFT spectrograms, run inference
- **Annotate** (`app/tabs/annotate.py`) — manual labeling interface
- **Utilities** (`app/tabs/utilities.py`) — miscellaneous tools

Shared core modules (used by both the app and the `tokeye` CLI) live directly under `src/tokeye/`: `hub.py` (model registry + Hugging Face auto-download), `transforms.py` (STFT), `inference.py` (model inference, U-Net contract), `api.py` (the `TokEye` class — public Python API, lazily exported from the package root), `batch.py` (headless batch runner), `examples.py` (synthetic demo signal), `cli/` (the `tokeye` console entry point — one module per subcommand, heavy imports deferred into `_handle` functions).

### Mode-analysis suite
- `modespec/classic/` — **vendored** pymodespec (classic Mirnov mode-number analysis); `modespec/deep/` reserves the next-gen single-chord engine (sibling `integratedmode` project). Vendored code policy: minimal-touch, style rules relaxed in `ruff.toml`, every local change listed in the directory's `PROVENANCE.md`.
- `elmspec/` — ELM event extraction from the transient channel (`events.py` is pure numpy, model plumbing in the CLI handler).
- `alfvenspec/` — R-CNN detection wrapper (`inference.py`; list-of-images contract, windowed processing for wide spectrograms).
- `eigspec/` — **vendored** eigspec MATLAB-toolbox port (modal ID, SSI, random projection); sklearn-dependent clustering behind the `eigspec` extra.
- `modesearch/` — design-stage scaffold only (mode database vision).
- Suite roadmap and future ideas: `docs/ROADMAP.md`.

### Training (`training/`)
Multi-step data pipelines (step_0 through step_7) for preparing training data from raw signals. Two regimes: `big_tf_unet/` (original) and `big_tf_unet_multiscale/` (enhanced). Uses PyTorch Lightning.

### Extra (`extra/`)
Domain-specific utilities: DIII-D tokamak helpers (`extra/D3D/`), evaluation tools for DCLDE marine mammal dataset (`extra/eval/silbidopy/`).

## Code Style

- Ruff for linting (config in `ruff.toml`): line length 88, pathlib over os.path (`PTH`), `from __future__ import annotations` enforced (`FA`)
- Python >= 3.13 required
- Pre-trained model weights (`.pt`, `.ckpt`) and data files (`.npy`) are gitignored — stored in `model/` and `data/` directories locally, hosted on Hugging Face

## CI

GitHub Actions runs on push/PR to main: `uv run ruff check .` then `uv run pytest` across Python 3.13–3.14.
