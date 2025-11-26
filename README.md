<p align="center">
  <img src="assets/logo.png" alt="TokEye Logo" width="400">
</p>

# TokEye

TokEye is a open-source Python-based application for automatic classification and localization of fluctuating signals. It is designed to be used in the context of plasma physics, but can be used for any type of fluctuating signal.

## Example Demonstration
![Example Demonstration](assets/example.gif)

## Installation
Installation is meant to be as simple as possible.

containerized installation (recommended)
```bash
docker run -it --rm -v $(pwd):/app -w /app ghcr.io/plasma-control/tokeye:latest bash
```

[uv](https://docs.astral.sh/uv/)
```bash
uv add tokeye
```

pip
```bash
pip install tokeye
```

## Usage
```bash
tokeye
```

## Training
```bash
tokeye train
```

## Evaluation
```bash
tokeye evaluate
```

## Citation
If you use this code in your research, please cite:
```bibtex
@article{tokeye,
  title={TokEye: A Time-series to Spectrogram Segmentation Application for Plasma Physics Analysis},
  author={TokEye},
  journal={arXiv preprint arXiv:2511.14898},
  year={2025}
}
```