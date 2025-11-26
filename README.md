<p align="center">
  <img src="assets/logo.png" alt="TokEye Logo" width="400">
</p>

# TokEye

(This repository is a work in progress. Please check back for updates or reach out to  nathaniel@princeton.edu.)

TokEye is a open-source Python-based application for automatic classification and localization of fluctuating signals. It is designed to be used in the context of plasma physics, but can be used for any type of fluctuating signal.

Check out [this poster from APS DPP 2025](assets/aps_dpp_2025.pdf) for more information.

## Example Demonstration
![Example Demonstration](assets/example.gif)

Expected processing time:
- A100: < 0.5 seconds on any size spectrogram
- CPU: not yet tested.

## Verified Datatypes
- DIII-D Fast Magnetics (cite)
- DIII-D CO2 Interferometer (cite)
- DIII-D Electron Cyclotron Emission (cite)
- DIII-D Beam Emission Spectroscopy (cite)

With more data, comes better models. Please contribute to the project!

## Installation
Installation is meant to be as simple as possible and work across all devices.

containerized installation (recommended) -- not yet available
```bash
docker run -it --rm -v $(pwd):/app -w /app ghcr.io/plasma-control/tokeye:latest bash
```

[uv](https://docs.astral.sh/uv/) -- not yet available
```bash
uv add tokeye
```

pip -- not yet available
```bash
pip install tokeye
```

pip (from source)
```bash
pip install git+https://github.com/plasma-control/tokeye.git
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
@article{NaN,
  title={Paper not yet published},
  author={Nathaniel Chen},
  year={2025}
}
```