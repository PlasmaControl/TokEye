<p align="center">
  <img src="assets/logo.png" alt="TokEye Logo" width="400">
</p>

# TokEye

[![Python package](https://github.com/PlasmaControl/tokeye/actions/workflows/python-package.yml/badge.svg)](https://github.com/PlasmaControl/tokeye/actions/workflows/python-package.yml)

TokEye is a open-source Python-based application for automatic classification and localization of fluctuating signals.
It is designed to be used in the context of plasma physics, but can be used for any type of fluctuating signal.

Check out [this poster from APS DPP 2025](assets/aps_dpp_2025.pdf) or [this preprint](https://arxiv.org/abs/2602.20317) for more information.

## Example Demonstration
![Example Demonstration](assets/example.gif)

Expected processing time:
- A100: < 0.5 seconds on any size spectrogram after warmup.
- CPU: ~5-10 seconds.

## Verified Datatypes
- DIII-D Fast Magnetics (cite)
- DIII-D CO2 Interferometer (cite)
- DIII-D Electron Cyclotron Emission (cite)
- DIII-D Beam Emission Spectroscopy (cite)

## Evaluation
Recall Scores:
- TJII2021: 0.8254
- DCLDE2011 (Delphinus capensis): 0.7708
- DCLDE2011 (Delphinus delphis): 0.7953

With more data, comes better models. Please contribute to the project!

## Installation

[uv](https://docs.astral.sh/uv/) (recommended)
```bash
git clone git@github.com:PlasmaControl/TokEye.git
cd TokEye
uv sync
```

pip (from PyPI)
```bash
pip install tokeye
```

pip (from source)
```bash
git clone git@github.com:PlasmaControl/TokEye.git
cd TokEye
python3 -m venv .venv
source venv/bin/activate
pip install uv
uv sync
```

## Usage
```bash
python -m TokEye.app
```

This will start a web app on `http://localhost:8888`.

If you are on a remote server, you can use SSH port forwarding to access the web app on your local machine:
```bash
ssh -L 8888:localhost:7860 user@remote_server
```
Then open your web browser and navigate to `http://localhost:8888`.

## Models
Pre-trained models are available at [this link](https://huggingface.co/collections/nc1/tokeye).

(soon to be deprecated) Pre-trained models are available at [this link](https://drive.google.com/drive/folders/1rXllPXB3eWhMvSIlp0CDSFx68lJOQG1u?usp=drive_link).

Copy them into the `models/` directory after downloading them.
- big_mode_v1.pt: Original training regime (window = 1024, hop = 128)
- big_mode_v2.pt: Trained on multiscale (multiwindow, multihop) spectrograms
- big_tf_unet_251210.pt: Trained on multiscale (multiwindow, multihop) spectrograms

Input should be a tensor that has shape (B, 1, H, W) where B, H, and W can vary
Output will be a tensor of shape (B, 2, H, W)

Best performance when spectrograms are oriented so that when they are plotted with matplotlib, the lowest frequency bin is oriented with the bottom when `origin='lower'`. Spectrograms should be standardized (mean = 0, std = 1). If baseline activity is very strong, clipping the input may help, but is generally not needed.

The first channel of the output will return preferential measurements of coherent activity (useful for most tasks)
THe second channel of the output will return preferential measurements of transient activity

## Data
Right now, keep all data as 1d numpy float arrays. No need to normalize or preprocess them.
Copy them into the `data/` directory.

## Citation
If you use this code in your research, please cite:
```bibtex
@article{chen_TokEye_2026,
  title={TokEye: Fast Signal Extraction for Fluctuating Time Series via Offline Self-Supervised Learning From Fusion Diagnostics to Bioacoustics},
  author={Chen, Nathaniel},
  year={2026},
  publisher={ArXiv},
  doi={10.48550/arXiv.2602.20317},
  url={https://www.arxiv.org/abs/2602.20317}
}
```

## Contact
Please check back for updates or reach out to Nathaniel Chen at nathaniel@princeton.edu.
