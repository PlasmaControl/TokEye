# TokEye Ablation Pipeline + TJ-II Uncertainty — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully-automated `big_tf_unet_ablation` pipeline that runs the self-supervised recipe end-to-end under 4 leave-one-out toggles (`full`/`mag`/`nobaseline`/`nodenoise`) and a TJ-II ablation eval reporting across-fold + across-image uncertainty on recall/F1/IoU.

**Architecture:** Reuse the proven automation utilities from `big_tf_unet_multiscale` (auto-params, HDF5 I/O, config loader, losses, augmentations — copied in for a self-contained paper artifact), but reconstruct the **paper-correct dual-mask dataflow** from the original `big_tf_unet` (coherent mask = threshold(denoised step_3b); transient mask = threshold(step_2b_baseline); `step_6a` stacks them into the 2-channel target). Single STFT scale (1024/128). A new model-based `step_2f` window filter caps windows/shot at 25 and drops empties. `step_6d` trains 5 CV folds per variant for uncertainty.

**Tech Stack:** Python 3.13, PyTorch + Lightning, h5py, OmegaConf, pybaselines, scikit-image, tifffile, pytest, SLURM (A100 train / V100 head-node label-gen+eval).

**Reference spec:** `docs/superpowers/specs/2026-05-28-tokeye-ablation-design.md`

**Canonical dataflow (verified — get this right or the ablation is meaningless):**
```
SHARED once:  0a 0b (done) → 0c (window+precap) → 2a (complex STFT) → 2f (model filter, ≤25/shot)
PER VARIANT:  2b (baseline toggle) ─┬─ coherent:  [3a 3b denoise toggle/repr] → 4a → step_4a_threshold.h5
                                    └─ transient: step_2b_baseline.h5         → 4a → step_4a_threshold_baseline.h5
              5a (combine per-shot: img + coherent + transient) → 6a (TIF dual-mask) → 6b refiner → 6c → 6d (5-fold)
EVAL:         TJII2021_ablation.py over variant×fold → mean±CI across folds + image bootstrap
```

**Naming reference (verified):**
- `UNet(in_channels, out_channels, num_layers, first_layer_size)` in `tokeye.models.modules.unet`.
- Filter model: `BigTFUNetModel(BigTFUNetConfig(in_channels=1, out_channels=2, num_layers=5, first_layer_size=32, dropout_rate=0.0))`, weights via `remap_legacy_state_dict` (copy from `scripts/eval/TJII2021.py:43-68`); weights at `/scratch/gpfs/nc1514/aemodes/model/big_mode_v1-5_weights.pt` (fallback `model/big_tf_unet_251210_weights.pt`).
- `hdf5_io`: `create_step_file(path, metadata)`, `write_sample(h5, idx, arr)`, `read_sample(path, idx)`, `iter_samples(path)→(idx,arr)`, `get_sample_count(path)`.
- Denoiser data shape `(C, F, T, 2)` = `[real, imag]`. `step_4a` already handles `ndim==3` (complex, RMS over last axis) vs `ndim==2` (magnitude) — so the `mag`/`nodenoise` magnitude inputs work unchanged.

---

## Phase 0 — Scaffold

### Task 0: Create package skeleton + copy reusable utils

**Files:**
- Create: `src/tokeye/training/big_tf_unet_ablation/__init__.py` (empty)
- Create: `src/tokeye/training/big_tf_unet_ablation/README.md`
- Create dir: `src/tokeye/training/big_tf_unet_ablation/utils/` (copy of multiscale utils)
- Create dir: `src/tokeye/training/big_tf_unet_ablation/config/`

- [ ] **Step 1: Copy package + utils from multiscale, then prune**

```bash
cd /scratch/gpfs/nc1514/tokeye
SRC=src/tokeye/training/big_tf_unet_multiscale
DST=src/tokeye/training/big_tf_unet_ablation
mkdir -p $DST
# copy the generic, automation utilities verbatim (self-contained artifact)
cp -r $SRC/utils $DST/utils
cp -r $SRC/preprocess $DST/preprocess
# copy step files we will adapt
cp $SRC/step_0a_extract_faithdata.py $DST/
cp $SRC/step_0b_filter_faithdata.py $DST/
cp $SRC/step_0c_convert_faithdata.py $DST/
cp $SRC/step_2a_make_spectrogram.py $DST/
cp $SRC/step_2b_filter_spectrogram.py $DST/
cp $SRC/step_3a_correlation_analysis.py $DST/
cp $SRC/step_3b_extract_correlation.py $DST/
cp $SRC/step_4a_threshold.py $DST/
cp $SRC/step_5a_combine_spectrogram.py $DST/
cp $SRC/step_6a_convert_tif.py $DST/
cp $SRC/step_6b_refiner.py $DST/
cp $SRC/step_6c_convert_predictions.py $DST/
cp $SRC/step_6d_final.py $DST/
touch $DST/__init__.py
echo "ablation pipeline (paper)" > $DST/README.md
# fix relative imports: utils/configuration.py points _PIPELINE_YAML at ../config/pipeline.yaml;
# we use ablation.yaml — update in Task 1.
ls $DST
```

- [ ] **Step 2: Verify imports resolve**

Run: `cd /scratch/gpfs/nc1514/tokeye && .venv/bin/python -c "import tokeye.training.big_tf_unet_ablation.step_2a_make_spectrogram as m; print('ok', m.default_settings['nfft'])"`
Expected: `ok 1024`

- [ ] **Step 3: Commit**

```bash
git add src/tokeye/training/big_tf_unet_ablation
git commit -m "scaffold big_tf_unet_ablation from multiscale utils"
```

---

## Phase 1 — Config & variant matrix

### Task 1: Ablation config YAML + variant dataclass + matrix

**Files:**
- Create: `src/tokeye/training/big_tf_unet_ablation/config/ablation.yaml`
- Create: `src/tokeye/training/big_tf_unet_ablation/config/variants.py`
- Create: `src/tokeye/training/big_tf_unet_ablation/ablation_matrix.py`
- Test: `tests/test_ablation_matrix.py`
- Modify: `src/tokeye/training/big_tf_unet_ablation/utils/configuration.py` (point `_PIPELINE_YAML` to `ablation.yaml`)

- [ ] **Step 1: Write `config/ablation.yaml`**

```yaml
# Ablation pipeline config — single source of truth.
stft: { nfft: 1024, hop_length: 128 }

modalities:
  co2: { input_key: co2, channels: [0, 1, 2, 3] }
  mhr: { input_key: mhr, channels: [3, 4, 5, 6] }
  ece: { input_key: ece, channels: [8, 12, 16, 20, 24, 28, 32, 36] }
  bes: { input_key: bes, channels: [26, 28, 30, 32, 34, 36, 38, 40] }

extraction:
  subseq_len: 66000
  preemphasis_coeff: 0.99
  fs_khz: 500
  ip_threshold: 0.1
  max_windows_per_shot_precap: 60   # generous cap before STFT (bounds step_2a work)

window_filter:
  enabled: true
  weights: /scratch/gpfs/nc1514/aemodes/model/big_mode_v1-5_weights.pt
  weights_fallback: model/big_tf_unet_251210_weights.pt
  max_windows_per_shot: 25
  activity_threshold: 0.5      # sigmoid cutoff for "active" pixel
  min_activity: 0.0005         # min active-pixel fraction; drop near-empty windows
  mean: 17.84620821169868      # eval-consistent normalization
  std: 25.016818830630463

baseline:
  enabled: true                # << toggled per variant
  method: fabc
  method_kwargs: { lam: 1.0e5 }
  bin_cutting: auto
  gradient_threshold: 0.5

correlation:
  enabled: true                # << toggled per variant (denoise on/off)
  representation: complex      # << complex | magnitude  (toggled per variant)
  first_layer_size: 32
  num_layers: 5
  clamp_range: auto
  clamp_percentiles: [1, 99]
  bin_masking: auto
  batch_size: 36
  max_epochs: 30
  tv_patience: 3

threshold:
  min_size: auto
  min_size_fraction: 0.0002
  remove_bottom_rows: auto
  remove_top_rows: auto
  row_removal_fraction_bottom: 0.01
  row_removal_fraction_top: 0.004

refiner: { first_layer_size: 32, num_layers: 5, batch_size: 56, max_epochs: 200, n_folds: 5, loss_type: symmetric_bce_dice, mc_dropout_samples: 15 }
final:   { first_layer_size: 32, num_layers: 5, max_epochs: 100, n_folds: 5, loss_type: focal, gamma: 2.0 }

variants:
  - { id: full,       baseline: true,  denoise: true,  representation: complex }
  - { id: mag,        baseline: true,  denoise: true,  representation: magnitude }
  - { id: nobaseline, baseline: false, denoise: true,  representation: complex }
  - { id: nodenoise,  baseline: true,  denoise: false, representation: complex }

paths:
  shots_path: data/autoprocess/settings/shots.txt
  faith_cfg_path: data/autoprocess/settings/faith_dataset_multiscale.yaml
  cache_dir: data/cache/ablation
  model_dir: model/ablation
  task_matrix_path: data/cache/ablation/task_matrix.json

cleanup:
  enabled: true
  keep_steps: [step_6a, step_6b, step_6c, step_6d]
  delete_steps: [step_0c, step_2a, step_2b, step_2b_baseline, step_3a, step_3b, step_4a_threshold, step_4a_threshold_baseline, step_5a]

smoke: { enabled: false, n_shots: 2, max_windows_per_shot: 2, n_folds: 2, max_epochs: 1, refiner_max_epochs: 1, final_max_epochs: 1 }
```

- [ ] **Step 2: Point config loader at ablation.yaml**

In `src/tokeye/training/big_tf_unet_ablation/utils/configuration.py`, change the `_PIPELINE_YAML` line:
```python
_PIPELINE_YAML = Path(__file__).resolve().parent.parent / "config" / "ablation.yaml"
```

- [ ] **Step 3: Write the failing test `tests/test_ablation_matrix.py`**

```python
from __future__ import annotations
from tokeye.training.big_tf_unet_ablation.config.variants import AblationVariant, build_variants
from tokeye.training.big_tf_unet_ablation.ablation_matrix import variant_from_index, n_variants

CFG = {
    "variants": [
        {"id": "full", "baseline": True, "denoise": True, "representation": "complex"},
        {"id": "mag", "baseline": True, "denoise": True, "representation": "magnitude"},
        {"id": "nobaseline", "baseline": False, "denoise": True, "representation": "complex"},
        {"id": "nodenoise", "baseline": True, "denoise": False, "representation": "complex"},
    ]
}

def test_build_variants_count_and_ids():
    vs = build_variants(CFG)
    assert [v.id for v in vs] == ["full", "mag", "nobaseline", "nodenoise"]
    assert n_variants(CFG) == 4

def test_variant_flags():
    vs = {v.id: v for v in build_variants(CFG)}
    assert vs["full"].baseline and vs["full"].denoise and vs["full"].representation == "complex"
    assert vs["mag"].representation == "magnitude"
    assert vs["nobaseline"].baseline is False
    assert vs["nodenoise"].denoise is False

def test_variant_from_index_deterministic():
    assert variant_from_index(CFG, 0).id == "full"
    assert variant_from_index(CFG, 3).id == "nodenoise"
```

- [ ] **Step 4: Run test, verify it fails**

Run: `cd /scratch/gpfs/nc1514/tokeye && .venv/bin/python -m pytest tests/test_ablation_matrix.py -q`
Expected: FAIL (ModuleNotFoundError: variants).

- [ ] **Step 5: Write `config/variants.py`**

```python
"""Ablation variant definitions and enumeration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AblationVariant:
    id: str
    baseline: bool        # apply ALS broadband-coherent separation to coherent path
    denoise: bool         # run multichannel self-supervised denoiser
    representation: str    # "complex" | "magnitude" (denoiser input)

    def __post_init__(self) -> None:
        if self.representation not in ("complex", "magnitude"):
            raise ValueError(f"bad representation: {self.representation}")

def build_variants(config: dict) -> list[AblationVariant]:
    return [
        AblationVariant(
            id=v["id"], baseline=bool(v["baseline"]),
            denoise=bool(v["denoise"]), representation=str(v["representation"]),
        )
        for v in config["variants"]
    ]
```

- [ ] **Step 6: Write `ablation_matrix.py`**

```python
"""Map SLURM array index → ablation variant."""
from __future__ import annotations
from .config.variants import AblationVariant, build_variants

def n_variants(config: dict) -> int:
    return len(build_variants(config))

def variant_from_index(config: dict, index: int) -> AblationVariant:
    vs = build_variants(config)
    if not (0 <= index < len(vs)):
        raise IndexError(f"variant index {index} out of range 0..{len(vs)-1}")
    return vs[index]
```

- [ ] **Step 7: Run test, verify PASS**

Run: `cd /scratch/gpfs/nc1514/tokeye && .venv/bin/python -m pytest tests/test_ablation_matrix.py -q`
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add src/tokeye/training/big_tf_unet_ablation/config tests/test_ablation_matrix.py src/tokeye/training/big_tf_unet_ablation/ablation_matrix.py src/tokeye/training/big_tf_unet_ablation/utils/configuration.py
git commit -m "ablation config + variant matrix"
```

---

## Phase 2 — Model-based window filter (step_2f)

### Task 2: `window_filter.py` — score windows with the existing TokEye model

**Files:**
- Create: `src/tokeye/training/big_tf_unet_ablation/window_filter.py`
- Create: `src/tokeye/training/big_tf_unet_ablation/step_2f_filter_windows.py`
- Test: `tests/test_window_filter.py`

- [ ] **Step 1: Write the failing test `tests/test_window_filter.py`**

```python
from __future__ import annotations
import numpy as np
from tokeye.training.big_tf_unet_ablation.window_filter import (
    activity_score_from_sigmoid, select_window_indices,
)

def test_activity_score_counts_active_fraction():
    sig = np.zeros((2, 10, 10), dtype=np.float32)   # (2ch, H, W)
    sig[0, :5, :] = 0.9                              # half active in coherent ch
    score = activity_score_from_sigmoid(sig, threshold=0.5)
    assert abs(score - 0.5) < 1e-6                   # max over channels of active fraction

def test_select_window_indices_caps_and_drops_empties():
    # per-shot scores; keep top-k above floor
    scores = {0: 0.9, 1: 0.0001, 2: 0.4, 3: 0.7, 4: 0.0}
    kept = select_window_indices(scores, max_windows=2, min_activity=0.001)
    assert kept == [0, 3]                            # top-2 above floor, sorted by score desc
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cd /scratch/gpfs/nc1514/tokeye && .venv/bin/python -m pytest tests/test_window_filter.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write `window_filter.py`**

```python
"""Model-based window activity filter.

Runs the existing full-recipe TokEye surrogate on each window's per-channel
magnitude spectrogram and keeps the most-active windows per shot. Applied
identically to all variants, so it cannot bias the relative comparison.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import torch

logger = logging.getLogger(__name__)

def _remap_legacy_state_dict(sd: dict) -> dict:
    idx_map = {"0": "0", "1": "1", "4": "3", "5": "4"}
    out = {}
    for k, v in sd.items():
        nk = k.replace(".double_conv.", ".conv.").replace(".maxpool_conv.1.", ".down.1.")
        parts = nk.split(".")
        for i, p in enumerate(parts):
            if p == "conv" and i + 1 < len(parts) and parts[i + 1] in idx_map:
                parts[i + 1] = idx_map[parts[i + 1]]
                break
        out[".".join(parts)] = v
    return out

def load_filter_model(weights: str | Path, fallback: str | Path | None, device: str):
    from tokeye.models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig
    from tokeye.models.big_tf_unet.model_big_tf_unet import BigTFUNetModel
    wp = Path(weights)
    if not wp.exists() and fallback is not None:
        wp = Path(fallback)
    cfg = BigTFUNetConfig(in_channels=1, out_channels=2, num_layers=5,
                          first_layer_size=32, dropout_rate=0.0)
    model = BigTFUNetModel(cfg)
    sd = _remap_legacy_state_dict(torch.load(wp, weights_only=True, map_location="cpu"))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    logger.info(f"filter model loaded from {wp}")
    return model

def _pad_to_multiple(x: torch.Tensor, m: int = 32) -> torch.Tensor:
    # x: (1,1,H,W) → pad H,W up to multiple of m (reflect)
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x

def activity_score_from_sigmoid(sig: np.ndarray, threshold: float) -> float:
    """sig: (2, H, W) sigmoid outputs → max over channels of active-pixel fraction."""
    active = (sig > threshold).mean(axis=(1, 2))   # per channel fraction
    return float(active.max())

@torch.no_grad()
def score_window(model, complex_window: np.ndarray, mean: float, std: float,
                 threshold: float, device: str) -> float:
    """complex_window: (C, F, T, 2). Returns max activity over channels."""
    C = complex_window.shape[0]
    best = 0.0
    for c in range(C):
        mag = np.sqrt(complex_window[c, ..., 0] ** 2 + complex_window[c, ..., 1] ** 2)
        mag = np.log1p(mag).astype(np.float32)
        mag = (mag - mean) / std
        x = torch.from_numpy(mag).float().unsqueeze(0).unsqueeze(0).to(device)
        x = _pad_to_multiple(x, 32)
        out = model(x)[0]                          # (1,2,H,W)
        sig = torch.sigmoid(out[0]).cpu().numpy()   # (2,H,W)
        best = max(best, activity_score_from_sigmoid(sig, threshold))
    return best

def select_window_indices(scores: dict[int, float], max_windows: int,
                          min_activity: float) -> list[int]:
    """Keep indices above floor, top-`max_windows` by score, returned sorted by score desc."""
    above = [(i, s) for i, s in scores.items() if s >= min_activity]
    above.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in above[:max_windows]]
```

- [ ] **Step 4: Run test, verify PASS**

Run: `cd /scratch/gpfs/nc1514/tokeye && .venv/bin/python -m pytest tests/test_window_filter.py -q`
Expected: 2 passed.

- [ ] **Step 5: Write `step_2f_filter_windows.py`**

```python
"""Step 2f: filter windows by model activity → reduced shared step_2a_filtered.h5.

Groups windows by shot via frame_info.csv (written by step_0c), scores each
window, keeps <=max_windows_per_shot above the activity floor, and writes a new
HDF5 containing only kept windows (re-indexed 0..N-1) plus a filtered frame_info.
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
import pandas as pd
from .utils.configuration import load_settings
from .utils.hdf5_io import create_step_file, iter_samples, read_sample, write_sample, get_sample_count
from .window_filter import load_filter_model, score_window, select_window_indices

logger = logging.getLogger(__name__)

default_settings = {
    "input_h5": Path("data/cache/ablation/shared/step_2a.h5"),
    "output_h5": Path("data/cache/ablation/shared/step_2a_filtered.h5"),
    "frame_info_csv": Path("data/cache/ablation/shared/frame_info.csv"),
    "frame_info_out": Path("data/cache/ablation/shared/frame_info_filtered.csv"),
    "enabled": True, "max_windows_per_shot": 25, "activity_threshold": 0.5,
    "min_activity": 0.0005, "mean": 17.8462, "std": 25.0168,
    "weights": "/scratch/gpfs/nc1514/aemodes/model/big_mode_v1-5_weights.pt",
    "weights_fallback": "model/big_tf_unet_251210_weights.pt",
}

def main(config_path=None, settings=None):
    import torch
    if settings is None:
        settings = load_settings(config_path, default_settings)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_h5 = Path(settings["input_h5"]); out_h5 = Path(settings["output_h5"])
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    fi = pd.read_csv(settings["frame_info_csv"])     # index aligns with sample idx

    if not settings.get("enabled", True):
        # passthrough: copy all
        h5 = create_step_file(out_h5, metadata={"filtered": False})
        for idx, data in iter_samples(in_h5):
            write_sample(h5, idx, data)
        h5.close(); fi.to_csv(settings["frame_info_out"], index=False); return

    model = load_filter_model(settings["weights"], settings.get("weights_fallback"), device)
    n = get_sample_count(in_h5)
    # score per window
    scores: dict[int, float] = {}
    for idx in range(n):
        scores[idx] = score_window(model, read_sample(in_h5, idx),
                                   settings["mean"], settings["std"],
                                   settings["activity_threshold"], device)
    # group by shot, select per shot
    kept: list[int] = []
    for shotn, grp in fi.groupby("shotn"):
        shot_scores = {i: scores[i] for i in grp.index if i in scores}
        kept.extend(select_window_indices(shot_scores,
                    settings["max_windows_per_shot"], settings["min_activity"]))
    kept.sort()
    logger.info(f"window filter: kept {len(kept)}/{n} windows")
    # write reduced h5 + frame_info
    h5 = create_step_file(out_h5, metadata={"filtered": True, "kept": len(kept)})
    for new_idx, old_idx in enumerate(kept):
        write_sample(h5, new_idx, read_sample(in_h5, old_idx))
    h5.close()
    fi.iloc[kept].reset_index(drop=True).to_csv(settings["frame_info_out"], index=False)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
```

- [ ] **Step 6: Commit**

```bash
git add src/tokeye/training/big_tf_unet_ablation/window_filter.py src/tokeye/training/big_tf_unet_ablation/step_2f_filter_windows.py tests/test_window_filter.py
git commit -m "model-based window activity filter (step_2f)"
```

---

## Phase 3 — Step toggles (baseline, representation/denoise)

### Task 3: `step_2b` baseline toggle

**Files:**
- Modify: `src/tokeye/training/big_tf_unet_ablation/step_2b_filter_spectrogram.py`
- Test: `tests/test_step2b_toggle.py`

- [ ] **Step 1: Write failing test `tests/test_step2b_toggle.py`**

```python
from __future__ import annotations
import numpy as np
from tokeye.training.big_tf_unet_ablation.step_2b_filter_spectrogram import _process_rotation

def test_baseline_off_is_logmag_without_subtraction():
    rng = np.random.default_rng(0)
    data = rng.random((64, 32)).astype(np.float32) + 0.1
    out_off, bl_off = _process_rotation(data, 2, 2, "fabc", {"lam": 1e5}, baseline_enabled=False)
    expected = np.log1p(np.abs(data))
    # edges masked but interior should equal log1p(|.|) (no division by baseline)
    assert np.allclose(out_off[5:-5], expected[5:-5], atol=1e-5)

def test_baseline_on_differs_from_off():
    rng = np.random.default_rng(1)
    data = rng.random((64, 32)).astype(np.float32) + 0.1
    on, _ = _process_rotation(data, 2, 2, "fabc", {"lam": 1e5}, baseline_enabled=True)
    off, _ = _process_rotation(data, 2, 2, "fabc", {"lam": 1e5}, baseline_enabled=False)
    assert not np.allclose(on, off)
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `.venv/bin/python -m pytest tests/test_step2b_toggle.py -q`
Expected: FAIL (`_process_rotation` takes no `baseline_enabled`).

- [ ] **Step 3: Edit `step_2b_filter_spectrogram.py`**

Change `_process_rotation` signature and body (currently lines 59-80) to:
```python
def _process_rotation(
    data: np.ndarray, lower_idx: int, upper_idx: int,
    method: str, method_kwargs: dict, baseline_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Process one real/imag component of one channel."""
    sxx = np.abs(data)
    sxx = np.log1p(sxx)
    edge_val_lo = sxx[lower_idx].mean() if lower_idx > 0 else sxx.mean()
    edge_val_hi = sxx[-upper_idx].mean() if upper_idx > 0 else sxx.mean()
    if lower_idx > 0:
        sxx[:lower_idx] = edge_val_lo
    if upper_idx > 0:
        sxx[-upper_idx + 1:] = edge_val_hi
    baseline = _fit_baseline(sxx, method, method_kwargs)   # always compute (for transient h5)
    if baseline_enabled:
        sxx = (sxx - baseline) / (baseline + 1e-6)
    return sxx, baseline
```
Thread `baseline_enabled` through `_process_channel` (add param, pass to both `_process_rotation` calls) and read it in `main`:
```python
    baseline_enabled = settings.get("baseline_enabled", settings.get("baseline", {}).get("enabled", True))
```
and pass `baseline_enabled` into `_process_channel(...)`. Note: the baseline output h5 (`output_baseline_h5`) is unchanged (always the computed baseline) so the transient branch is identical across baseline on/off — isolating the subtraction effect on the coherent path only.

- [ ] **Step 4: Run test, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_step2b_toggle.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokeye/training/big_tf_unet_ablation/step_2b_filter_spectrogram.py tests/test_step2b_toggle.py
git commit -m "step_2b baseline on/off toggle"
```

### Task 4: `step_3a` magnitude representation (`BTNMag`)

**Files:**
- Modify: `src/tokeye/training/big_tf_unet_ablation/step_3a_correlation_analysis.py`
- Test: `tests/test_btn_mag.py`

- [ ] **Step 1: Write failing test `tests/test_btn_mag.py`**

```python
from __future__ import annotations
import torch
from tokeye.training.big_tf_unet_ablation.step_3a_correlation_analysis import BTN, BTNMag

def test_complex_btn_shapes():
    m = BTN(in_channels=4, num_layers=3, first_layer_size=8)
    x = torch.randn(2, 4, 64, 32, 2)            # (B,C,F,T,2)
    y = m(x)
    assert y.shape == (2, 4, 64, 32, 2)

def test_mag_btn_shapes():
    m = BTNMag(in_channels=4, num_layers=3, first_layer_size=8)
    x = torch.randn(2, 4, 64, 32)               # (B,C,F,T) magnitude
    y = m(x)
    assert y.shape == (2, 4, 64, 32)
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `.venv/bin/python -m pytest tests/test_btn_mag.py -q`
Expected: FAIL (BTNMag not defined).

- [ ] **Step 3: Add `BTNMag` + magnitude dataset/loss/predict paths**

In `step_3a_correlation_analysis.py`, after the `BTN` class (line ~198) add:
```python
class BTNMag(nn.Module):
    """Magnitude-only denoiser: single-component U-Net (no real/imag split)."""
    def __init__(self, in_channels=4, num_layers=5, first_layer_size=32):
        super().__init__()
        self.in_channels = in_channels
        self.unet = UNet(in_channels=in_channels, out_channels=in_channels,
                         num_layers=num_layers, first_layer_size=first_layer_size)
    def forward(self, x):                         # x: (B, C, F, T)
        return self.unet(x)
```
Then make the representation switch in `BTNModule.__init__` (replace the `self.unet = BTN(...)` block ~line 222):
```python
        self.representation = self.settings.get("representation", "complex")
        if self.representation == "magnitude":
            self.unet = BTNMag(in_channels=self.in_channels, num_layers=num_layers,
                               first_layer_size=first_layer_size)
        else:
            self.unet = BTN(in_channels=self.in_channels, num_layers=num_layers,
                            first_layer_size=first_layer_size)
```
Add magnitude-aware helpers: in `_single_channel_loss` / `_multichannel_loss`, when `self.representation == "magnitude"` drop the `.flip(-1)` (the flip handles the imag axis; magnitude has none). Concretely:
```python
    def _tflip(self, t):
        return t if self.representation == "magnitude" else t.flip(-1)
```
and replace `y.flip(-1)` → `self._tflip(y)` and `y_i.flip(-1)` → `self._tflip(y_i)`; in TV updates guard the `[..., 1]` imag-axis updates behind `if self.representation != "magnitude"`. In `SpecDataset.__getitem__`, when magnitude, collapse the last axis before returning: add at the end of `__getitem__`:
```python
        out = (data - mean) / (std + 1e-6)
        if self.settings.get("representation") == "magnitude":
            out = torch.sqrt(out[..., 0] ** 2 + out[..., 1] ** 2)   # (C,F,T)
        return out
```
In `predict_step`, the magnitude path writes `(C,F,T)` (no concat of two halves with last-dim 2); guard the output assembly:
```python
        if self.representation == "magnitude":
            out_data = torch.cat([y_hats1, y_hats2], dim=1).float().cpu().numpy()  # (B,2C,F,T)
        else:
            out_data = torch.cat([y_hats1, y_hats2], dim=-1).float().cpu().numpy()  # (B,...,2)
```
(Downstream `step_4a` handles both `ndim==3` complex and `ndim==2` magnitude per channel; the magnitude denoised output is `(C,F,T)` so `step_4a` uses the `np.abs` branch. Keep behavior identical.)

- [ ] **Step 4: Run test, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_btn_mag.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokeye/training/big_tf_unet_ablation/step_3a_correlation_analysis.py tests/test_btn_mag.py
git commit -m "step_3a magnitude denoiser (BTNMag) representation toggle"
```

### Task 5: `step_5a` + `step_6a` dual-mask paths (HDF5)

**Files:**
- Modify: `src/tokeye/training/big_tf_unet_ablation/step_5a_combine_spectrogram.py`
- Modify: `src/tokeye/training/big_tf_unet_ablation/step_6a_convert_tif.py`
- Test: `tests/test_step6a_dual_mask.py`

Goal: ensure `step_6a` produces the **2-channel** mask `[coherent, transient]`. The multiscale `step_6a` was single-mask and had legacy joblib paths. Reuse the original `big_tf_unet/step_6a_convert_tif.py` dual-mask logic (`process_data_mask_dual`, `validate_directory` reading `step_4a_threshold` + `step_4a_threshold_baseline`) but adapt it to read the ablation's HDF5 outputs and write TIF pairs.

- [ ] **Step 1: Write failing test `tests/test_step6a_dual_mask.py`**

```python
from __future__ import annotations
import numpy as np
from tokeye.training.big_tf_unet_ablation.step_6a_convert_tif import process_data_mask_dual

def test_dual_mask_stacks_two_channels():
    coh = np.ones((20, 30, 1), dtype=np.float32)
    tra = np.zeros((20, 30, 1), dtype=np.float32)
    out = process_data_mask_dual(coh, tra)
    assert out.shape == (2, 20, 30)
    assert out[0].sum() > 0 and out[1].sum() == 0      # ch0 coherent, ch1 transient
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `.venv/bin/python -m pytest tests/test_step6a_dual_mask.py -q`
Expected: FAIL (function missing / signature differs).

- [ ] **Step 3: Port dual-mask logic into ablation `step_6a`**

Copy `process_data_mask_dual` and `process_data_img` from `src/tokeye/training/big_tf_unet/step_6a_convert_tif.py:170-215` into the ablation `step_6a`. Rework `main` to consume HDF5 instead of joblib dirs:
- read img from `settings["input_img_h5"]` (= shared `step_2a_filtered.h5`),
- read coherent mask from `settings["input_mask_h5"]` (= `step_4a_threshold.h5`),
- read transient mask from `settings["input_mask_baseline_h5"]` (= `step_4a_threshold_baseline.h5`),
- compute per-directory (per-variant) normalization stats from the img magnitudes (reuse `collect_image_statistics` logic but over HDF5 samples),
- for each sample, for each channel `c`: `img = process_data_img(img_sample[c], c, stats, zscore_clip)`; `mask = process_data_mask_dual(coh[c], tra[c])`; write `{idx}_img.tif` (shape `(1,H,W)`) and `{idx}_mask.tif` (shape `(2,H,W)`), incrementing a global channel index.

Note the original `process_data_mask_dual` zeros `mask_normal[-4:]`; keep that (it is a fixed edge guard, not a per-image eyeballed value).

- [ ] **Step 4: Ensure `step_5a` writes per-shot img+coherent+transient to one HDF5**

In ablation `step_5a`, set inputs: `input_img_h5` (shared step_2a_filtered), `input_mask_h5` (coherent step_4a_threshold), `input_mask_baseline_h5` (transient). Output `step_5a.h5` with three datasets per sample group OR (simpler) keep `step_5a` only for img+coherent and let `step_6a` read the transient mask directly from `step_4a_threshold_baseline.h5`. **Decision:** skip per-shot combination complexity — `step_6a` reads the three HDF5s directly (img, coherent, transient), all per-window aligned by index. Mark `step_5a` as a no-op passthrough in the ablation (kept for numbering parity) OR remove it from the step list. Remove `step_5a` from the ablation step list in the orchestrator (Task 7).

- [ ] **Step 5: Run test, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_step6a_dual_mask.py -q`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/tokeye/training/big_tf_unet_ablation/step_6a_convert_tif.py src/tokeye/training/big_tf_unet_ablation/step_5a_combine_spectrogram.py tests/test_step6a_dual_mask.py
git commit -m "step_6a dual-mask (coherent+transient) over HDF5"
```

---

## Phase 4 — 5-fold final surrogate

### Task 6: `step_6d` train 5 CV folds

**Files:**
- Modify: `src/tokeye/training/big_tf_unet_ablation/step_6d_final.py`
- Test: `tests/test_step6d_folds.py`

- [ ] **Step 1: Write failing test `tests/test_step6d_folds.py`**

```python
from __future__ import annotations
from tokeye.training.big_tf_unet_ablation.step_6d_final import make_fold_indices

def test_make_fold_indices_partition():
    folds = make_fold_indices(n_samples=10, n_folds=5, seed=42)
    assert len(folds) == 5
    # each fold: (train_idx, val_idx); val sets partition 0..9
    val_all = sorted(i for _, val in folds for i in val)
    assert val_all == list(range(10))
    for train, val in folds:
        assert set(train).isdisjoint(val)
        assert len(train) + len(val) == 10
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `.venv/bin/python -m pytest tests/test_step6d_folds.py -q`
Expected: FAIL (make_fold_indices missing).

- [ ] **Step 3: Add `make_fold_indices` and a fold loop to `step_6d`**

Add near the top of `step_6d_final.py`:
```python
def make_fold_indices(n_samples: int, n_folds: int, seed: int = 42):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(train.tolist(), val.tolist()) for train, val in kf.split(range(n_samples))]
```
Refactor `main` so the existing single-model training body becomes `_train_one_fold(settings, train_indices, val_indices, out_dir)` (saving `out_dir/best_model.ckpt` and `out_dir/final.torchscript.pt`). New `main`:
```python
def main(config_path=None, settings=None):
    if settings is None:
        settings = load_settings(config_path, default_settings)
    n_folds = int(settings.get("n_folds", 1))
    model_dir = Path(settings["model_dir"])
    # determine n_samples from the input dir (count *_img.tif)
    n_samples = len(list(Path(settings["input_dir"]).glob("*_img.tif")))
    if n_folds <= 1:
        _train_one_fold(settings, list(range(n_samples)), list(range(n_samples)), model_dir)
        return
    folds = make_fold_indices(n_samples, n_folds, seed=settings.get("seed", 42))
    for k, (train_idx, val_idx) in enumerate(folds):
        fold_dir = model_dir / f"fold_{k}"
        _train_one_fold(settings, train_idx, val_idx, fold_dir)
```
Ensure `_train_one_fold` uses `train_indices`/`val_indices` in the data module (the module already accepts them, lines 128-162) instead of `all_indices`.

- [ ] **Step 4: Run test, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_step6d_folds.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokeye/training/big_tf_unet_ablation/step_6d_final.py tests/test_step6d_folds.py
git commit -m "step_6d: 5-fold CV surrogates for uncertainty"
```

---

## Phase 5 — Orchestrator + task matrix (correct dual-branch wiring)

### Task 7: `task_matrix.py` + `orchestrator.py`

**Files:**
- Create: `src/tokeye/training/big_tf_unet_ablation/task_matrix.py` (copy from multiscale, adapt keys)
- Create: `src/tokeye/training/big_tf_unet_ablation/orchestrator.py`
- Test: `tests/test_ablation_orchestrator_paths.py`

- [ ] **Step 1: Copy task_matrix**

```bash
cp src/tokeye/training/big_tf_unet_multiscale/task_matrix.py src/tokeye/training/big_tf_unet_ablation/task_matrix.py
```
(API reused: `is_shared_step_complete`, `mark_shared_step_complete`, `is_step_complete(combo_id, step)`, `mark_step_complete`, `print_status`. Use `combo_id = variant.id` and a synthetic `step` name per branch, e.g. `step_4a_coh`, `step_4a_tra`, `step_6d_fold_{k}`.)

- [ ] **Step 2: Write failing test `tests/test_ablation_orchestrator_paths.py`**

```python
from __future__ import annotations
from pathlib import Path
from tokeye.training.big_tf_unet_ablation.config.variants import AblationVariant
from tokeye.training.big_tf_unet_ablation.orchestrator import build_variant_step_settings

CFG = {
    "stft": {"nfft": 1024, "hop_length": 128},
    "paths": {"cache_dir": "data/cache/ablation"},
    "baseline": {"enabled": True}, "correlation": {}, "threshold": {},
    "refiner": {}, "final": {"n_folds": 5},
}

def _v(id, base, den, rep):
    return AblationVariant(id=id, baseline=base, denoise=den, representation=rep)

def test_coherent_threshold_reads_denoised_when_denoise_on():
    s = build_variant_step_settings(CFG, _v("full", True, True, "complex"), "step_4a_coh")
    assert s["input_h5"].name == "step_3b.h5"

def test_coherent_threshold_reads_step2b_when_denoise_off():
    s = build_variant_step_settings(CFG, _v("nodenoise", True, False, "complex"), "step_4a_coh")
    assert s["input_h5"].name == "step_2b.h5"

def test_transient_threshold_always_reads_baseline():
    s = build_variant_step_settings(CFG, _v("full", True, True, "complex"), "step_4a_tra")
    assert s["input_h5"].name == "step_2b_baseline.h5"

def test_baseline_flag_propagates_to_step2b():
    s = build_variant_step_settings(CFG, _v("nobaseline", False, True, "complex"), "step_2b")
    assert s["baseline_enabled"] is False

def test_representation_propagates_to_step3a():
    s = build_variant_step_settings(CFG, _v("mag", True, True, "magnitude"), "step_3a")
    assert s["representation"] == "magnitude"
```

- [ ] **Step 3: Run test, verify FAIL**

Run: `.venv/bin/python -m pytest tests/test_ablation_orchestrator_paths.py -q`
Expected: FAIL (orchestrator missing).

- [ ] **Step 4: Write `orchestrator.py`**

```python
"""Ablation pipeline orchestrator.

Modes:
  --shared-steps step_0a,step_0b,step_0c,step_2a,step_2f     (V100/CPU, once)
  --variant-index N [--steps ...]                            (SLURM array 0..3)
  --eval                                                      (delegates to eval script)
  --status
"""
from __future__ import annotations
import argparse, importlib, logging, shutil
from pathlib import Path
from .ablation_matrix import build_variants, variant_from_index, n_variants
from .config.variants import AblationVariant
from .task_matrix import TaskMatrix
from .utils.configuration import load_pipeline_config

logger = logging.getLogger(__name__)

_STEP_MODULES = {
    "step_0a": ".step_0a_extract_faithdata", "step_0b": ".step_0b_filter_faithdata",
    "step_0c": ".step_0c_convert_faithdata", "step_2a": ".step_2a_make_spectrogram",
    "step_2f": ".step_2f_filter_windows", "step_2b": ".step_2b_filter_spectrogram",
    "step_3a": ".step_3a_correlation_analysis", "step_3b": ".step_3b_extract_correlation",
    "step_4a_coh": ".step_4a_threshold", "step_4a_tra": ".step_4a_threshold",
    "step_6a": ".step_6a_convert_tif", "step_6b": ".step_6b_refiner",
    "step_6c": ".step_6c_convert_predictions", "step_6d": ".step_6d_final",
}
PKG = "tokeye.training.big_tf_unet_ablation"
VARIANT_STEPS = ["step_2b", "step_3a", "step_3b", "step_4a_coh", "step_4a_tra",
                 "step_6a", "step_6b", "step_6c", "step_6d"]

def _run(step, settings):
    base = step.split("_coh")[0].split("_tra")[0] if step.startswith("step_4a") else step
    mod = importlib.import_module(_STEP_MODULES[step], package=PKG)
    mod.main(settings=settings)

def _shared_dir(cfg) -> Path:
    return Path(cfg["paths"]["cache_dir"]) / "shared"

def _variant_dir(cfg, v: AblationVariant) -> Path:
    return Path(cfg["paths"]["cache_dir"]) / v.id

def build_variant_step_settings(cfg: dict, v: AblationVariant, step: str) -> dict:
    """Return the input/output/flags settings dict for one variant step."""
    shared, vdir = _shared_dir(cfg), _variant_dir(cfg, v)
    stft = cfg["stft"]
    s: dict = {"overwrite": True, "combo_id": v.id,
               "nfft": stft["nfft"], "hop_length": stft["hop_length"],
               "frame_info_csv": shared / "frame_info_filtered.csv"}
    # merge config sections
    for sec in ("baseline", "correlation", "threshold", "refiner", "final"):
        if sec in cfg:
            s[sec] = cfg[sec]
    if step == "step_2b":
        s["input_h5"] = shared / "step_2a_filtered.h5"
        s["output_h5"] = vdir / "step_2b.h5"
        s["output_baseline_h5"] = vdir / "step_2b_baseline.h5"
        s["baseline_enabled"] = v.baseline
    elif step == "step_3a":
        s["input_h5"] = vdir / "step_2b.h5"
        s["output_h5"] = vdir / "step_3a.h5"
        s["representation"] = v.representation
        s.update(cfg.get("correlation", {}))
    elif step == "step_3b":
        s["input_h5"] = vdir / "step_3a.h5"
        s["reference_h5"] = vdir / "step_2b.h5"
        s["output_h5"] = vdir / "step_3b.h5"
    elif step == "step_4a_coh":
        s["input_h5"] = (vdir / "step_3b.h5") if v.denoise else (vdir / "step_2b.h5")
        s["output_h5"] = vdir / "step_4a_threshold.h5"
        s["threshold_output_path"] = vdir / "thresholds_coh.csv"
    elif step == "step_4a_tra":
        s["input_h5"] = vdir / "step_2b_baseline.h5"
        s["output_h5"] = vdir / "step_4a_threshold_baseline.h5"
        s["threshold_output_path"] = vdir / "thresholds_tra.csv"
    elif step == "step_6a":
        s["input_img_h5"] = shared / "step_2a_filtered.h5"
        s["input_mask_h5"] = vdir / "step_4a_threshold.h5"
        s["input_mask_baseline_h5"] = vdir / "step_4a_threshold_baseline.h5"
        s["output_dir"] = vdir / "step_6a"
    elif step in ("step_6b", "step_6c"):
        s["input_dir"] = vdir / "step_6a"
        s["output_dir"] = vdir / step
        s["model_dir"] = vdir / step / "models"
    elif step == "step_6d":
        s["input_dir"] = vdir / "step_6c"   # refined labels (6c) feed final
        s["model_dir"] = Path(cfg["paths"]["model_dir"]) / v.id
        s.update(cfg.get("final", {}))
    return s

def run_shared_steps(cfg, tm, steps):
    shared = _shared_dir(cfg); shared.mkdir(parents=True, exist_ok=True)
    for step in steps:
        if tm.is_shared_step_complete(step):
            logger.info(f"skip shared {step}"); continue
        s = {"overwrite": True}
        if "extraction" in cfg: s.update(cfg["extraction"])
        for k in ("shots_path", "faith_cfg_path"):
            if k in cfg.get("paths", {}): s[k] = Path(cfg["paths"][k])
        # wire shared input/outputs (0a→0b→0c→2a→2f)
        # (engineer: set input_dir/output_dir/input_h5/output_h5 per step to the shared dir;
        #  step_0c writes step_0c.h5 + frame_info.csv; step_2a reads step_0c.h5 → step_2a.h5;
        #  step_2f reads step_2a.h5 + frame_info.csv → step_2a_filtered.h5 + frame_info_filtered.csv)
        _wire_shared_io(cfg, step, s, shared)
        _run(step, s); tm.mark_shared_step_complete(step)

def run_variant_steps(cfg, tm, vidx, steps):
    v = variant_from_index(cfg, vidx)
    for step in steps:
        if step == "step_3a" and not v.denoise: continue
        if step == "step_3b" and not v.denoise: continue
        if tm.is_step_complete(v.id, step):
            logger.info(f"skip {v.id}/{step}"); continue
        s = build_variant_step_settings(cfg, v, step)
        _run(step, s); tm.mark_step_complete(v.id, step)
    if cfg.get("cleanup", {}).get("enabled") and _variant_done(cfg, tm, v):
        _cleanup(cfg, v)
```
Add the small helpers `_wire_shared_io`, `_variant_done`, `_cleanup`, and `main()` (argparse mirroring multiscale's, with `--shared-steps`, `--variant-index`, `--steps`, `--status`, `--config`). For `--variant-index` default `steps = VARIANT_STEPS`.

- [ ] **Step 5: Run test, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_ablation_orchestrator_paths.py -q`
Expected: 5 passed.

- [ ] **Step 6: Adapt `step_3b` to use `reference_h5`/`input_h5`/`output_h5` HDF5 keys**

The multiscale `step_3b` already unbatches HDF5; confirm it reads `settings["input_h5"]` and writes `settings["output_h5"]`. If it expects a `reference_dir`, adjust to index-based unbatch (no reference needed for HDF5). Keep output one-sample-per-window aligned with `step_2b`.

- [ ] **Step 7: Commit**

```bash
git add src/tokeye/training/big_tf_unet_ablation/orchestrator.py src/tokeye/training/big_tf_unet_ablation/task_matrix.py src/tokeye/training/big_tf_unet_ablation/step_3b_extract_correlation.py tests/test_ablation_orchestrator_paths.py
git commit -m "ablation orchestrator with correct dual-branch dataflow + task matrix"
```

---

## Phase 6 — Evaluation with uncertainty

### Task 8: `TJII2021_ablation.py` (per variant × fold, across-fold + image bootstrap)

**Files:**
- Create: `scripts/eval/TJII2021_ablation.py`
- Test: `tests/test_ablation_eval_aggregate.py`

- [ ] **Step 1: Write failing test `tests/test_ablation_eval_aggregate.py`**

```python
from __future__ import annotations
import numpy as np
from scripts_eval_shim import aggregate_across_folds   # see Step 3 import note

def test_aggregate_mean_std_ci():
    rows = [{"recall": r, "f1": r, "iou": r} for r in [0.50, 0.52, 0.48, 0.54, 0.46]]
    agg = aggregate_across_folds(rows, ["recall", "f1", "iou"], ci=95.0)
    assert abs(agg["recall"]["mean"] - 0.50) < 1e-9
    assert agg["recall"]["std"] > 0
    assert agg["recall"]["ci_lo"] < agg["recall"]["mean"] < agg["recall"]["ci_hi"]
    assert agg["recall"]["n"] == 5
```

(Import note: put `aggregate_across_folds` in a small importable module `src/tokeye/extra/eval/fold_stats.py` and import it both from the test and the script. Adjust the test import to `from tokeye.extra.eval.fold_stats import aggregate_across_folds`.)

- [ ] **Step 2: Create `src/tokeye/extra/eval/fold_stats.py`**

```python
"""Across-fold aggregation (mean ± std, normal-approx 95% CI)."""
from __future__ import annotations
import numpy as np

def aggregate_across_folds(rows: list[dict], metrics: list[str], ci: float = 95.0) -> dict:
    from scipy import stats as st
    out = {}
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=float)
        n = len(vals); mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            sem = std / np.sqrt(n)
            t = st.t.ppf(1 - (1 - ci / 100) / 2, df=n - 1)
            lo, hi = mean - t * sem, mean + t * sem
        else:
            lo = hi = mean
        out[m] = {"mean": mean, "std": std, "ci_lo": float(lo), "ci_hi": float(hi), "n": n}
    return out
```

- [ ] **Step 3: Run test, verify PASS**

Run: `.venv/bin/python -m pytest tests/test_ablation_eval_aggregate.py -q` (after fixing the import to `tokeye.extra.eval.fold_stats`)
Expected: 1 passed.

- [ ] **Step 4: Write `scripts/eval/TJII2021_ablation.py`**

Base it on `scripts/eval/TJII2021.py` (same TJ-II loading, normalization, `Metrics`, `PRSweep`). For each variant in `ablation.yaml` and each fold checkpoint under `model/ablation/{variant}/fold_{k}/best_model.ckpt`:
- build model, run all 493 TJ-II images, compute recall@0.5, F1@F1-opt, per-image IoU@F1-opt, and the image-level `PRSweep.bootstrap_ci`.
- collect one row per (variant, fold).
Then per variant call `aggregate_across_folds(fold_rows, ["recall","f1","iou"])` and write:
- `data/eval/results/TJII2021_ablation_folds.csv` (raw per-fold),
- `data/eval/results/TJII2021_ablation.csv` (variant: metric mean±std, ci_lo, ci_hi, n_folds=5, n_images=493),
- `data/eval/results/TJII2021_ablation_imageci.csv` (image bootstrap per variant, pooled or fold-0).

Use the same `remap_legacy_state_dict` + `BigTFUNetConfig(first_layer_size=cfg.final.first_layer_size)`. Resolve checkpoints with a glob; `strict=False` load (fold checkpoints are Lightning `.ckpt`, so load `["state_dict"]` and strip the `model.`/`net.` prefix as needed — verify against `step_6d` save format).

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/TJII2021_ablation.py src/tokeye/extra/eval/fold_stats.py tests/test_ablation_eval_aggregate.py
git commit -m "TJ-II ablation eval with across-fold + image-bootstrap uncertainty"
```

### Task 9: `ablation_figure.py` — figure + LaTeX table

**Files:**
- Create/extend: `scripts/eval/ablation_figure.py`

- [ ] **Step 1: Write the figure+table generator**

Read `TJII2021_ablation.csv`; produce:
- `data/eval/results/figures/tjii_ablation.png` — grouped bar chart (recall, F1, per-image IoU) with **error bars = 95% CI across folds**, one group per variant (`full`, `mag`, `nobaseline`, `nodenoise`), annotated n_folds/n_images.
- `data/eval/results/tjii_ablation_table.tex` — LaTeX `tabular`: Variant × {Recall, F1, per-img IoU} each as `mean ± std (95% CI)`.
Match the style of `scripts/eval/paper_figures.py`/`paper_tables.py` (font sizes, savefig dpi).

- [ ] **Step 2: Commit**

```bash
git add scripts/eval/ablation_figure.py
git commit -m "ablation figure + LaTeX table generator"
```

---

## Phase 7 — SLURM wrappers + smoke harness

### Task 10: SLURM scripts + smoke runner

**Files:**
- Create: `scripts/commands/ablation/shared.sh`, `variant.sh`, `train.sh`, `eval.sh`
- Create: `scripts/test_ablation_smoke.py`

- [ ] **Step 1: Write `scripts/commands/ablation/shared.sh`** (V100 head-node or short A100)

```bash
#!/bin/bash
#SBATCH --job-name=abl_shared
#SBATCH --output=logs/abl_shared_%j.out
#SBATCH --error=logs/abl_shared_%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
cd $SCRATCH/tokeye && source .venv/bin/activate
srun python -m tokeye.training.big_tf_unet_ablation.orchestrator \
  --shared-steps step_0c,step_2a,step_2f
```

- [ ] **Step 2: Write `scripts/commands/ablation/variant.sh`** (A100 array 0-3)

```bash
#!/bin/bash
#SBATCH --job-name=abl_var
#SBATCH --output=logs/abl_var_%A_%a.out
#SBATCH --error=logs/abl_var_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3%2     # at most 2 concurrent (2 A100s)
cd $SCRATCH/tokeye && source .venv/bin/activate
srun python -m tokeye.training.big_tf_unet_ablation.orchestrator \
  --variant-index $SLURM_ARRAY_TASK_ID
```

- [ ] **Step 3: Write `scripts/commands/ablation/eval.sh`** (V100)

```bash
#!/bin/bash
#SBATCH --job-name=abl_eval
#SBATCH --output=logs/abl_eval_%j.out
#SBATCH --error=logs/abl_eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
cd $SCRATCH/tokeye && source .venv/bin/activate
srun python scripts/eval/TJII2021_ablation.py
srun python scripts/eval/ablation_figure.py
```

- [ ] **Step 4: Write `scripts/test_ablation_smoke.py`** (end-to-end on V100, tiny)

```python
"""Tiny end-to-end smoke: 2 shots, 2 windows/shot, 2 folds, 1 epoch, all 4 variants.

Writes a temp config with smoke.enabled=true overrides, runs shared + each variant +
eval, and asserts that model/ablation/<variant>/fold_0/best_model.ckpt and
data/eval/results/TJII2021_ablation.csv exist.
"""
# (engineer: load ablation.yaml via OmegaConf, deep-merge the `smoke` overrides into
#  extraction.max_windows_per_shot_precap, window_filter.max_windows_per_shot,
#  refiner.n_folds/max_epochs, final.n_folds/max_epochs, and a 2-shot shots file;
#  call orchestrator.run_shared_steps then run_variant_steps for indices 0..3;
#  then run the eval main(); assert outputs exist.)
```

- [ ] **Step 5: Commit**

```bash
git add scripts/commands/ablation scripts/test_ablation_smoke.py
git commit -m "ablation SLURM wrappers + smoke harness"
```

---

## Phase 8 — Smoke, lint, launch

### Task 11: Lint + smoke test on V100

- [ ] **Step 1: Lint**

Run: `cd /scratch/gpfs/nc1514/tokeye && uv run ruff check src/tokeye/training/big_tf_unet_ablation scripts/eval/TJII2021_ablation.py scripts/eval/ablation_figure.py`
Fix any findings.

- [ ] **Step 2: Run unit tests**

Run: `.venv/bin/python -m pytest tests/test_ablation_matrix.py tests/test_window_filter.py tests/test_step2b_toggle.py tests/test_btn_mag.py tests/test_step6a_dual_mask.py tests/test_step6d_folds.py tests/test_ablation_orchestrator_paths.py tests/test_ablation_eval_aggregate.py -q`
Expected: all pass.

- [ ] **Step 3: End-to-end smoke on V100**

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/test_ablation_smoke.py`
Expected: prints kept-window counts, trains tiny models for all 4 variants, writes `TJII2021_ablation.csv`. Debug until green (this is where dataflow bugs surface — verify `step_6a` produced `(2,H,W)` masks and `step_4a_coh` consumed `step_3b`).

- [ ] **Step 4: Commit smoke fixes**

```bash
git add -A && git commit -m "fix issues found in end-to-end smoke"
```

### Task 12: Launch full ablation on SLURM

- [ ] **Step 1: Disable smoke, set real scale**

Confirm `ablation.yaml`: `smoke.enabled: false`, `window_filter.max_windows_per_shot: 25`, `refiner.n_folds: 5`, `final.n_folds: 5`. Confirm `paths.shots_path` = the 22-shot file.

- [ ] **Step 2: Submit shared, then variants, then eval (dependency chain)**

```bash
cd /scratch/gpfs/nc1514/tokeye
JID_SHARED=$(sbatch --parsable scripts/commands/ablation/shared.sh)
JID_VAR=$(sbatch --parsable --dependency=afterok:$JID_SHARED scripts/commands/ablation/variant.sh)
sbatch --dependency=afterok:$JID_VAR scripts/commands/ablation/eval.sh
squeue -u $USER
```

- [ ] **Step 3: Monitor**

Poll `python -m tokeye.training.big_tf_unet_ablation.orchestrator --status` and `du -sh data/cache/ablation` between stages; watch `logs/abl_*`. Re-submit failed variants (resumable via task_matrix).

- [ ] **Step 4: Produce final artifacts**

Confirm `data/eval/results/TJII2021_ablation.csv`, `figures/tjii_ablation.png`, `tjii_ablation_table.tex`. Report the table to the user.

---

## Self-review notes (coverage)

- Spec §3 variants → Task 1 (config) + Task 7 (toggles wired). ✓
- Spec §5 toggles: baseline → Task 3; representation/denoise → Task 4 + Task 7 (coherent input switch). ✓
- Spec §6 window filter → Task 2. ✓
- Spec §7 data reduction: precap (Task 0/config), window cap (Task 2), cleanup (Task 7 `_cleanup`). ✓
- Spec §8 UQ: 5-fold (Task 6) + across-fold agg + image bootstrap (Task 8). ✓
- Spec §9 eval/outputs → Task 8 + Task 9. ✓
- Spec §11 orchestration/SLURM → Task 7 + Task 10. ✓
- Spec §12 execution → Task 11 (smoke) + Task 12 (launch). ✓
- Canonical dual-mask dataflow → Task 5 (step_6a dual mask) + Task 7 (dual-branch step_4a). ✓
```
