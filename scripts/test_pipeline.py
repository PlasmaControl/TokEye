"""Quick end-to-end pipeline test with minimal data.

Runs steps 0a → 0b → 0c → 1a → 2a → 2b → 3a → 3b → 4a
for a single combo (bes/1024_128) to validate the full chain.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_pipeline")

# Make sure imports work
from tokeye.training.big_tf_unet_multiscale.config.modality import (
    ComboConfig,
    ModalityConfig,
    STFTConfig,
)
from tokeye.training.big_tf_unet_multiscale.utils.configuration import (
    load_pipeline_config,
    load_settings,
)
from tokeye.training.big_tf_unet_multiscale.utils.hdf5_io import get_sample_count

config = load_pipeline_config()
cache_dir = Path(config["paths"]["cache_dir"])
test_combo = ComboConfig(
    modality=ModalityConfig(name="bes", input_key="bes", channels=(26, 28, 30, 32, 34, 36, 38, 40)),
    stft=STFTConfig(nfft=1024, hop_length=128),
)
combo_dir = cache_dir / "bes" / "1024_128"


def run_step(name, func, **kwargs):
    logger.info(f"{'='*60}")
    logger.info(f"RUNNING: {name}")
    logger.info(f"{'='*60}")
    try:
        func(**kwargs)
        logger.info(f"OK: {name}")
        return True
    except Exception as e:
        logger.error(f"FAILED: {name}: {e}")
        traceback.print_exc()
        return False


# ── Step 0a: Extract ──
from tokeye.training.big_tf_unet_multiscale import step_0a_extract_faithdata as s0a

step_0a_dir = ComboConfig.shared_dir(cache_dir, "step_0a")
faith_cfg = load_settings(config["paths"]["faith_cfg_path"])

ok = run_step("step_0a", s0a.main, settings={
    "shots_path": Path(config["paths"]["shots_path"]),
    "faith_cfg_path": Path(config["paths"]["faith_cfg_path"]),
    "output_dir": step_0a_dir,
    "overwrite": True,
})
if not ok:
    sys.exit(1)

import os

n_0a = len([f for f in os.listdir(step_0a_dir) if f.endswith(".joblib")])
logger.info(f"  step_0a produced {n_0a} joblib files")
if n_0a == 0:
    logger.error("No files extracted! Check shots.txt and raw_data_dir")
    sys.exit(1)

# ── Step 0b: Preemphasis filter ──
from tokeye.training.big_tf_unet_multiscale import step_0b_filter_faithdata as s0b

step_0b_dir = ComboConfig.shared_dir(cache_dir, "step_0b")
ok = run_step("step_0b", s0b.main, settings={
    "input_dir": step_0a_dir,
    "output_dir": step_0b_dir,
    "preemphasis_coeff": 0.99,
    "overwrite": True,
})
if not ok:
    sys.exit(1)

n_0b = len([f for f in os.listdir(step_0b_dir) if f.endswith(".joblib")])
logger.info(f"  step_0b produced {n_0b} joblib files")

# ── Step 0c: Window into per-modality chunks → HDF5 ──
from tokeye.training.big_tf_unet_multiscale import step_0c_convert_faithdata as s0c

mod_dir = cache_dir / "bes"
mod_dir.mkdir(parents=True, exist_ok=True)
ok = run_step("step_0c", s0c.main, settings={
    "input_dir": step_0b_dir,
    "output_dir": mod_dir,
    "input_key": "bes",
    "input_channels": [26, 28, 30, 32, 34, 36, 38, 40],
    "subseq_len": 66000,
    "frame_info_path": mod_dir / "frame_info.csv",
    "overwrite": True,
})
if not ok:
    sys.exit(1)

h5_0c = mod_dir / "step_0c.h5"
n_0c = get_sample_count(h5_0c) if h5_0c.exists() else 0
logger.info(f"  step_0c produced {n_0c} samples in HDF5")
if n_0c == 0:
    logger.error("No samples in step_0c! Check data flow")
    sys.exit(1)

# ── Step 1a: Copy timeseries ──
from tokeye.training.big_tf_unet_multiscale import step_1a_make_timeseries as s1a

ok = run_step("step_1a", s1a.main, settings={
    "input_h5": h5_0c,
    "output_h5": mod_dir / "step_1a.h5",
    "overwrite": True,
})
if not ok:
    sys.exit(1)

n_1a = get_sample_count(mod_dir / "step_1a.h5")
logger.info(f"  step_1a: {n_1a} samples")

# ── Step 2a: STFT ──
from tokeye.training.big_tf_unet_multiscale import step_2a_make_spectrogram as s2a

combo_dir.mkdir(parents=True, exist_ok=True)
ok = run_step("step_2a", s2a.main, settings={
    "nfft": 1024,
    "hop_length": 128,
    "input_h5": mod_dir / "step_1a.h5",
    "output_h5": combo_dir / "step_2a.h5",
    "overwrite": True,
})
if not ok:
    sys.exit(1)

n_2a = get_sample_count(combo_dir / "step_2a.h5")
logger.info(f"  step_2a: {n_2a} samples")

# ── Step 2b: Baseline filter ──
from tokeye.training.big_tf_unet_multiscale import step_2b_filter_spectrogram as s2b

ok = run_step("step_2b", s2b.main, settings={
    "input_h5": combo_dir / "step_2a.h5",
    "output_h5": combo_dir / "step_2b.h5",
    "output_baseline_h5": combo_dir / "step_2b_baseline.h5",
    "baseline_method": "fabc",
    "baseline_method_kwargs": {"lam": 1e5},
    "bin_cutting": "auto",
    "gradient_threshold": 0.5,
    "overwrite": True,
})
if not ok:
    sys.exit(1)

n_2b = get_sample_count(combo_dir / "step_2b.h5")
logger.info(f"  step_2b: {n_2b} samples")

# ── Step 3a: Correlation analysis (GPU, but fast_dev_run) ──
from tokeye.training.big_tf_unet_multiscale import step_3a_correlation_analysis as s3a

ok = run_step("step_3a", s3a.main, settings={
    "input_h5": combo_dir / "step_2b.h5",
    "output_h5": combo_dir / "step_3a.h5",
    "adjacent_channels": 3,
    "total_channels": 8,
    "clamp_range": "auto",
    "clamp_percentiles": [1, 99],
    "bin_masking": "auto",
    "bin_mask_value": "mean",
    "gradient_threshold": 0.5,
    "num_layers": 5,
    "first_layer_size": 32,
    "batch_size": 4,
    "num_workers": 0,
    "prefetch_factor": None,
    "max_epochs": 1,
    "precision": "32-true",
    "devices": 1,
    "tv_early_stopping": False,
    "tv_patience": 3,
    "enable_progress_bar": True,
    "fast_dev_run": True,
    "ckpt_path": None,
    "log_every_n_steps": 1,
    "overwrite": True,
})
if not ok:
    sys.exit(1)

n_3a = get_sample_count(combo_dir / "step_3a.h5")
logger.info(f"  step_3a: {n_3a} batches")

# ── Step 3b: Extract correlation ──
from tokeye.training.big_tf_unet_multiscale import step_3b_extract_correlation as s3b

ok = run_step("step_3b", s3b.main, settings={
    "input_h5": combo_dir / "step_3a.h5",
    "output_h5": combo_dir / "step_3b.h5",
    "overwrite": True,
})
if not ok:
    sys.exit(1)

n_3b = get_sample_count(combo_dir / "step_3b.h5")
logger.info(f"  step_3b: {n_3b} samples")

# ── Step 4a: Threshold ──
from tokeye.training.big_tf_unet_multiscale import step_4a_threshold as s4a

ok = run_step("step_4a", s4a.main, settings={
    "input_h5": combo_dir / "step_2b_baseline.h5",
    "output_h5": combo_dir / "step_4a.h5",
    "frame_info_csv": mod_dir / "frame_info.csv",
    "threshold_output_path": combo_dir / "thresholds.csv",
    "min_size": "auto",
    "min_size_fraction": 0.0002,
    "remove_bottom_rows": "auto",
    "remove_top_rows": "auto",
    "row_removal_fraction_bottom": 0.01,
    "row_removal_fraction_top": 0.004,
    "overwrite": True,
})
if not ok:
    sys.exit(1)

n_4a = get_sample_count(combo_dir / "step_4a.h5")
logger.info(f"  step_4a: {n_4a} samples")

# ── Summary ──
logger.info(f"\n{'='*60}")
logger.info("TEST PIPELINE COMPLETE")
logger.info(f"{'='*60}")
logger.info(f"  0a: {n_0a} joblib files")
logger.info(f"  0b: {n_0b} joblib files")
logger.info(f"  0c: {n_0c} HDF5 samples")
logger.info(f"  1a: {n_1a} HDF5 samples")
logger.info(f"  2a: {n_2a} HDF5 samples")
logger.info(f"  2b: {n_2b} HDF5 samples")
logger.info(f"  3a: {n_3a} HDF5 batches")
logger.info(f"  3b: {n_3b} HDF5 samples")
logger.info(f"  4a: {n_4a} HDF5 samples")
logger.info("All steps passed!")
