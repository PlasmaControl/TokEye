from __future__ import annotations

from tokeye.training.big_tf_unet_ablation.config.modalities import Modality
from tokeye.training.big_tf_unet_ablation.config.variants import AblationVariant
from tokeye.training.big_tf_unet_ablation.orchestrator import (
    build_variant_step_settings,
)

CFG = {
    "stft": {"nfft": 1024, "hop_length": 128},
    "paths": {"cache_dir": "data/cache/ablation", "model_dir": "model/ablation"},
    "modalities": {
        "bes": {"input_key": "bes", "channels": [26, 28, 30, 32, 34, 36, 38, 40]},
    },
    "baseline": {"enabled": True},
    "correlation": {"first_layer_size": 32},
    "threshold": {},
    "refiner": {"n_folds": 5},
    "final": {"n_folds": 5},
}
BES = Modality(name="bes", input_key="bes", channels=(26, 28, 30, 32, 34, 36, 38, 40))


def _v(id, base, den, rep):
    return AblationVariant(id=id, baseline=base, denoise=den, representation=rep)


def test_coherent_threshold_reads_denoised_when_denoise_on():
    s = build_variant_step_settings(CFG, _v("full", True, True, "complex"), "step_4a_coh", BES)
    assert s["input_h5"].name == "step_3b.h5"


def test_coherent_threshold_reads_step2b_when_denoise_off():
    s = build_variant_step_settings(CFG, _v("nodenoise", True, False, "complex"), "step_4a_coh", BES)
    assert s["input_h5"].name == "step_2b.h5"


def test_transient_threshold_always_reads_baseline():
    s = build_variant_step_settings(CFG, _v("full", True, True, "complex"), "step_4a_tra", BES)
    assert s["input_h5"].name == "step_2b_baseline.h5"


def test_baseline_flag_propagates_to_step2b():
    s = build_variant_step_settings(CFG, _v("nobaseline", False, True, "complex"), "step_2b", BES)
    assert s["baseline_enabled"] is False


def test_representation_propagates_to_step3a():
    s = build_variant_step_settings(CFG, _v("mag", True, True, "magnitude"), "step_3a", BES)
    assert s["representation"] == "magnitude"


def test_step6a_modality_inputs_have_three_h5_each():
    s = build_variant_step_settings(CFG, _v("full", True, True, "complex"), "step_6a")
    assert len(s["modality_inputs"]) == 1
    mi = s["modality_inputs"][0]
    assert {"name", "img_h5", "coh_h5", "tra_h5"} <= set(mi)
    assert mi["coh_h5"].endswith("step_4a_threshold.h5")
    assert mi["tra_h5"].endswith("step_4a_threshold_baseline.h5")


def test_step6d_reads_refined_and_writes_model_dir():
    s = build_variant_step_settings(CFG, _v("full", True, True, "complex"), "step_6d")
    assert s["input_dir"].name == "step_6c"
    assert s["model_dir"].name == "full"
