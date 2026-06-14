from __future__ import annotations

from tokeye.training.big_tf_unet_ablation.ablation_matrix import (
    n_variants,
    variant_from_index,
)
from tokeye.training.big_tf_unet_ablation.config.variants import build_variants

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
