"""End-to-end smoke test for the ablation pipeline (tiny, fast).

Builds a smoke config (2 cached shots, <=2 windows/shot, 2 folds, 1 epoch, no
torch.compile, cleanup disabled), runs the shared prefix, all 4 variants, and the
TJ-II eval, then asserts the key artifacts exist. Run on the V100 head node:

    CUDA_VISIBLE_DEVICES=0 python scripts/test_ablation_smoke.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path("/scratch/gpfs/nc1514/tokeye")
sys.path.insert(0, str(ROOT / "src"))

from tokeye.training.big_tf_unet_ablation import orchestrator as orch  # noqa: E402
from tokeye.training.big_tf_unet_ablation.ablation_matrix import (
    n_variants,  # noqa: E402
)
from tokeye.training.big_tf_unet_ablation.task_matrix import TaskMatrix  # noqa: E402

SMOKE_YAML = ROOT / ".tmp" / "ablation_smoke.yaml"
BASE_YAML = ROOT / "src/tokeye/training/big_tf_unet_ablation/config/ablation.yaml"

OVERRIDES = {
    "window_filter": {"max_windows_per_shot": 2, "min_activity": 0.0},
    "correlation": {"max_epochs": 1, "compile": False, "batch_size": 4,
                    "tv_patience": 1, "num_workers": 2},
    "refiner": {"n_folds": 2, "max_epochs": 1, "mc_dropout_samples": 2,
                "batch_size": 8, "num_workers": 2},
    "final": {"n_folds": 2, "max_epochs": 1, "batch_size": 8, "num_workers": 2},
    "paths": {
        "cache_dir": "data/cache/ablation_smoke",
        "model_dir": "model/ablation_smoke",
        "task_matrix_path": "data/cache/ablation_smoke/task_matrix.json",
    },
    "cleanup": {"enabled": False},
}


def _build_smoke_config() -> Path:
    base = OmegaConf.load(BASE_YAML)
    merged = OmegaConf.merge(base, OmegaConf.create(OVERRIDES))
    SMOKE_YAML.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(merged, SMOKE_YAML)
    return SMOKE_YAML


def main() -> int:
    # fresh smoke dirs
    for d in (ROOT / "data/cache/ablation_smoke", ROOT / "model/ablation_smoke"):
        if d.exists():
            shutil.rmtree(d)

    smoke_yaml = _build_smoke_config()
    cfg = OmegaConf.to_container(OmegaConf.load(smoke_yaml), resolve=True)
    tm = TaskMatrix(cfg["paths"]["task_matrix_path"])

    print("=== SHARED STEPS ===")
    orch.run_shared_steps(cfg, tm, ["step_0c", "step_2a", "step_2f"])

    print("=== VARIANTS ===")
    for vidx in range(n_variants(cfg)):
        print(f"--- variant {vidx} ---")
        orch.run_variant_steps(cfg, tm, vidx, orch.VARIANT_STEPS)

    # assert fold models exist for every variant
    model_root = ROOT / cfg["paths"]["model_dir"]
    from tokeye.training.big_tf_unet_ablation.ablation_matrix import build_variants

    ok = True
    for v in build_variants(cfg):
        ckpts = list((model_root / v.id).glob("fold_*/final.torchscript.pt"))
        print(f"  {v.id}: {len(ckpts)} fold torchscript models")
        if len(ckpts) < 2:
            ok = False
            print(f"  !! expected >=2 fold models for {v.id}")

    print("=== EVAL ===")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tjii_ablation_eval", ROOT / "scripts/eval/TJII2021_ablation.py"
    )
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    ev.main(str(smoke_yaml))
    out_csv = ROOT / "data/eval/results/TJII2021_ablation.csv"
    print(f"eval csv exists: {out_csv.exists()}")
    ok = ok and out_csv.exists()

    print("\nSMOKE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
