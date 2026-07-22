# scripts/

- **`usage/`** — portable demo notebooks (`big_tf_unet.ipynb`,
  `ae_tf_maskrcnn.ipynb`). Run anywhere: model weights auto-download from
  Hugging Face on first use, and the input is a generated synthetic signal
  (`tokeye.examples.make_example_signal`) — no cluster paths or local
  checkpoints required.

- **`commands/ablation/`** — SLURM scripts driving the
  `big_tf_unet_ablation` training runs on Princeton's della cluster (see
  `src/tokeye/training/big_tf_unet_ablation/README.md`). Site-specific
  (`$SCRATCH`, della partitions); not expected to run elsewhere.

- **`upload_model.py`** — maintainer-only tool for publishing a verified
  checkpoint to the Hugging Face Hub. See its docstring before running it.
