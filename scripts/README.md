# scripts/

- **`big_tf_unet_2/`** — the run-folder **template** for the current
  single-scale teacher pipeline. Copy it to start a run; the copy's directory
  name becomes the run id:

  ```bash
  cp -r scripts/big_tf_unet_2 dev/training/nfft512_hop128
  cd dev/training/nfft512_hop128 && sbatch step_0_intake.sh
  ```

  Flat: one shared `run.yml` plus a `.yml`/`.sh`/`.ipynb` triplet per step —
  knobs, submission, read-only diagnosis. One `sbatch` per step, nothing
  chained. See its own `README.md` (which travels with the copy) and
  `src/tokeye/training/big_tf_unet_2/README.md`. Site-specific: each `.sh`
  hardcodes the repo root in `#SBATCH --chdir`.

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
