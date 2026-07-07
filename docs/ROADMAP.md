# TokEye roadmap — toward the go-to mode-analysis tool

TokEye's goal is to cover the jobs DIII-D researchers currently spread across
separate tools (modespec, ad-hoc ELM scripts, per-group AE workflows), so one
install answers "what modes are in this shot?". This file tracks the suite and
collects future ideas worth building.

## Suite status (0.12.0)

| Tool | Command | Status |
|------|---------|--------|
| Segmentation | `tokeye run`, `tokeye app` | shipped (big_tf_unet) |
| modespec (classic) | `tokeye modespec <config.yaml>` | shipped — vendored pymodespec (Mirnov n-number fits; needs MDSplus or a cache) |
| modespec (deep) | `tokeye modespec --engine deep` | reserved — single-chord CO2 n-inference, developed in the sibling `integratedmode` project |
| elmspec | `tokeye elmspec INPUTS...` | shipped — ELM events from the transient channel |
| alfvenspec | `tokeye alfvenspec INPUTS...` | shipped, deliberately thin — ae_tf_maskrcnn boxes/masks; awaiting EP-group requirements |
| eigspec | — | gated — vendoring blocked until upstream relicenses GPL-3 → MIT; also needs import fixes (see task notes) |
| modesearch | `tokeye modesearch` | design stage — prints the plan |

## Near-term engineering

- **Upload `ae_tf_maskrcnn` weights to `nc1/ae_tf_maskrcnn`** (registry entry
  and upload-script probe are in place; needs a write-scoped HF token).
- **Vendor eigspec** once relicensed: modal identification, stochastic
  subspace ID, random-projection spectral analysis — the system-ID view of
  mode activity that complements matched-filter fits.
- **AE weights provenance**: score calibration and a labeled validation set
  for alfvenspec before promoting it beyond "runs the model".

## Mode catalogue schema (the keystone)

A single record type that every detector emits, so downstream tools compose:

    shot, machine, diagnostic, t_start, t_end, f_low, f_high,
    n (nullable), m (nullable), amplitude, confidence,
    detector, detector_version, artifact_ref

- `big_tf_unet` masks → connected regions → records (coherent/transient class)
- `modespec` CSV rows → records with `n` filled
- `elmspec` events → transient records tagged ELM
- `alfvenspec` boxes → records tagged AE

Once this exists, modesearch is "crawler + storage + filters" rather than a
research project. It also gives papers a uniform unit of comparison across
detectors.

## modesearch build-out

1. Crawler: batch job over shot archives (local HDF5 first; MDSplus/toksearch
   where reachable) running the suite and emitting catalogue records.
2. Storage: start boring — one parquet/SQLite per campaign; revisit only if
   query load demands it.
3. Query CLI: `tokeye modesearch find --n 2 --f 2e3:4e3 --no-elm` → shot list
   with matching events.
4. Consumers: the fusion-world-model shot designer learns mode-occurrence
   statistics conditioned on plasma parameters; shotsearch intersection
   ("shots near this setup that developed a locked mode").

## Ideas that would be extremely useful to mode researchers

- **Mode-number labeling of TokEye masks.** Fuse modespec n-fits with U-Net
  regions: overlap a mask region with the (t, f) support of an n-fit and the
  region inherits the mode number. Turns "coherent activity" into "n=2 TM",
  which is what people actually search for.
- **Mode trajectory tracking.** Follow a detected mode's (f, amplitude, n)
  through time: frequency chirps, mode locking (f → 0), rotation braking.
  Locked-mode precursors as a first-class query.
- **Cross-diagnostic confirmation.** The same mode seen on Mirnov, CO2, ECE,
  and BES with consistent frequency is real; single-diagnostic detections get
  a lower confidence. The catalogue schema's `diagnostic` field enables this.
- **ELM database.** elmspec over campaigns → ELM frequency/size statistics vs
  pedestal parameters; ELM-free-window finder for AE/TM studies.
- **AE taxonomy.** Classify alfvenspec detections (TAE/RSAE/EAE/BAE) from
  frequency-vs-time shape and q-profile context — the EP group's actual need;
  gather their requirements before building.
- **Sawtooth/MRE integration.** The vendored classic tree already carries ECE
  sawtooth and MRE helpers (`ece_sawteeth.py`, `mre_utils.py`); surface them
  as first-class detectors emitting catalogue records.
- **Inter-shot mode.** A between-shots summary (30 s budget): run the suite on
  the last shot, print/annotate the mode inventory for the control room.
- **Cross-machine record.** TJ-II validation already exists for the U-Net;
  keep the catalogue schema machine-agnostic so C-Mod/NSTX-U/MAST-U archives
  can be crawled without schema surgery.
- **Confidence calibration.** Per-detector reliability curves (detected vs
  human-labeled) so catalogue confidences are comparable across detectors —
  prerequisite for any world-model consumer treating them as probabilities.
- **OMFIT/toksearch hooks.** Thin adapters so existing GA workflows can call
  `tokeye.api.TokEye` and the suite CLIs without leaving their environment.

## Non-goals (for now)

Real-time control integration (inter-shot is the nearer target), automatic
retraining pipelines, and cross-machine transfer learning beyond what the
existing TJ-II validation demonstrates.
