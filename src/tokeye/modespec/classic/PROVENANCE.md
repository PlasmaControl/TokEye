# Vendored code provenance

- Upstream: `git@github.com:PlasmaControl/pymodespec.git` (private)
- Pinned commit: `1e0e48fc6a32f2eccbf1a88a7c21ce8480f1db8d`
- Vendored: 2026-07-06
- License: MIT (see `LICENSE`, copied from upstream)

## Files taken

All eight Python modules plus the example config and license:
`modespec.py`, `generate_modes.py`, `data_utils.py`, `ece_coherence.py`,
`ece_ms_zoom.py`, `ece_sawteeth.py`, `mpi_coherence.py`, `mre_utils.py`,
`modes.yaml`, `LICENSE`. Upstream notebooks and pixi files were not vendored.

## Local modifications

- `generate_modes.py`: `from modespec import ...` made relative
  (`from .modespec import ...`); `main()` body extracted into
  `run_config(config_path) -> int` (returns failed-shot count) so the
  `tokeye modespec` subcommand can call it.
- `modespec.py` (4 sites) and `data_utils.py` (1 site): the
  "MDSplus not available" errors now explain where MDSplus comes from
  (GA cluster / conda-forge) and that fetching needs atlas.gat.com access.
- `__init__.py` and this file are additions, not upstream files.
- Style rules are relaxed for this directory in `ruff.toml`
  (vendored-code policy); correctness rules (F821 etc.) remain active.

Re-vendoring: clone upstream at a newer commit, re-copy the files above,
re-apply the modifications in this list, and update the pinned commit.
