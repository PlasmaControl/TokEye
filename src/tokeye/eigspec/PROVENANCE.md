# Vendored code provenance

- Upstream: `git@github.com:PlasmaControl/eigspec.git` (public)
- Pinned commit: `923ad5da97ed8c69cf83c99e0163510c6abe20c2` ("Add MIT License to the project")
- Vendored: 2026-07-06
- License: MIT (see `LICENSE.md`, copied from upstream)

## Files taken

Everything under upstream `src/eigspec/` (the package code: `analysis/`,
`io/`, `utils/`, `vis/`, `cli.py`, `__init__.py`) plus `LICENSE.md`.
Upstream `matlab/`, `assets/`, `demo/`, `examples/`, `docs/`, tests, and
figure PNGs were not vendored.

## Local modifications

Upstream could not be imported at all — `lambda` used as an attribute name is
a hard SyntaxError, and several annotations referenced names that were never
imported. Fixes:

- `utils/data_extraction.py` (2 sites) and `vis/spectral_plots.py` (5 sites):
  `block.mrep.m0.lambda` → `getattr(..., 'lambda')`.
- `utils/subspace_identification.py`: added the missing
  `Any, Dict, Tuple, Union` typing imports and `import numpy.typing as npt`
  (annotations referenced them → `NameError` at import on Python 3.13).
- `utils/subspace_identification.py` — two numeric bugs in
  `covariance_driven_ssi` (worth upstreaming):
  1. The block-Hankel matrix was flattened channel-major
     (`data_block.T.flatten()`), but every downstream slice
     (`[:m*p]`, `[m:m*f]`, `[:m*(f-1)]`) assumes time-block-major rows like
     the MATLAB original — past/future blocks were scrambled and recovered
     pole frequencies were wrong (e.g. 2x for a 2-channel sin/cos pair).
     Fixed by flattening the (time, channel) block in C order.
  2. The `A = O1 \\ O2` step carried a spurious `.T` ("transpose to match
     MATLAB"): `np.linalg.lstsq(O1, O2)[0]` is already `pinv(O1) @ O2`,
     identical to backslash. Eigenvalues survive a transpose but mode
     shapes do not. Removed (both single- and multi-order paths).
  After the fixes, a damped-sinusoid test recovers both the pole frequency
  and the damping ratio to <1% (see tests/test_eigspec.py). The sibling
  `canonical_correlation_ssi` / `ssi1ca` / `ssicca` functions were NOT
  audited for the same Hankel-layout issue.
- `PROVENANCE.md` (this file) is an addition, not an upstream file.
- Style rules are relaxed for this directory in `ruff.toml`
  (vendored-code policy); correctness rules (F821 etc.) remain active.

Re-vendoring: clone upstream at a newer commit, re-copy `src/eigspec/*`,
re-apply (or upstream) the fixes above, and update the pinned commit.
