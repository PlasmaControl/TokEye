# modespec/deep — reserved for the next-generation mode-number engine

Classic modespec (`../classic`) needs a spatial probe array (toroidal Mirnov
set) to fit mode numbers. The "deep" engine aims to recover toroidal mode
numbers from a *single* line-integrated chord (the DIII-D CO2 interferometer)
using physics side channels instead of spatial phase fits: multitaper
spectrograms, calibrated line detection (no free parameters), per-chord track
building, and harmonic-comb families (a rotating island's harmonics at
`k * f0` carry `n = k * n1`).

That work lives in the sibling project `integratedmode`
(`/scratch/gpfs/nc1514/integratedmode` — see its `CLAUDE.md` and
`docs/specs/2026-07-01-calibrated-detection-and-n-inference.md`). It stays
there until the analysis is validated; this directory only reserves the
integration point. Intended CLI shape once it lands:

    tokeye modespec --engine deep <config.yaml>

(`--engine classic` stays the default; no breaking changes to the classic
config format.)
