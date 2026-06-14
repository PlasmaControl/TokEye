"""Map SLURM array index -> ablation variant."""

from __future__ import annotations

from .config.variants import AblationVariant, build_variants

__all__ = ["AblationVariant", "build_variants", "n_variants", "variant_from_index"]


def n_variants(config: dict) -> int:
    return len(build_variants(config))


def variant_from_index(config: dict, index: int) -> AblationVariant:
    vs = build_variants(config)
    if not (0 <= index < len(vs)):
        raise IndexError(f"variant index {index} out of range 0..{len(vs) - 1}")
    return vs[index]
