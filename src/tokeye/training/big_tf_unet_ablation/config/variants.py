"""Ablation variant definitions and enumeration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationVariant:
    id: str
    baseline: bool  # apply ALS broadband-coherent separation to coherent path
    denoise: bool  # run multichannel self-supervised denoiser
    representation: str  # "complex" | "magnitude" (denoiser input)

    def __post_init__(self) -> None:
        if self.representation not in ("complex", "magnitude"):
            raise ValueError(f"bad representation: {self.representation}")


def build_variants(config: dict) -> list[AblationVariant]:
    return [
        AblationVariant(
            id=v["id"],
            baseline=bool(v["baseline"]),
            denoise=bool(v["denoise"]),
            representation=str(v["representation"]),
        )
        for v in config["variants"]
    ]
