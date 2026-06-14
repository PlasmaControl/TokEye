"""Modality definitions for the ablation pipeline.

The ablation trains on all four DIII-D fluctuation diagnostics (matching the
paper). Each modality has its own channel set; label-generation runs per
modality and step_6a combines them into one training set.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Modality:
    name: str
    input_key: str
    channels: tuple[int, ...]

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def total_channels(self) -> int:
        return self.num_channels

    @property
    def adjacent_channels(self) -> int:
        # neighbours used on each side by the denoiser; mirrors multiscale (num//2)
        return max(1, self.num_channels // 2)


def build_modalities(config: dict) -> list[Modality]:
    mods = config["modalities"]
    return [
        Modality(name=name, input_key=spec["input_key"], channels=tuple(spec["channels"]))
        for name, spec in mods.items()
    ]


def modality_from_index(config: dict, index: int) -> Modality:
    mods = build_modalities(config)
    if not (0 <= index < len(mods)):
        raise IndexError(f"modality index {index} out of range 0..{len(mods) - 1}")
    return mods[index]
