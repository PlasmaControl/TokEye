"""TokEye: classification and localization of fluctuating signals.

``from tokeye import TokEye`` is the one-import Python API. The class is
resolved lazily (PEP 562) so ``import tokeye`` — and therefore the CLI —
stays free of torch/gradio imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokeye.api import TokEye

__all__ = ["TokEye"]


def __getattr__(name: str) -> type[TokEye]:
    if name == "TokEye":
        from tokeye.api import TokEye

        return TokEye
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
