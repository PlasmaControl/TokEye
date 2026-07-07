"""Vendored ``pymodespec`` — classic DIII-D Mirnov mode analysis.

See PROVENANCE.md (upstream, pinned commit, local modifications) and LICENSE
(MIT) in this directory. Heavy submodules (``matplotlib.pyplot`` is imported
at module load) resolve lazily via PEP 562 so importing this package stays
cheap; ``tokeye.cli.modespec`` sets the Agg backend before touching them.
"""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "fetch_mirnov": "modespec",
    "mode_spectrogram": "modespec",
    "mode_fit_timeslice": "modespec",
    "mode_svd_spectrogram": "modespec",
    "plot_modespec": "modespec",
    "plot_svd": "modespec",
    "fetch_ece": "modespec",
    "ece_mode_location": "modespec",
    "load_config": "generate_modes",
    "detect_modes": "generate_modes",
    "run_config": "generate_modes",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        from importlib import import_module

        module = import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
