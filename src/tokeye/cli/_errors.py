"""Shared error reporting for CLI subcommands."""

from __future__ import annotations

import sys


def print_hub_error(name: str, exc: Exception) -> None:
    """Print a friendly message for a failed Hugging Face model download."""
    from tokeye.hub import repo_for

    print(
        f"error: could not download model {name!r} from Hugging Face "
        f"repo {repo_for(name)!r}: {exc}. If the repo has moved, set "
        "TOKEYE_HF_REPO to override.",
        file=sys.stderr,
    )
