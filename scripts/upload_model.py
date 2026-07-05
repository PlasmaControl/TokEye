"""Publish a verified TokEye checkpoint to the Hugging Face Hub.

Run this ONCE per checkpoint update, by a maintainer with write access to the
target Hub repo. `tokeye.hub.download_model` is what every user's install
pulls weights from — this script's verification gate is the only thing
standing between a bad artifact and a broken install for all of them.

Prereq: authenticate first, either via `hf auth login` or by setting the
`HF_TOKEN` environment variable, with write access to the target org/repo.

Usage:

    uv run python scripts/upload_model.py --create
    uv run python scripts/upload_model.py --model big_tf_unet \
        --repo PlasmaControl/tokeye

That is equivalent to the raw one-liner:

    hf upload PlasmaControl/tokeye model/big_tf_unet_251210.pt \
        big_tf_unet_251210.pt

...except that `hf upload` does not check that the file loads into the
architecture `tokeye.hub` expects. This script's added value is the
verification gate in `verify_checkpoint`: it refuses to upload anything that
does not load, strictly, into the registered model class and survive a tiny
forward pass. That is what protects the repo from an accidental upload of a
file like `model/big_tf_unet_251210_weights.pt`, which uses old
(`in_conv.double_conv.*`-style) key names and would silently break every
downstream user.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, LocalTokenNotFoundError

from tokeye.hub import DEFAULT_MODEL, DEFAULT_REPO_ID, MODEL_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable

_PROBE_SHAPE = (1, 1, 64, 64)


def verify_checkpoint(path: Path, builder: Callable[[], nn.Module]) -> None:
    """Refuse (via ``SystemExit``) to proceed unless ``path`` is a good checkpoint.

    "Good" means: a weights-only state dict that loads strictly into a fresh
    instance of the architecture ``builder`` returns, and survives a tiny
    forward pass. This is the safety gate the whole script exists for.
    """
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(
            f"error: could not load {path} as a weights-only state dict "
            f"({exc}).\nIf this is a pickled full nn.Module (a legacy "
            "checkpoint), it must be converted to a plain state_dict before "
            "upload. Refusing to upload."
        ) from exc

    if not isinstance(state_dict, Mapping):
        raise SystemExit(
            f"error: {path} loaded as a {type(state_dict).__name__}, not a "
            "state_dict mapping. Refusing to upload."
        )

    model = builder()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        mismatches = "\n".join(str(exc).splitlines()[:8])
        raise SystemExit(
            f"error: {path} does not match the {type(model).__name__} "
            f"architecture. Refusing to upload.\n{mismatches}"
        ) from exc

    model.eval()
    with torch.no_grad():
        try:
            model(torch.randn(*_PROBE_SHAPE))
        except Exception as exc:
            raise SystemExit(
                f"error: {path} loaded but failed a forward-pass sanity "
                f"check ({exc}). Refusing to upload."
            ) from exc

    print(f"verified: {path} matches {type(model).__name__} and runs forward pass.")


def _resolve_file(model_name: str, file_arg: str | None) -> Path:
    if file_arg is not None:
        return Path(file_arg)
    return Path("model") / MODEL_REGISTRY[model_name].filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="upload_model.py",
        description=(
            "Verify a TokEye checkpoint against its registered architecture, "
            "then publish it to the Hugging Face Hub."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=sorted(MODEL_REGISTRY),
        help="Registry entry to publish (default: %(default)s).",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Local checkpoint to upload (default: model/<registry filename>).",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO_ID,
        help="Target Hugging Face Hub repo id (default: %(default)s).",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the repo first (HfApi().create_repo(repo, exist_ok=True)).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = MODEL_REGISTRY[args.model]
    file_path = _resolve_file(args.model, args.file)

    if not file_path.exists():
        raise SystemExit(f"error: checkpoint not found: {file_path}")

    verify_checkpoint(file_path, spec.builder)

    api = HfApi()
    try:
        if args.create:
            api.create_repo(args.repo, exist_ok=True)
        commit_info = api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=spec.filename,
            repo_id=args.repo,
            commit_message=f"Upload {spec.filename} (tokeye model={args.model!r})",
        )
    except (HfHubHTTPError, LocalTokenNotFoundError) as exc:
        raise SystemExit(
            f"error: Hugging Face Hub upload failed ({exc}).\n"
            "Run `hf auth login` (or set the HF_TOKEN env var) with write "
            f"access to {args.repo!r} and try again."
        ) from exc

    print(f"uploaded: {commit_info.commit_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
