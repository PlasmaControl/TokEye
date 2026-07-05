"""HPC-safe import guarantees.

``tokeye.batch`` must stay importable without pulling in gradio (it needs to
run on HPC login/compute nodes that may not have a display or gradio
installed). ``tokeye.cli`` must stay importable without pulling in gradio
*or* torch, so plain ``tokeye --help`` returns instantly.

These checks run in a subprocess: importing the modules in-process (as the
rest of the test suite does throughout the run) would leave the relevant
modules in ``sys.modules`` regardless of what any single import statement
pulls in, masking a regression.
"""

from __future__ import annotations

import subprocess
import sys


def test_batch_import_does_not_pull_in_gradio():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tokeye.batch, sys; "
            "assert 'gradio' not in sys.modules; "
            "print('ok')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_package_import_does_not_pull_in_torch():
    """``from tokeye import TokEye`` is lazy (PEP 562): the bare package
    import must stay torch-free or ``tokeye --help`` slows to a crawl."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tokeye, sys; "
            "assert 'torch' not in sys.modules; "
            "print('ok')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_cli_import_does_not_pull_in_gradio_or_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tokeye.cli, sys; "
            "assert 'gradio' not in sys.modules; "
            "assert 'torch' not in sys.modules; "
            "print('ok')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
