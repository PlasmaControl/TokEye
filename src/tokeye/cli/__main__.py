"""Allow ``python -m tokeye.cli`` to behave like the ``tokeye`` script."""

from __future__ import annotations

import sys

from tokeye.cli import main

if __name__ == "__main__":
    sys.exit(main())
