"""Make the ``src/`` layout importable without an editable install.

``pytest tests/`` must work on a fresh clone with nothing installed but the
runtime dependencies -- otherwise the invariant suite is not actually
reproducible for anyone who clones the repository.  CI additionally does
``pip install -e .``; this file is the belt to that braces.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
