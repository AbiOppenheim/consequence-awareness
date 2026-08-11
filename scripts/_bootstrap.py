"""Put src/ on sys.path so scripts run without an editable install.

Import this first in every stage script:  import _bootstrap  # noqa: F401
(Harmless if the package is already installed via `pip install -e .`.)
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
