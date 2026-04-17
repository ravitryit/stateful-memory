from __future__ import annotations

import sys
from pathlib import Path


# Make `import hydradb_plus` work when pytest runs from inside the package root.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PACKAGE_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

