from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


mark_slow = pytest.mark.slow
