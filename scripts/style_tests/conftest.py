"""Put this directory on `sys.path` so `test_style_gate` can import `fixtures`.

`--import-mode=importlib`, which .github/workflows/ci-test.yml passes, imports a
test module without adding its directory to `sys.path`, so a sibling helper is
unimportable by bare name. The default `prepend` mode adds the directory but not
the repo root, so an absolute `scripts.style_tests.fixtures` fails there instead.
A conftest is loaded first under both modes, so doing it here is the one spelling
that works in both, and for a bare `pytest scripts/style_tests` by hand.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
