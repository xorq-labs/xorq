"""One deliberately broken file per style rule.

Each entry is the smallest source that rule must flag, keyed by rule id. The
paths are relative to a throwaway project root and matter: several rules key off
the filename (`enums.py`, `exceptions.py`, `test_*.py`) or off where the file
sits relative to `src-roots`, so a fixture laid out the way `src-roots` names is
what makes a misconfigured root visible.
"""

__all__ = ["FIXTURES", "SUPPORT"]


# The module `unlisted-import` resolves against; never checked itself.
SUPPORT: dict[str, str] = {
    "python/pkg/provider.py": "__all__ = ['shown']\n\nshown = 1\nhidden = 2\n",
}


FIXTURES: dict[str, tuple[str, str]] = {
    "relative-import": (
        "python/pkg/mod.py",
        "from . import sibling\n",
    ),
    "test-class": (
        "python/pkg/test_thing.py",
        "class TestThing:\n    def test_x(self):\n        pass\n",
    ),
    "deferred-import-test": (
        "python/pkg/test_deferred.py",
        "def test_x():\n    import os\n\n    return os\n",
    ),
    "deferred-stdlib": (
        "python/pkg/deferred.py",
        "def f():\n    import os\n\n    return os\n",
    ),
    "os-environ": (
        "python/pkg/env.py",
        "import os\n\n\nX = os.environ['HOME']\n",
    ),
    "future-annotations": (
        "python/pkg/future.py",
        "def f(a: int) -> int:\n    return a\n",
    ),
    "os-path": (
        "python/pkg/paths.py",
        "import os\n\n\nX = os.path.join('a', 'b')\n",
    ),
    "dataclasses": (
        "python/pkg/dc.py",
        "from dataclasses import dataclass\n\n\n@dataclass\nclass C:\n    x: int\n",
    ),
    "cache-method": (
        "python/pkg/cache.py",
        "import functools\n\n\nclass C:\n    @functools.cache\n    def m(self):\n        return 1\n",
    ),
    # Not exceptions.py: the rule skips that module outright, on the grounds
    # that a project's own base class has to inherit from something.
    "exception-hierarchy": (
        "python/pkg/errors.py",
        "class MyError(Exception):\n    pass\n",
    ),
    "redundant-import": (
        "python/pkg/redundant.py",
        "import os\n\n\ndef f():\n    import os\n\n    return os\n",
    ),
    "print": (
        "python/pkg/printer.py",
        "def f():\n    print('hi')\n",
    ),
    "type-annotations": (
        "python/pkg/untyped.py",
        "def f(a, b):\n    return a + b\n",
    ),
    # The rule looks for a `default=` on an attrs field call, not for a mutable
    # annotation assignment.
    "attrs-mutable-default": (
        "python/pkg/attrsy.py",
        "import attrs\n\n\n@attrs.define\nclass C:\n    x = attrs.field(default=[])\n",
    ),
    "protected-access": (
        "python/pkg/protected.py",
        "def f(obj):\n    return obj._secret\n",
    ),
    "pytest-param-id": (
        "python/pkg/test_param.py",
        "import pytest\n\n\n@pytest.mark.parametrize('a', [1, 2])\ndef test_x(a):\n    pass\n",
    ),
    "pytest-mark-qualify": (
        "python/pkg/test_mark.py",
        "from pytest import mark\n\n\n@mark.skip\ndef test_x():\n    pass\n",
    ),
    "stdlib-logging": (
        "python/pkg/logs.py",
        "import logging\n\n\nlog = logging.getLogger(__name__)\n",
    ),
    "pytest-tmp-path": (
        "python/pkg/test_tmp.py",
        "def test_x(tmpdir):\n    pass\n",
    ),
    "import-aliasing": (
        "python/pkg/aliased.py",
        "import os as _os\n\n\nX = _os\n",
    ),
    "strenum-compat": (
        "python/pkg/strenums.py",
        "from enum import StrEnum\n\n\nclass C(StrEnum):\n    A = 'a'\n",
    ),
    "enum-placement": (
        "python/pkg/notenums.py",
        "import enum\n\n\nclass Color(enum.Enum):\n    RED = 1\n",
    ),
    "exception-placement": (
        "python/pkg/notexceptions.py",
        "class MyError(Exception):\n    pass\n",
    ),
    "leaf-enum-import": (
        "python/pkg/enums.py",
        "import attrs\n\n\nX = attrs\n",
    ),
    # Resolving `pkg.provider` to a file is what src-roots controls, so this is
    # the fixture that goes dark when the roots are wrong.
    "unlisted-import": (
        "python/pkg/consumer.py",
        "from pkg.provider import hidden\n\n\nX = hidden\n",
    ),
    "init-reexport": (
        "python/pkg/reexport.py",
        "import os\n\n\n__all__ = ['os']\n",
    ),
}
