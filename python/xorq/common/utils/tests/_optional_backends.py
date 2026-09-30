"""Backend classes whose import pulls an optional extra, resolved lazily.

A test module that imports a backend at module scope is collected by EVERY CI
job, because the matrix runs ``pytest -m <backend>`` with no path filter and
selection happens after collection. ``xorq.backends.redshift`` reaches
``vendor/ibis/backends/postgres``, which imports ``psycopg`` unguarded, and
``psycopg`` lives in the ``postgres`` extra -- so nine of the eleven matrix jobs
hit ``ModuleNotFoundError`` at collection and went red without running a thing.

``backends/conftest.py`` already solves this for modules under
``backends/<name>/`` by skipping the directory. Modules deliberately sited
outside that tree -- because a backend marker would confine them to the
credential-gated workflow -- get no such cover, and a whole-module
``importorskip`` is wrong for a file like ``test_dasher.py`` whose other
hundred tests must keep running everywhere.

Hence this: the fqn as a plain string, always available, and the class only
where its driver is installed.
"""

from __future__ import annotations

import importlib
import importlib.util


# The rule key registered in ``dasher._EXTRA_RULES``. A string, so the drift
# guard can name what it could not check without importing anything.
REDSHIFT_BACKEND_FQN = "xorq.backends.redshift.Backend"


def redshift_backend() -> type | None:
    """The Redshift backend class, or ``None`` when ``psycopg`` is absent.

    ``find_spec`` rather than a ``try``/``ImportError``: an ImportError raised
    from *inside* the backend module for an unrelated reason would otherwise be
    swallowed and reported as "extra not installed", which is the same
    conflation of absence with unreadability that the Redshift freshness probe
    itself exists to avoid.
    """
    if importlib.util.find_spec("psycopg") is None:
        return None
    return importlib.import_module("xorq.backends.redshift").Backend
