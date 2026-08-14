from __future__ import annotations

from xorq.common.compat import StrEnum


XORQ_METADATA_PREFIX = "xorq:"


class BackendName(StrEnum):
    """Backend names that identity/normalization dispatch keys on.

    One canonical spelling per name, so a backend rename cannot leave a stale
    string behind in one dispatch table while the others move (gh-1842 did
    exactly that).  ``test_backend_names.py`` anchors every member to the live
    ``xorq.backends`` entry-point group and holds the full story.  Derived
    dispatch sets live in ``xorq.common.constants``.
    """

    PANDAS = "pandas"
    DUCKDB = "duckdb"
    DATAFUSION = "datafusion"
    XORQ_DATAFUSION = "xorq_datafusion"
    SQLITE = "sqlite"
    BIGQUERY = "bigquery"


class RunLogFile(StrEnum):
    LOG = "run.jsonl"
    META = "meta.json"


class ProvenanceField(StrEnum):
    expr_hash = f"{XORQ_METADATA_PREFIX}expr_hash"
    cache_strategy = f"{XORQ_METADATA_PREFIX}cache_strategy"
    cache_storage = f"{XORQ_METADATA_PREFIX}cache_storage"
    cache_ttl_seconds = f"{XORQ_METADATA_PREFIX}cache_ttl_seconds"
    dasher_rules_fingerprint = f"{XORQ_METADATA_PREFIX}dasher_rules_fingerprint"
    normalize_registry_fingerprint = (
        f"{XORQ_METADATA_PREFIX}normalize_registry_fingerprint"
    )
