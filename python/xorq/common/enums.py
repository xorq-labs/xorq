from xorq.common.compat import StrEnum


XORQ_METADATA_PREFIX = "xorq:"


class RunLogFile(StrEnum):
    LOG = "run.jsonl"
    META = "meta.json"


class ProvenanceField(StrEnum):
    expr_hash = f"{XORQ_METADATA_PREFIX}expr_hash"
    cache_strategy = f"{XORQ_METADATA_PREFIX}cache_strategy"
    cache_storage = f"{XORQ_METADATA_PREFIX}cache_storage"
    cache_ttl_seconds = f"{XORQ_METADATA_PREFIX}cache_ttl_seconds"
