from __future__ import annotations

import re
from typing import Any


def tokenize_to_int(*args: Any) -> int:
    """Derive a deterministic integer from arbitrary args via dask tokenize."""
    import dask  # noqa: PLC0415

    return int(dask.base.tokenize(args), 16) % (2**31)


def make_name(prefix: str, to_tokenize: Any) -> str:
    import dask  # noqa: PLC0415

    from xorq.ibis_yaml.config import config  # noqa: PLC0415

    tokenized = dask.base.tokenize(to_tokenize)
    name = f"_{prefix}_{tokenized[: config.hash_length]}".lower()
    return name


def _clean_udf_name(udf_name: str) -> str:
    if udf_name.isidentifier():
        return udf_name
    else:
        return f"fun_{re.sub(r'[^0-9a-zA-Z_]', '_', udf_name)}".lower()


def get_uid_prefix(name: str, pattern: str = "^(ibis_[\\w-]+_)\\w{26}$") -> str | None:
    # xorq.vendor.ibis.util.gen_name
    if match := re.match(pattern, name):
        return match.group(1)
    else:
        return None
