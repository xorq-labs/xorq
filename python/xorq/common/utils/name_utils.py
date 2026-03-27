import re

import dask  # noqa: PLC0415


def tokenize_to_int(*args) -> int:
    """Derive a deterministic integer from arbitrary args via dask tokenize."""
    return int(dask.base.tokenize(args), 16) % (2**31)


def make_name(prefix, to_tokenize):
    import dask  # noqa: PLC0415

    from xorq.ibis_yaml.config import config  # noqa: PLC0415

    tokenized = dask.base.tokenize(to_tokenize)
    name = f"_{prefix}_{tokenized[: config.hash_length]}".lower()
    return name


def _clean_udf_name(udf_name):
    if udf_name.isidentifier():
        return udf_name
    else:
        return f"fun_{re.sub(r'[^0-9a-zA-Z_]', '_', udf_name)}".lower()


def get_uid_prefix(name, pattern="^(ibis_[\\w-]+_)\\w{26}$"):
    # xorq.vendor.ibis.util.gen_name
    if match := re.match(pattern, name):
        return match.group(1)
    else:
        return None
