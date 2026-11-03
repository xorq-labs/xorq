import re


def tokenize_to_int(*args) -> int:
    """Derive a deterministic integer from arbitrary args."""
    from xorq.common.utils.dasher import tokenize  # noqa: PLC0415

    return int(tokenize(args), 16) % (2**31)


def make_name(prefix, to_tokenize):
    from xorq.common.utils.dasher import tokenize  # noqa: PLC0415
    from xorq.ibis_yaml.config import config  # noqa: PLC0415

    tokenized = tokenize(to_tokenize)
    name = f"_{prefix}_{tokenized[: config.hash_length]}".lower()
    return name


def _clean_udf_name(udf_name):
    if udf_name.isidentifier():
        return udf_name
    else:
        return f"fun_{re.sub(r'[^0-9a-zA-Z_]', '_', udf_name)}".lower()


def get_uid_prefix(name, pattern="^(ibis_[\\w-]+_)\\w{14,26}$"):
    # xorq.vendor.ibis.util.gen_name (26-char uid), or xorq.expr.relations.gen_name,
    # which truncates the whole name to 35 chars (14-char uid)
    if match := re.match(pattern, name):
        return match.group(1)
    else:
        return None
