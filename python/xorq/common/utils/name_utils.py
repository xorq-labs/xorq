import re


def make_name(prefix, to_tokenize):
    import dask

    from xorq.ibis_yaml.config import config

    tokenized = dask.base.tokenize(to_tokenize)
    name = ("_" + prefix + "_" + tokenized)[: config.hash_length].lower()
    return name


def _clean_udf_name(udf_name):
    if udf_name.isidentifier():
        return udf_name
    else:
        return f"fun_{re.sub(r'[^0-9a-zA-Z_]', '_', udf_name)}".lower()
