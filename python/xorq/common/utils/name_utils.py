import re


def _clean_udf_name(udf_name):
    if udf_name.isidentifier():
        return udf_name
    else:
        return f"fun_{re.sub(r'[^0-9a-zA-Z_]', '_', udf_name)}".lower()
