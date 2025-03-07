import toolz


@toolz.curry
def schema_contains(required, schema_in):
    return not set(required.items()).difference(schema_in.items())


@toolz.curry
def schema_concat(schema_in, to_concat):
    return schema_in | to_concat
