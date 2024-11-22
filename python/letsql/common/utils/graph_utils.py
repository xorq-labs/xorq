try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata


version = tuple(
    int(part) for part in importlib_metadata.version("ibis-framework").split(".")
)


def replace_fix(fun):
    def wrapper(*args, **kwargs):
        if version > (9, 4, 0):
            node, kw = args
            return fun(node, None, **(kw or dict(zip(node.argnames, node.args))))
        else:
            return fun(*args, **kwargs)

    return wrapper


def get_args(*args, **kwargs):
    if version > (9, 4, 0):
        node, kw = args
        return node, None, (kw or dict(zip(node.argnames, node.args)))
    else:
        node, results = args
        return node, results, kwargs
