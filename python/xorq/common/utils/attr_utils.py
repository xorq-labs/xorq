from toolz import compose


convert_sorted_kwargs_tuple = compose(tuple, sorted, dict.items, dict)


def validate_kwargs_tuple(instance, attribute, value):
    assert isinstance(value, tuple) and all(
        isinstance(el, tuple) and len(el) == 2 for el in value
    )
