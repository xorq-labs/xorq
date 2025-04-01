# https://stackoverflow.com/a/12141386
from ctypes import (
    POINTER,
    Structure,
    c_size_t,
    c_void_p,
    cast,
    py_object,
)


PyObject_HEAD = [
    ("ob_refcnt", c_size_t),
    ("ob_type", c_void_p),
]


def make_class(*fields):
    _fields_ = PyObject_HEAD + [(field, c_void_p) for field in fields]
    return type("ctypes-hack", (Structure,), {"_fields_": _fields_})


# FIXME: define singledispatch and register common types
def get_ctypes_field(fields, field, obj):
    assert field in fields
    c_methcallobj = cast(c_void_p(id(obj)), POINTER(make_class(*fields))).contents

    return cast(getattr(c_methcallobj, field), py_object).value
