import importlib

import pytest

from xorq.common.utils.dispatch import FQNDispatch
from xorq.expr.ml.pipeline_lib import _predict_return_type_dispatch
from xorq.expr.ml.structer import structer_from_instance


ALL_REGISTERED_FQNS = (
    *_predict_return_type_dispatch.registered_fqns,
    *structer_from_instance.registered_fqns,
)


@pytest.mark.parametrize("fqn_string", ALL_REGISTERED_FQNS)
def test_fqn_resolves_to_real_class(fqn_string):
    module_path, _, class_name = fqn_string.rpartition(".")
    pytest.importorskip(module_path)
    mod = importlib.import_module(module_path)
    assert hasattr(mod, class_name), f"{fqn_string} does not exist"


def test_fqn_dispatch_basic():
    def handle_int(x):
        return "int"

    def handle_str(x):
        return "str"

    d = FQNDispatch((("builtins.int", handle_int), ("builtins.str", handle_str)))
    assert d(1) == "int"
    assert d("x") == "str"


def test_fqn_dispatch_mro():
    def handle_base(x):
        return "base"

    d = FQNDispatch((("builtins.object", handle_base),))
    assert d(42) == "base"
    assert d("x") == "base"


def test_fqn_dispatch_default():
    def fallback(x):
        return "default"

    d = FQNDispatch((), default=fallback)
    assert d(42) == "default"


def test_fqn_dispatch_no_match_raises():
    d = FQNDispatch(())
    with pytest.raises(TypeError, match="No dispatch"):
        d(42)


def test_fqn_dispatch_caches():
    def handle(x):
        return "ok"

    d = FQNDispatch((("builtins.int", handle),))
    d(1)
    assert int in d._cache
    d(2)
    assert d._cache[int] is handle
