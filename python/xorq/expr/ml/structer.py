import operator

import pandas as pd
import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)
from dask.utils import Dispatch

import xorq.expr.datatypes as dt


ENCODED = "encoded"


@frozen
class Structer:
    struct = field(validator=instance_of(dt.Struct))

    @property
    def dtype(self):
        return toolz.valmap(operator.methodcaller("to_pandas"), self.struct.fields)

    @property
    def return_type(self):
        return self.struct

    def get_convert_array(self):
        return self.convert_array(self.struct)

    @classmethod
    @toolz.curry
    def convert_array(cls, struct, array):
        self = cls(struct)
        return (
            pd.DataFrame(array, columns=struct.fields)
            .astype(self.dtype)
            .to_dict(orient="records")
        )

    @classmethod
    def from_names_typ(cls, names, typ):
        struct = dt.Struct({name: typ for name in names})
        return cls(struct)

    @classmethod
    @toolz.curry
    def from_n_typ_prefix(cls, n, typ=float, prefix="transformed_"):
        names = tuple(f"{prefix}{i}" for i in range(n))
        return cls.from_names_typ(names, typ)

    @classmethod
    def from_instance_expr(cls, instance, expr, features=None):
        return structer_from_instance(instance, expr, features=features)


structer_from_instance = Dispatch()


@structer_from_instance.register(object)
def register_object(instance, expr, features=None):
    raise ValueError(f"can't handle type {instance.__class__}")


@structer_from_instance.register_lazy("sklearn")
def lazy_register_sklearn():
    from sklearn.feature_selection import (
        SelectKBest,
    )
    from sklearn.impute import (
        SimpleImputer,
    )
    from sklearn.preprocessing import (
        StandardScaler,
    )

    @structer_from_instance.register(SimpleImputer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        typ = float
        structer = Structer.from_names_typ(features, typ)
        return structer

    @structer_from_instance.register(StandardScaler)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        typ = float
        structer = Structer.from_names_typ(features, typ)
        return structer

    @structer_from_instance.register(SelectKBest)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        (typ, *rest) = set(expr.select(features).schema().values())
        if rest:
            raise ValueError
        structer = Structer.from_n_typ_prefix(n=instance.k, typ=typ)
        return structer
