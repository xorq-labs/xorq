import operator


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    deep_iterable,
    instance_of,
    optional,
)
from dask.utils import Dispatch

import xorq.expr.datatypes as dt
from xorq.common.utils.func_utils import if_not_none


class KVField(StrEnum):
    KEY = "key"
    VALUE = "value"


ENCODED = "encoded"
# NOTE: may want other types supported
KV_ENCODED_TYPE = dt.Array(
    dt.Struct({KVField.KEY.value: dt.string, KVField.VALUE.value: dt.float64})
)


class KVEncoder:
    """
    Encoder for key-value format Array[Struct{key, value}].

    Used for sklearn transformers where output schema is not known until fit time
    (e.g., OneHotEncoder, CountVectorizer, ColumnTransformer).

    The encode method converts fitted sklearn transform output to encoded format.
    The decode method decodes the encoded column back to individual columns.
    """

    return_type = KV_ENCODED_TYPE

    @staticmethod
    @toolz.curry
    def encode(model, df):
        """
        Encode fitted sklearn transform output to Array[Struct{key, value}] format.

        Called at execution time on a fitted model. The feature names are obtained
        from the fitted model via get_feature_names_out(). The value type is inferred
        from the actual transformed data.

        Parameters
        ----------
        model : sklearn estimator
            Fitted sklearn transformer with get_feature_names_out() method
        df : pandas.DataFrame
            Input DataFrame to transform

        Returns
        -------
        pandas.Series
            Series of tuples, each containing dicts with 'key' and 'value'
        """
        import pandas as pd

        names = model.get_feature_names_out()
        result = model.transform(df)
        # Handle sparse matrices
        if hasattr(result, "toarray"):
            result = result.toarray()

        # Force float64 to match return_type and ensure PyArrow compatibility
        return pd.Series(
            (
                tuple(
                    {KVField.KEY: key, KVField.VALUE: float(value)}
                    for key, value in zip(names, row)
                )
                for row in result
            )
        )

    @staticmethod
    def decode(series):
        """
        Decode Array[Struct{key, value}] column to individual columns.

        Extracts keys and values from the encoded format, then creates
        a DataFrame with the keys as column names.

        Parameters
        ----------
        series : pandas.Series
            A KV-encoded column

        Returns
        -------
        pandas.DataFrame
            DataFrame with decoded columns
        """
        import pandas as pd

        if len(series) == 0:
            raise ValueError

        # Extract keys and values from the encoded format
        keys, values = (
            tuple(tuple(dct[which] for dct in lst) for lst in series)
            for which in (KVField.KEY, KVField.VALUE)
        )
        # All rows should have the same keys
        (columns, *rest) = set(keys)
        assert not rest, "Inconsistent keys across rows"

        decoded = pd.DataFrame(
            values,
            index=series.index,
            columns=columns,
        )
        return decoded

    @classmethod
    def decode_encoded_column(cls, df, features, encoded_col):
        """Decode a single KV-encoded column."""
        if (col := df.get(encoded_col)) is None:
            raise ValueError(f"{encoded_col} not in DataFrame")
        if col.empty:
            raise ValueError(f"cannot decode empty column {encoded_col}")
        # remove encoded_col from features and append the columns it becomes
        new_features = tuple(c for c in features if c != encoded_col) + tuple(
            item[KVField.KEY] for item in col.iloc[0]
        )
        result_df = df.drop(columns=[encoded_col]).join(cls.decode(df[encoded_col]))
        return result_df, new_features

    @classmethod
    def decode_encoded_columns(cls, df, features, encoded_cols):
        """Decode multiple KV-encoded columns."""
        for encoded_col in encoded_cols:
            df, features = cls.decode_encoded_column(df, features, encoded_col)
        return df, features

    @classmethod
    def is_kv_encoded_type(cls, typ):
        return typ == cls.return_type

    @staticmethod
    def get_kv_encoded_cols(expr, features=None):
        """
        Get column names from expr that have KV-encoded type.

        Parameters
        ----------
        expr : ibis expression
            Expression to check schema
        features : tuple, optional
            If provided, only check these columns. Otherwise check all columns.

        Returns
        -------
        tuple
            Column names that have KV_ENCODED_TYPE
        """
        schema = expr.schema()
        cols_to_check = features if features else tuple(schema.keys())
        return tuple(
            col
            for col in cols_to_check
            if KVEncoder.is_kv_encoded_type(schema.get(col))
        )

    @staticmethod
    def get_kv_value_type(typ):
        """Extract the value type from KV-encoded format Array[Struct{key, value}]."""
        if KVEncoder.is_kv_encoded_type(typ):
            return typ.value_type.fields[KVField.VALUE]
        return typ


@frozen
class Structer:
    """
    Describes the output schema of a transformer.

    Three modes:
    1. Known schema (struct is set, no kv_child_names): Output columns are known at build time
    2. KV-encoded (struct is None): Output uses KVEncoder format, resolved at runtime
    3. Hybrid (struct is set, kv_child_names is set): Mixed known + named KV array fields

    For KV-encoded mode, input_columns tracks which columns get transformed.
    passthrough_columns tracks columns that pass through unchanged (e.g., from
    ColumnTransformer with remainder='passthrough').

    The needs_target field indicates whether the transformer requires a target
    variable (y) during fitting (e.g., supervised feature selectors like SelectKBest).

    For hybrid mode, kv_child_names tracks the names of KV-encoded child transformers.
    The struct contains both known fields and named KV array fields.
    """

    struct = field(
        validator=optional(instance_of(dt.Struct)),
        default=None,
    )
    input_columns = field(
        validator=optional(deep_iterable(instance_of(str), instance_of(tuple))),
        default=None,
        converter=if_not_none(tuple),
    )
    passthrough_columns = field(
        validator=optional(deep_iterable(instance_of(str), instance_of(tuple))),
        default=(),
        converter=if_not_none(tuple),
    )
    needs_target = field(
        validator=instance_of(bool),
        default=False,
    )
    is_series = field(
        validator=instance_of(bool),
        default=False,
    )
    kv_child_names = field(
        validator=optional(deep_iterable(instance_of(str), instance_of(tuple))),
        default=(),
        converter=if_not_none(tuple),
    )

    @property
    def is_hybrid(self):
        """True if this Structer has mixed known + KV-encoded children."""
        return self.struct is not None and len(self.kv_child_names) > 0

    @property
    def is_kv_encoded(self):
        """True if this Structer uses KV-encoded format (not hybrid)."""
        return self.struct is None

    @property
    def dtype(self):
        if self.is_kv_encoded:
            raise ValueError(
                "KV-encoded Structer has no dtype - schema resolved at runtime"
            )
        return toolz.valmap(operator.methodcaller("to_pandas"), self.struct.fields)

    @property
    def return_type(self):
        if self.is_kv_encoded:
            return KV_ENCODED_TYPE
        return self.struct

    @property
    def output_columns(self):
        """Return the output column names (only for known schema)."""
        if self.is_kv_encoded:
            raise ValueError(
                "KV-encoded Structer has no output_columns - schema resolved at runtime"
            )
        return tuple(self.struct.fields.keys())

    def get_output_columns(self, dest_col="transformed"):
        """Return the output column names for use as features in the next step."""
        if self.is_kv_encoded:
            return (dest_col,)
        return tuple(self.dtype.keys())

    def maybe_unpack(self, expr, col_name):
        """Unpack struct column if needed, otherwise return expr unchanged."""
        if self.is_kv_encoded:
            return expr
        return expr.unpack(col_name)

    def get_convert_array(self):
        if self.is_kv_encoded:
            raise ValueError(
                "get_convert_array cannot be used with KV-encoded Structer"
            )
        return self.convert_array(self.struct)

    @classmethod
    @toolz.curry
    def convert_array(cls, struct, array):
        import pandas as pd

        self = cls(struct)
        return (
            pd.DataFrame(array, columns=struct.fields)
            .astype(self.dtype)
            .to_dict(orient="records")
        )

    @classmethod
    def from_names_typ(cls, names, typ):
        """Create a Structer with known schema."""
        struct = dt.Struct({name: typ for name in names})
        return cls(struct=struct)

    @classmethod
    @toolz.curry
    def from_n_typ_prefix(cls, n, typ=float, prefix="transformed_"):
        """Create a Structer with known schema using numbered column names."""
        names = tuple(f"{prefix}{i}" for i in range(n))
        return cls.from_names_typ(names, typ)

    @classmethod
    def kv_encoded(cls, input_columns):
        """
        Create a Structer for KV-encoded output.

        Use this for transformers where output schema is not known until fit time.

        Parameters
        ----------
        input_columns : tuple
            Columns that will be transformed (output as KV-encoded)

        Returns
        -------
        Structer
            A Structer configured for KV-encoded output
        """
        return cls(
            struct=None,
            input_columns=input_columns,
        )

    @classmethod
    def from_instance_expr(cls, instance, expr, features=None):
        return structer_from_instance(instance, expr, features=features)


structer_from_instance = Dispatch()


@structer_from_instance.register(object)
def register_object(instance, expr, features=None):
    raise ValueError(f"can't handle type {instance.__class__}")


@structer_from_instance.register_lazy("sklearn")
def lazy_register_sklearn():
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    def _normalize_columns(columns):
        """Normalize columns to tuple format."""
        match columns:
            case list():
                return tuple(columns)
            case str():
                return (columns,)
            case tuple():
                return columns
            case None:
                return ()
            case _:
                raise TypeError(f"Unsupported columns type: {type(columns)}")

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
        # Get types and handle KV-encoded format - extract value type from Array[Struct{key, value}]
        types = {
            KVEncoder.get_kv_value_type(t)
            for t in expr.select(features).schema().values()
        }
        (typ, *rest) = types
        if rest:
            raise ValueError
        base_structer = Structer.from_n_typ_prefix(n=instance.k, typ=typ)
        # SelectKBest is a supervised feature selector that needs target during fit
        return Structer(
            struct=base_structer.struct,
            needs_target=True,
        )

    @structer_from_instance.register(OneHotEncoder)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    @structer_from_instance.register(TfidfVectorizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer(struct=None, input_columns=features, is_series=True)

    @structer_from_instance.register(ColumnTransformer)
    def _(instance, expr, features=None):
        """
        Compute ColumnTransformer schema from children.

        - If all children have known schemas, combine them into a single struct.
        - If mixed known/KV-encoded, return hybrid Structer with named KV fields.
        - If all KV-encoded, return pure KV Structer.
        - Passthrough columns are included in the schema.
        """
        transformers = getattr(instance, "transformers_", None) or instance.transformers

        classified = [
            (name, _normalize_columns(columns), transformer)
            for name, transformer, columns in transformers
            if transformer != "drop"
        ]

        passthrough_cols = [
            col
            for name, cols, transformer in classified
            if transformer == "passthrough"
            for col in cols
        ]
        transform_items = [
            (name, cols, transformer)
            for name, cols, transformer in classified
            if transformer != "passthrough"
        ]

        # Build child Structer instances for each transformer
        child_items = [
            (
                name,
                structer_from_instance(
                    transformer,
                    expr.select(*cols) if cols else expr,
                    features=cols or None,
                ),
            )
            for name, cols, transformer in transform_items
        ]

        all_input_cols = [col for name, cols, _ in transform_items for col in cols]
        any_kv_encoded = any(s.is_kv_encoded or s.is_hybrid for _, s in child_items)
        all_kv_encoded = all(s.is_kv_encoded for _, s in child_items)

        schema = expr.schema()
        combined_fields = {col: schema[col] for col in passthrough_cols}

        if instance.remainder == "passthrough":
            handled_cols = set(all_input_cols) | set(passthrough_cols)
            remainder_cols = [c for c in expr.columns if c not in handled_cols]
            passthrough_cols = passthrough_cols + remainder_cols
            combined_fields.update({col: schema[col] for col in remainder_cols})

        # All KV-encoded: entire output becomes KV-encoded
        if all_kv_encoded:
            return Structer(
                struct=None,
                input_columns=tuple(all_input_cols),
                passthrough_columns=tuple(passthrough_cols),
            )

        # Mixed known/KV-encoded: build hybrid struct with named KV array fields
        if any_kv_encoded:
            # KV-encoded children become named KV array fields
            kv_fields = {
                name: KV_ENCODED_TYPE for name, s in child_items if s.is_kv_encoded
            }
            # Known-schema and hybrid children contribute their struct fields
            known_fields = {
                k: v
                for name, s in child_items
                if not s.is_kv_encoded
                for k, v in s.struct.fields.items()
            }
            # Collect all KV child names (direct + from hybrid children)
            kv_child_names = tuple(
                name for name, s in child_items if s.is_kv_encoded
            ) + tuple(
                kv_name
                for _, s in child_items
                if s.is_hybrid
                for kv_name in s.kv_child_names
            )

            return Structer(
                struct=dt.Struct({**combined_fields, **known_fields, **kv_fields}),
                input_columns=tuple(all_input_cols),
                passthrough_columns=tuple(passthrough_cols),
                kv_child_names=kv_child_names,
            )

        # All known-schema: regular struct
        combined_fields.update(
            {
                k: v
                for name, child_structer in child_items
                for k, v in child_structer.struct.fields.items()
            }
        )
        return Structer(
            struct=dt.Struct(combined_fields),
            passthrough_columns=tuple(passthrough_cols),
        )

    @structer_from_instance.register(FeatureUnion)
    def _(instance, expr, features=None):
        """
        Compute FeatureUnion schema from children.

        FeatureUnion concatenates outputs from all transformers horizontally.
        - If all children have known schemas, combine them into a single struct.
        - If mixed known/KV-encoded, return hybrid Structer with named KV fields.
        - If all KV-encoded, return pure KV Structer.
        """
        features = features or tuple(expr.columns)

        # Build child Structer instances for each transformer
        child_items = [
            (name, structer_from_instance(transformer, expr, features=features))
            for name, transformer in instance.transformer_list
        ]

        any_kv_encoded = any(s.is_kv_encoded or s.is_hybrid for _, s in child_items)
        all_kv_encoded = all(s.is_kv_encoded for _, s in child_items)

        # All KV-encoded: entire output becomes KV-encoded
        if all_kv_encoded:
            return Structer(
                struct=None,
                input_columns=features,
            )

        # Mixed known/KV-encoded: build hybrid struct with named KV array fields
        if any_kv_encoded:
            # KV-encoded children become named KV array fields
            kv_fields = {
                name: KV_ENCODED_TYPE for name, s in child_items if s.is_kv_encoded
            }
            # Known-schema and hybrid children contribute their struct fields
            known_fields = {
                k: v
                for name, s in child_items
                if not s.is_kv_encoded
                for k, v in s.struct.fields.items()
            }
            # Collect all KV child names (direct + from hybrid children)
            kv_child_names = tuple(
                name for name, s in child_items if s.is_kv_encoded
            ) + tuple(
                kv_name
                for _, s in child_items
                if s.is_hybrid
                for kv_name in s.kv_child_names
            )

            return Structer(
                struct=dt.Struct({**known_fields, **kv_fields}),
                input_columns=features,
                kv_child_names=kv_child_names,
            )

        # All known-schema: regular struct with prefixed field names
        # FeatureUnion prefixes output columns with transformer name (e.g., "scaled__age")
        combined_fields = {
            f"{name}__{field_name}": field_type
            for name, child_structer in child_items
            for field_name, field_type in child_structer.struct.fields.items()
        }

        return Structer(
            struct=dt.Struct(combined_fields),
        )

    @structer_from_instance.register(SklearnPipeline)
    def _(instance, expr, features=None):
        """
        Compute sklearn Pipeline schema from final step.

        Pipeline chains transformers sequentially, so the output schema
        is determined by the final step. We chain through each step,
        updating features based on each step's output.

        - If any step is KV-encoded, the pipeline output is KV-encoded.
        - Raises if any step has an unregistered transformer.
        """
        features = features or tuple(expr.columns)

        if not instance.steps:
            raise ValueError("Pipeline has no steps")

        def get_structer(transformer, current_features):
            if transformer == "passthrough":
                return Structer.from_names_typ(current_features, dt.float64)
            return structer_from_instance(transformer, expr, features=current_features)

        # Chain through steps, but stop analyzing if we hit a KV/hybrid step
        current_features = features
        last_structer = None

        for name, transformer in instance.steps:
            child_structer = get_structer(transformer, current_features)

            # If this step is KV-encoded, downstream steps can't be statically analyzed
            if child_structer.is_kv_encoded:
                return Structer(
                    struct=None,
                    input_columns=features,
                    passthrough_columns=child_structer.passthrough_columns,
                )

            # If this step is hybrid (has KV array fields), downstream steps receive
            # a flattened numpy array with unknown columns from the KV arrays.
            # We return the hybrid structer - downstream analysis would fail anyway
            # because kv_child_names aren't real columns in expr.
            if child_structer.is_hybrid:
                return child_structer

            current_features = child_structer.output_columns
            last_structer = child_structer

        return Structer(
            struct=last_structer.struct,
            passthrough_columns=last_structer.passthrough_columns,
        )
