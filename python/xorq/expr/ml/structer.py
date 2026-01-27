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

    Two modes:
    1. Known schema (struct is set): Output columns are known at build time
    2. KV-encoded (struct is None): Output uses KVEncoder format, resolved at runtime

    For KV-encoded mode, input_columns tracks which columns get transformed.
    passthrough_columns tracks columns that pass through unchanged (e.g., from
    ColumnTransformer with remainder='passthrough').

    The needs_target field indicates whether the transformer requires a target
    variable (y) during fitting (e.g., supervised feature selectors like SelectKBest).
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

    @property
    def is_kv_encoded(self):
        """True if this Structer uses KV-encoded format."""
        # NOTE: this could be explicit flag
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
        - If any child is KV-encoded, the entire ColumnTransformer output is KV-encoded.
        - Passthrough columns are included in the schema when all children are known-schema.
        """
        transformers = getattr(instance, "transformers_", None) or instance.transformers

        combined_fields = {}
        any_kv_encoded = False
        all_input_cols = []
        passthrough_cols = []

        for name, transformer, columns in transformers:
            if transformer == "drop":
                continue

            cols = _normalize_columns(columns)

            if transformer == "passthrough":
                # Passthrough keeps original columns with original types
                passthrough_cols.extend(cols)
                for col in cols:
                    combined_fields[col] = expr.schema()[col]
                continue

            all_input_cols.extend(cols)

            # Get child's Structer (raises if unregistered)
            child_expr = expr.select(*cols) if cols else expr
            child_structer = structer_from_instance(
                transformer, child_expr, features=cols if cols else None
            )

            if child_structer.is_kv_encoded:
                any_kv_encoded = True
            else:
                # Add child's output columns to combined schema
                combined_fields.update(child_structer.struct.fields)

        # Handle remainder if set to passthrough
        if instance.remainder == "passthrough":
            handled_cols = set(all_input_cols) | set(passthrough_cols)
            remainder_cols = [c for c in expr.columns if c not in handled_cols]
            passthrough_cols.extend(remainder_cols)
            for col in remainder_cols:
                combined_fields[col] = expr.schema()[col]

        # Build the Structer
        if any_kv_encoded:
            return Structer(
                struct=None,
                input_columns=tuple(all_input_cols),
                passthrough_columns=tuple(passthrough_cols),
            )

        # All children have known schemas - combine them
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
        - If any child is KV-encoded, the entire FeatureUnion output is KV-encoded.
        """
        features = features or tuple(expr.columns)

        combined_fields = {}
        any_kv_encoded = False

        for name, transformer in instance.transformer_list:
            # Get child's Structer (raises if unregistered)
            child_structer = structer_from_instance(
                transformer, expr, features=features
            )

            if child_structer.is_kv_encoded:
                any_kv_encoded = True
            else:
                # Add child's output columns to combined schema
                combined_fields.update(child_structer.struct.fields)

        if any_kv_encoded:
            return Structer(
                struct=None,
                input_columns=features,
            )

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
        current_features = features
        last_structer = None

        for name, transformer in instance.steps:
            if transformer == "passthrough":
                # Passthrough keeps current features unchanged
                last_structer = Structer.from_names_typ(
                    current_features,
                    dt.float64,  # Default type; actual types preserved at runtime
                )
                continue

            # Get child's Structer (raises if unregistered)
            child_structer = structer_from_instance(
                transformer, expr, features=current_features
            )
            last_structer = child_structer

            if child_structer.is_kv_encoded:
                # Once we hit KV-encoded, we stay KV-encoded
                return Structer(
                    struct=None,
                    input_columns=features,
                    passthrough_columns=child_structer.passthrough_columns,
                )

            # Update features for next step based on this step's output
            current_features = child_structer.output_columns

        if last_structer is None:
            raise ValueError("Pipeline has no steps")

        return Structer(
            struct=last_structer.struct,
            passthrough_columns=last_structer.passthrough_columns,
        )
