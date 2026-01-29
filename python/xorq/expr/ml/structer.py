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
        """True if this Structer uses KV-encoded format (leaf transformers only).

        For containers with hybrid output (struct set but containing KV fields),
        use any_kv_encoded or all_kv_encoded instead.
        """
        # NOTE: this could be explicit flag
        return self.struct is None

    @property
    def any_kv_encoded(self):
        """True if any field in struct is KV-encoded (for containers with hybrid output)."""
        if self.struct is None:
            return True
        return any(
            KVEncoder.is_kv_encoded_type(typ) for typ in self.struct.fields.values()
        )

    @property
    def all_kv_encoded(self):
        """True if all fields in struct are KV-encoded."""
        if self.struct is None:
            return True
        if not self.struct.fields:
            return False
        return all(
            KVEncoder.is_kv_encoded_type(typ) for typ in self.struct.fields.values()
        )

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
        """Return the output column names for use as features in the next step.

        For pure KV-encoded output (is_kv_encoded, struct is None),
        returns the dest_col since the actual columns are resolved at runtime.
        For hybrid output (any_kv_encoded but struct is set), returns the struct
        field names which include both known-schema columns and KV column names.
        """
        if self.is_kv_encoded:
            return (dest_col,)
        return tuple(self.dtype.keys())

    def maybe_unpack(self, expr, col_name):
        """Unpack struct column if needed, otherwise return expr unchanged.

        For pure KV-encoded output (is_kv_encoded, struct is None),
        we don't unpack since the schema is resolved at runtime.
        For hybrid output (any_kv_encoded but struct is set), we unpack to get
        both known-schema columns and KV columns as separate fields.
        """
        if self.is_kv_encoded:
            return expr
        return expr.unpack(col_name)

    def get_convert_array(self):
        if self.is_kv_encoded:
            raise ValueError(
                "get_convert_array cannot be used with KV-encoded Structer"
            )
        return self.convert_array(self.struct)

    def get_convert_hybrid(self):
        """Return a curried function for converting hybrid output.

        For containers with both known-schema and KV-encoded columns.
        """
        if not self.any_kv_encoded:
            raise ValueError("get_convert_hybrid requires any_kv_encoded to be True")
        return self.convert_hybrid(self.struct)

    @classmethod
    @toolz.curry
    def convert_hybrid(cls, struct, model, df):
        """Convert sklearn output to hybrid format with known-schema and KV columns.

        Parameters
        ----------
        struct : dt.Struct
            The struct schema describing output fields
        model : sklearn transformer
            Fitted sklearn transformer with get_feature_names_out() method
        df : pandas.DataFrame
            Input DataFrame to transform

        Returns
        -------
        list[dict]
            List of dicts, one per row, with hybrid struct fields
        """
        # Get sklearn output and feature names
        result = model.transform(df)
        if hasattr(result, "toarray"):
            result = result.toarray()
        feature_names = model.get_feature_names_out()

        # Get schema fields from struct
        schema_fields = struct.fields

        # Build output: for each row, create a dict with schema fields
        rows = []
        for row_idx in range(len(result)):
            row_dict = {}
            sklearn_col_idx = 0

            for field_name, field_type in schema_fields.items():
                if KVEncoder.is_kv_encoded_type(field_type):
                    # KV-encoded field: collect all sklearn columns with this prefix
                    kv_items = []
                    prefix = f"{field_name}__"
                    while sklearn_col_idx < len(feature_names) and feature_names[
                        sklearn_col_idx
                    ].startswith(prefix):
                        key = feature_names[sklearn_col_idx]
                        value = float(result[row_idx, sklearn_col_idx])
                        kv_items.append({KVField.KEY: key, KVField.VALUE: value})
                        sklearn_col_idx += 1
                    row_dict[field_name] = tuple(kv_items)
                else:
                    # Known-schema field: take the next sklearn column as float
                    row_dict[field_name] = float(result[row_idx, sklearn_col_idx])
                    sklearn_col_idx += 1

            rows.append(row_dict)

        return rows

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


def _get_transformer_items(model):
    """
    Get transformer items from unfitted sklearn container.

    Abstracts over sklearn's inconsistent naming:
    - ColumnTransformer uses `transformers`
    - FeatureUnion uses `transformer_list`

    Parameters
    ----------
    model : sklearn container
        ColumnTransformer or FeatureUnion

    Returns
    -------
    list
        For ColumnTransformer: [(name, transformer, columns), ...]
        For FeatureUnion: [(name, transformer), ...]
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import FeatureUnion

    match model:
        case ColumnTransformer(transformers=transformers):
            return transformers
        case FeatureUnion(transformer_list=transformer_list):
            return transformer_list
        case _:
            raise TypeError(f"Unsupported model type: {type(model)}")


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


def get_structer_out(sklearnish, expr, features=None):
    """
    Compute Structer from sklearn-like objects.

    Single source of truth for container schema computation. Handles
    ColumnTransformer, FeatureUnion, and Pipeline with support for hybrid
    output (mixed known-schema + KV-encoded columns).

    For leaf transformers, delegates to structer_from_instance.

    Parameters
    ----------
    sklearnish : sklearn estimator or container
        The sklearn object to compute Structer for (unfitted)
    expr : ibis.Expr
        Expression providing input schema context
    features : tuple, optional
        Features to consider. If None, uses all expr columns.

    Returns
    -------
    Structer
        Complete Structer with struct and input_columns.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline as SklearnPipeline

    features = features or tuple(expr.columns)

    match sklearnish:
        case ColumnTransformer():
            transformer_items = _get_transformer_items(sklearnish)
            schema_fields = {}
            kv_input_cols = []

            for name, transformer, columns in transformer_items:
                cols = _normalize_columns(columns)

                match transformer:
                    case "drop":
                        continue
                    case "passthrough":
                        input_schema = expr.schema()
                        for col in cols:
                            schema_fields[f"{name}__{col}"] = input_schema[col]
                    case _:
                        child = get_structer_out(transformer, expr, features=cols)

                        if child.any_kv_encoded:
                            schema_fields[name] = KV_ENCODED_TYPE
                            kv_input_cols.extend(cols)
                        else:
                            for col_name, col_type in child.struct.fields.items():
                                schema_fields[f"{name}__{col_name}"] = col_type

            if sklearnish.remainder == "passthrough":
                handled = set()
                for name, transformer, columns in transformer_items:
                    if transformer != "drop":
                        handled.update(_normalize_columns(columns))
                for col in features:
                    if col not in handled:
                        schema_fields[f"remainder__{col}"] = expr.schema()[col]

            return Structer(
                struct=dt.Struct(schema_fields),
                input_columns=tuple(kv_input_cols) if kv_input_cols else None,
            )

        case FeatureUnion():
            transformer_items = _get_transformer_items(sklearnish)
            schema_fields = {}
            kv_input_cols = []

            for name, transformer in transformer_items:
                child = get_structer_out(transformer, expr, features=features)

                if child.any_kv_encoded:
                    schema_fields[name] = KV_ENCODED_TYPE
                    kv_input_cols.extend(features)
                else:
                    for col_name, col_type in child.struct.fields.items():
                        schema_fields[f"{name}__{col_name}"] = col_type

            return Structer(
                struct=dt.Struct(schema_fields),
                input_columns=tuple(kv_input_cols) if kv_input_cols else None,
            )

        case SklearnPipeline():
            current_features = features
            current_structer = None

            for name, transformer in sklearnish.steps:
                if transformer == "passthrough":
                    continue
                current_structer = get_structer_out(
                    transformer, expr, features=current_features
                )
                if not current_structer.any_kv_encoded:
                    current_features = current_structer.output_columns

            if current_structer is None:
                return Structer.from_names_typ(features, dt.float64)

            # Wrap pure KV-encoded output in struct with "encoded" column
            if current_structer.is_kv_encoded:
                return Structer(
                    struct=dt.Struct({"encoded": KV_ENCODED_TYPE}),
                    input_columns=current_structer.input_columns,
                )

            return current_structer

        case _:
            return structer_from_instance(sklearnish, expr, features)


def get_schema_out(sklearnish, expr, features=None):
    """
    Compute output schema from sklearn-like objects.

    Parameters
    ----------
    sklearnish : sklearn estimator or container
        The sklearn object to compute schema for (unfitted)
    expr : ibis.Expr
        Expression providing input schema context
    features : tuple, optional
        Features to consider. If None, uses all expr columns.

    Returns
    -------
    ibis.Schema
        The output schema for the sklearn object.
    """
    from xorq.vendor import ibis

    structer = get_structer_out(sklearnish, expr, features)
    return ibis.schema(structer.struct.fields)


@structer_from_instance.register_lazy("sklearn")
def lazy_register_sklearn():
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        return get_structer_out(instance, expr, features)

    @structer_from_instance.register(FeatureUnion)
    def _(instance, expr, features=None):
        return get_structer_out(instance, expr, features)

    @structer_from_instance.register(SklearnPipeline)
    def _(instance, expr, features=None):
        return get_structer_out(instance, expr, features)
