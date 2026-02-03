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
    def _make_kv_tuple(names, values):
        """Convert parallel names/values to tuple of {key, value} dicts."""
        return tuple(
            {KVField.KEY: key, KVField.VALUE: float(value)}
            for key, value in zip(names, values)
        )

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
        return pd.Series(KVEncoder._make_kv_tuple(names, row) for row in result)

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
        return (
            typ.value_type.fields[KVField.VALUE]
            if KVEncoder.is_kv_encoded_type(typ)
            else typ
        )


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

        When True, struct is None and output schema is resolved at runtime.
        For containers with KV fields in struct, use struct_has_kv_fields instead.
        """
        return self.struct is None

    @property
    def struct_has_kv_fields(self):
        """True if struct contains any KV-encoded field types.

        Used for containers (ColumnTransformer, FeatureUnion) where the struct
        wraps one or more KV-encoded fields. Handles both mixed output (KV + non-KV)
        and all-KV output cases via convert_struct_with_kv.

        Returns False when struct is None (use is_kv_encoded for leaf case).
        """
        return (
            False
            if self.struct is None
            else any(
                KVEncoder.is_kv_encoded_type(typ) for typ in self.struct.fields.values()
            )
        )

    @property
    def has_kv_output(self):
        """True if this Structer produces any KV-encoded output.

        Combines is_kv_encoded (leaf) and struct_has_kv_fields (container).
        Use this when you need to check if any KV output exists regardless of form.
        """
        return self.is_kv_encoded or self.struct_has_kv_fields

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
        For hybrid output (struct_has_kv_fields), returns the struct field names
        which include both known-schema columns and KV column names.
        """
        if self.is_kv_encoded:
            return (dest_col,)
        return tuple(self.dtype.keys())

    def maybe_unpack(self, expr, col_name):
        """Unpack struct column if needed, otherwise return expr unchanged.

        For pure KV-encoded output (is_kv_encoded, struct is None),
        we don't unpack since the schema is resolved at runtime.
        For hybrid output (struct_has_kv_fields), we unpack to get both
        known-schema columns and KV columns as separate fields.
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

    def get_convert_struct_with_kv(self):
        """Return a curried function for converting struct with KV fields.

        For containers with both known-schema and KV-encoded columns.
        """
        if not self.struct_has_kv_fields:
            raise ValueError(
                "get_convert_struct_with_kv requires struct_has_kv_fields to be True"
            )
        return self.convert_struct_with_kv(self.struct)

    @classmethod
    @toolz.curry
    def convert_struct_with_kv(cls, struct, model, df):
        """Convert sklearn output to struct with known-schema and KV-encoded columns.

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
            List of dicts, one per row, with struct fields
        """
        result = model.transform(df)
        if hasattr(result, "toarray"):
            result = result.toarray()
        feature_names = tuple(model.get_feature_names_out())

        def accumulate_sklearn_col_slice(acc, field_item):
            """Accumulate (col_idx, slices) -> (new_col_idx, updated_slices).

            For KV-encoded fields, delta is the count of feature names matching
            the field prefix (e.g., "cat__color_red", "cat__color_blue" for "cat").
            For non-KV fields, delta is always 1 (single column).
            """
            col_idx, slices = acc
            field_name, field_type = field_item
            is_kv = KVEncoder.is_kv_encoded_type(field_type)
            start = col_idx
            delta = (
                sum(
                    1
                    for fn in feature_names[start:]
                    if fn.startswith(f"{field_name}__")
                )
                if is_kv
                else 1
            )
            end = start + delta
            return (end, {**slices, field_name: (start, end, is_kv)})

        _, field_slices = toolz.reduce(
            accumulate_sklearn_col_slice, struct.fields.items(), (0, {})
        )

        return [
            {
                field_name: (
                    KVEncoder._make_kv_tuple(feature_names[start:end], row[start:end])
                    if is_kv
                    else float(row[start])
                )
                for field_name, (start, end, is_kv) in field_slices.items()
            }
            for row in result
        ]

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


def _prefix_fields(fields, prefix):
    """Prefix all field names with 'prefix__'."""
    return {f"{prefix}__{name}": typ for name, typ in fields.items()}


def _process_child_structer(child, name, input_cols):
    """Process child structer, returning (fields_dict, kv_cols or None).

    Parameters
    ----------
    child : Structer
        Child structer from recursive get_structer_out call
    name : str
        Prefix name for the fields
    input_cols : tuple
        Columns to track if child has KV output

    Returns
    -------
    tuple
        (schema_fields dict, kv_input_cols or None)
    """
    if child.has_kv_output:
        return {name: KV_ENCODED_TYPE}, input_cols
    return _prefix_fields(child.struct.fields, name), None


@toolz.curry
def _process_ct_item(expr, item):
    """Process single ColumnTransformer item -> (fields, kv_cols or None)."""
    name, transformer, columns = item
    cols = _normalize_columns(columns)

    match transformer:
        case "drop":
            return ({}, None)
        case "passthrough":
            input_schema = expr.schema()
            return (
                _prefix_fields({col: input_schema[col] for col in cols}, name),
                None,
            )
        case _:
            child = get_structer_out(transformer, expr, features=cols)
            return _process_child_structer(child, name, cols)


def _merge_ct_results(acc, result):
    """Merge (fields, kv_cols) into accumulated (schema_fields, kv_input_cols)."""
    schema_fields, kv_input_cols = acc
    fields, kv_cols = result
    return (
        {**schema_fields, **fields},
        kv_input_cols + list(kv_cols) if kv_cols else kv_input_cols,
    )


def _get_remainder_fields(transformer_items, features, expr):
    """Get fields for remainder='passthrough' columns.

    Columns explicitly assigned to any transformer (including 'drop')
    are considered handled and excluded from remainder.
    """
    handled = {
        col
        for name, transformer, columns in transformer_items
        for col in _normalize_columns(columns)
    }
    return {col: expr.schema()[col] for col in features if col not in handled}


@toolz.curry
def _process_fu_item(expr, features, item):
    """Process single FeatureUnion item -> (fields, kv_cols or None)."""
    name, transformer = item
    child = get_structer_out(transformer, expr, features=features)
    return _process_child_structer(child, name, features)


@toolz.curry
def _accumulate_pipeline_step(expr, acc, step):
    """Accumulate (features, structer) through pipeline steps."""
    current_features, _ = acc
    name, transformer = step
    if transformer == "passthrough":
        return acc
    structer = get_structer_out(transformer, expr, features=current_features)
    new_features = (
        current_features if structer.has_kv_output else structer.output_columns
    )
    return (new_features, structer)


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

    Raises
    ------
    TypeError
        If sklearnish is not a sklearn BaseEstimator or known container type.
    """
    from sklearn.base import BaseEstimator
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline as SklearnPipeline

    features = features or tuple(expr.columns)

    match sklearnish:
        case ColumnTransformer():
            transformer_items = _get_transformer_items(sklearnish)

            schema_fields, kv_input_cols = toolz.reduce(
                _merge_ct_results,
                map(_process_ct_item(expr), transformer_items),
                ({}, []),
            )

            if sklearnish.remainder == "passthrough":
                remainder = _get_remainder_fields(transformer_items, features, expr)
                schema_fields = {
                    **schema_fields,
                    **_prefix_fields(remainder, "remainder"),
                }

            return Structer(
                struct=dt.Struct(schema_fields),
                input_columns=tuple(kv_input_cols) if kv_input_cols else None,
            )

        case FeatureUnion():
            transformer_items = _get_transformer_items(sklearnish)

            schema_fields, kv_input_cols = toolz.reduce(
                _merge_ct_results,
                map(_process_fu_item(expr, features), transformer_items),
                ({}, []),
            )

            return Structer(
                struct=dt.Struct(schema_fields),
                input_columns=tuple(kv_input_cols) if kv_input_cols else None,
            )

        case SklearnPipeline():
            _, final_structer = toolz.reduce(
                _accumulate_pipeline_step(expr),
                sklearnish.steps,
                (features, None),
            )
            # If all steps passthrough return default structer
            if final_structer is None:
                return Structer.from_names_typ(features, dt.float64)

            # Wrap pure KV-encoded output in struct with "encoded" column
            if final_structer.is_kv_encoded:
                return Structer(
                    struct=dt.Struct({"encoded": KV_ENCODED_TYPE}),
                    input_columns=final_structer.input_columns,
                )

            return final_structer

        case BaseEstimator():
            return structer_from_instance(sklearnish, expr, features)

        case _:
            raise TypeError(f"Unexpected type in get_structer_out: {type(sklearnish)}")


def get_schema_out(sklearnish, expr, features=None):
    """
    Compute output schema from sklearn-like objects.

    Note: Some sklearn estimators (e.g., OneHotEncoder) have output schemas
    that cannot be known until fit time. For these, the IR will show a
    KV-encoded column (Array[Struct{key, value}]) rather than the actual
    output columns.

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
    from sklearn.base import ClassNamePrefixFeaturesOutMixin, OneToOneFeatureMixin
    from sklearn.cluster import Birch
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomTreesEmbedding
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.feature_selection._base import SelectorMixin
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.kernel_approximation import AdditiveChi2Sampler
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import (
        KBinsDiscretizer,
        OneHotEncoder,
        PolynomialFeatures,
        SplineTransformer,
        TargetEncoder,
    )

    def _structer_from_maybe_kv_inputs(expr, features):
        """Create Structer for transformers that preserve column structure.

        If any input column is KV-encoded, output is KV-encoded (feature names
        resolved at runtime). Otherwise, output has known schema with float type.
        """
        features = features or tuple(expr.columns)
        # Check if any input columns are KV-encoded
        kv_cols = KVEncoder.get_kv_encoded_cols(expr, features)
        if kv_cols:
            # KV-encoded input -> KV-encoded output
            return Structer.kv_encoded(input_columns=features)
        # Known schema -> known schema with float output
        return Structer.from_names_typ(features, float)

    # Register SimpleImputer (doesn't inherit from OneToOneFeatureMixin)
    @structer_from_instance.register(SimpleImputer)
    def _(instance, expr, features=None):
        return _structer_from_maybe_kv_inputs(expr, features)

    # Register MissingIndicator: output depends on features parameter
    # - features='all': one indicator per input feature (one-to-one)
    # - features='missing-only' (default): output count unknown until fit time
    @structer_from_instance.register(MissingIndicator)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        if instance.features == "all":
            # One-to-one mapping: one indicator per input feature
            return Structer.from_names_typ(features, dt.boolean)
        # 'missing-only': output count depends on which features have missing values
        return Structer.kv_encoded(input_columns=features)

    # Register all one-to-one transformers (scalers) via OneToOneFeatureMixin
    @structer_from_instance.register(OneToOneFeatureMixin)
    def _(instance, expr, features=None):
        return _structer_from_maybe_kv_inputs(expr, features)

    # Register feature selectors
    def _get_n_features_out_selectors(instance, n_features_in):
        """Get number of output features from a feature selector instance.

        Returns the number of features the selector will output, or None if
        it cannot be determined at compile time.
        """
        # SelectKBest uses k parameter
        if hasattr(instance, "k"):
            k = instance.k
            if k == "all":
                return n_features_in
            return min(k, n_features_in)

        # SelectPercentile uses percentile parameter
        if hasattr(instance, "percentile"):
            return max(1, int(n_features_in * instance.percentile / 100))

        # RFE, RFECV, SequentialFeatureSelector use n_features_to_select
        if hasattr(instance, "n_features_to_select"):
            n = instance.n_features_to_select
            if n is None:
                # Default is half
                return max(1, n_features_in // 2)
            if isinstance(n, float) and 0 < n < 1:
                return max(1, int(n_features_in * n))
            return min(n, n_features_in)

        # SelectFromModel uses max_features or threshold
        if hasattr(instance, "max_features"):
            max_f = instance.max_features
            if max_f is not None:
                if callable(max_f):
                    return None  # Can't determine at compile time
                return min(max_f, n_features_in)
            # Threshold-based selection - can't determine at compile time
            return None

        # SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect - alpha/threshold based
        # Can't determine number of features at compile time
        return None

    def _structer_for_feature_selector(instance, expr, features):
        """Create Structer for feature selectors.

        If input is KV-encoded or output count is unknown, uses KV-encoding.
        Otherwise, creates stub column names (transformed_0, transformed_1, ...).
        """
        features = features or tuple(expr.columns)
        kv_cols = KVEncoder.get_kv_encoded_cols(expr, features)

        if kv_cols:
            # KV-encoded input -> KV-encoded output
            return Structer(struct=None, input_columns=features, needs_target=True)

        # Try to determine output feature count
        n_out = _get_n_features_out_selectors(instance, len(features))
        if n_out is None:
            # Unknown output count -> KV-encoded
            return Structer(struct=None, input_columns=features, needs_target=True)

        # Get the value type from input columns
        types = {
            KVEncoder.get_kv_value_type(t)
            for t in expr.select(features).schema().values()
        }
        (typ, *rest) = types
        if rest:
            raise ValueError(f"Mixed types in feature columns: {types}")

        # Known output count -> stub column names
        base_structer = Structer.from_n_typ_prefix(n=n_out, typ=typ)
        return Structer(
            struct=base_structer.struct,
            input_columns=features,
            needs_target=True,
        )

    # Register all feature selectors via SelectorMixin base class
    @structer_from_instance.register(SelectorMixin)
    def _(instance, expr, features=None):
        return _structer_for_feature_selector(instance, expr, features)

    def _get_n_components(instance):
        """Get number of output components from a transformer instance.

        Returns the number of components/clusters the transformer will output,
        or None if it cannot be determined at compile time.
        """
        # PCA, NMF, TruncatedSVD, FastICA, etc. use n_components
        if hasattr(instance, "n_components"):
            n = instance.n_components
            # None means auto-determined at fit time
            # "mle" is PCA-specific for auto-selection
            if n is None or n == "mle":
                return None
            if isinstance(n, int):
                return n
            return None

        # KMeans, MiniBatchKMeans use n_clusters
        if hasattr(instance, "n_clusters"):
            n = instance.n_clusters
            if isinstance(n, int):
                return n
            return None

        return None

    def _get_class_name_prefix(instance):
        """Get sklearn-style prefix from instance class name (e.g., 'pca', 'nmf')."""
        return instance.__class__.__name__.lower()

    def _structer_from_class_name_prefix_features(instance, expr, features):
        """Create Structer for transformers with dynamic output (PCA, NMF, etc.).

        If n_components/n_clusters is known, creates stub column names matching
        sklearn's get_feature_names_out() format (e.g., pca0, pca1, nmf0, nmf1).
        Otherwise, uses KV-encoding for runtime schema resolution. All classes inherit from ClassNamePrefixFeaturesOutMixin.
        """
        features = features or tuple(expr.columns)
        kv_cols = KVEncoder.get_kv_encoded_cols(expr, features)

        if kv_cols:
            return Structer.kv_encoded(input_columns=features)

        n_out = _get_n_components(instance)
        return (
            Structer.from_n_typ_prefix(
                n=n_out, typ=float, prefix=_get_class_name_prefix(instance)
            )
            if n_out is not None
            else Structer.kv_encoded(input_columns=features)
        )

    # Register all dynamic-output transformers via ClassNamePrefixFeaturesOutMixin
    # (PCA, NMF, TruncatedSVD, FastICA, KMeans, etc.)
    @structer_from_instance.register(ClassNamePrefixFeaturesOutMixin)
    def _(instance, expr, features=None):
        return _structer_from_class_name_prefix_features(instance, expr, features)

    @structer_from_instance.register(OneHotEncoder)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    @structer_from_instance.register(TfidfVectorizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer(struct=None, input_columns=features, is_series=True)

    # TargetEncoder: inherits OneToOneFeatureMixin but needs needs_target=True
    @structer_from_instance.register(TargetEncoder)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        kv_cols = KVEncoder.get_kv_encoded_cols(expr, features)
        if kv_cols:
            return Structer(struct=None, input_columns=features, needs_target=True)
        return Structer(
            struct=Structer.from_names_typ(features, float).struct,
            input_columns=features,
            needs_target=True,
        )

    # CountVectorizer: text vectorizer like TfidfVectorizer
    @structer_from_instance.register(CountVectorizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer(struct=None, input_columns=features, is_series=True)

    # PolynomialFeatures: output depends on degree and input feature count
    @structer_from_instance.register(PolynomialFeatures)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    # SplineTransformer: output depends on n_knots and input feature count
    @structer_from_instance.register(SplineTransformer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    # KBinsDiscretizer: output depends on n_bins and encode strategy
    @structer_from_instance.register(KBinsDiscretizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    # DictVectorizer: output depends on keys in input dicts
    @structer_from_instance.register(DictVectorizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer(struct=None, input_columns=features, is_series=True)

    # AdditiveChi2Sampler: doesn't inherit ClassNamePrefixFeaturesOutMixin
    @structer_from_instance.register(AdditiveChi2Sampler)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    # RandomTreesEmbedding: output depends on tree structure (leaf nodes)
    @structer_from_instance.register(RandomTreesEmbedding)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    # Birch: transform() returns distances to all subclusters (count not known until fit)
    # Must be registered before ClassNamePrefixFeaturesOutMixin to override the mixin
    @structer_from_instance.register(Birch)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    # FeatureAgglomeration: works via ClassNamePrefixFeaturesOutMixin + _get_n_components
    # (n_clusters is used for output count)

    @structer_from_instance.register(ColumnTransformer)
    def _(instance, expr, features=None):
        return get_structer_out(instance, expr, features)

    @structer_from_instance.register(FeatureUnion)
    def _(instance, expr, features=None):
        return get_structer_out(instance, expr, features)

    @structer_from_instance.register(SklearnPipeline)
    def _(instance, expr, features=None):
        return get_structer_out(instance, expr, features)
