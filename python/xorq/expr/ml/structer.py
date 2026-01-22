import operator

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


ENCODED = "encoded"

# Type for key-value encoded format: Array[Struct{key: string, value: float64}]
KV_ENCODED_TYPE = dt.Array(dt.Struct({"key": dt.string, "value": dt.float64}))


class KVEncoder:
    """
    Encoder for key-value format Array[Struct{key, value}].

    Used for sklearn transformers where output schema is not known until fit time
    (e.g., OneHotEncoder, CountVectorizer, ColumnTransformer).

    The encode method converts fitted sklearn transform output to packed format.
    The decode method extracts the packed column back to individual columns.
    """

    return_type = KV_ENCODED_TYPE

    @staticmethod
    @toolz.curry
    def encode(model, df):
        """
        Encode fitted sklearn transform output to Array[Struct{key, value}] format.

        Called at execution time on a fitted model. The feature names are obtained
        from the fitted model via get_feature_names_out().

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

        return pd.Series(
            (
                tuple(
                    {"key": key, "value": float(value)}
                    for key, value in zip(names, row)
                )
                for row in result
            )
        )

    @staticmethod
    def decode(df, col_name):
        """
        Decode Array[Struct{key, value}] column to individual columns.

        Extracts keys and values from the packed format, then creates
        a DataFrame with the keys as column names.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing a KV-encoded column
        col_name : str
            Name of the column containing Array[Struct{key, value}] data

        Returns
        -------
        pandas.DataFrame
            DataFrame with the packed column replaced by individual columns
        """
        import pandas as pd

        series = df[col_name]

        if len(series) == 0:
            return df.drop(columns=[col_name])

        # Extract keys and values from the packed format
        keys, values = (
            [tuple(dct[which] for dct in lst) for lst in series]
            for which in ("key", "value")
        )
        # All rows should have the same keys
        (columns, *rest) = keys
        assert all(el == columns for el in rest), "Inconsistent keys across rows"

        decoded = pd.DataFrame(
            values,
            index=series.index,
            columns=columns,
        )

        # Drop the packed column and join with decoded columns
        return df.drop(columns=[col_name]).join(decoded)


def is_kv_encoded_type(typ):
    """Check if a type is the KV-encoded format by comparing to KV_ENCODED_TYPE."""
    return typ == KV_ENCODED_TYPE


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
    return tuple(col for col in cols_to_check if is_kv_encoded_type(schema.get(col)))


def _get_kv_value_type(typ):
    """Extract the value type from KV-encoded format Array[Struct{key, value}]."""
    if is_kv_encoded_type(typ):
        return typ.value_type.fields["value"]
    return typ


@frozen
class Structer:
    """
    Describes the output schema of a transformer.

    Two modes:
    1. Known schema (struct is set): Output columns are known at build time
    2. KV-encoded (struct is None): Output uses KVEncoder format, resolved at runtime

    For KV-encoded mode, input_columns tracks which columns get transformed,
    and passthrough_columns tracks columns that pass through unchanged.
    """

    struct = field(
        validator=optional(instance_of(dt.Struct)),
        default=None,
    )
    input_columns = field(
        validator=optional(deep_iterable(instance_of(str), instance_of(tuple))),
        default=None,
        converter=lambda x: tuple(x) if x is not None else None,
    )
    passthrough_columns = field(
        validator=optional(deep_iterable(instance_of(str), instance_of(tuple))),
        default=(),
        converter=tuple,
    )

    @property
    def is_kv_encoded(self):
        """True if this Structer uses KV-encoded format."""
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

    def get_convert_array(self):
        if self.is_kv_encoded:
            # For KV-encoded, use KVEncoder.encode
            return KVEncoder.encode
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
    def kv_encoded(cls, input_columns, passthrough_columns=()):
        """
        Create a Structer for KV-encoded output.

        Use this for transformers where output schema is not known until fit time.

        Parameters
        ----------
        input_columns : tuple
            Columns that will be transformed (output as KV-encoded)
        passthrough_columns : tuple
            Columns that pass through unchanged

        Returns
        -------
        Structer
            A Structer configured for KV-encoded output
        """
        return cls(
            struct=None,
            input_columns=input_columns,
            passthrough_columns=passthrough_columns,
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
    from sklearn.decomposition import (
        NMF,
        PCA,
        FactorAnalysis,
        FastICA,
        KernelPCA,
        LatentDirichletAllocation,
        MiniBatchNMF,
        MiniBatchSparsePCA,
        SparsePCA,
        TruncatedSVD,
    )
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import (
        CountVectorizer,
        HashingVectorizer,
        TfidfVectorizer,
    )
    from sklearn.feature_selection import (
        RFE,
        SelectKBest,
    )
    from sklearn.impute import (
        KNNImputer,
        MissingIndicator,
        SimpleImputer,
    )
    from sklearn.kernel_approximation import (
        AdditiveChi2Sampler,
        Nystroem,
        RBFSampler,
        SkewedChi2Sampler,
    )
    from sklearn.preprocessing import (
        Binarizer,
        LabelBinarizer,
        MaxAbsScaler,
        MinMaxScaler,
        MultiLabelBinarizer,
        Normalizer,
        OneHotEncoder,
        OrdinalEncoder,
        PolynomialFeatures,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        SplineTransformer,
        StandardScaler,
    )
    from sklearn.random_projection import (
        GaussianRandomProjection,
        SparseRandomProjection,
    )

    # Scalers - preserve columns exactly
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

    @structer_from_instance.register(MinMaxScaler)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        typ = float
        structer = Structer.from_names_typ(features, typ)
        return structer

    @structer_from_instance.register(RobustScaler)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        typ = float
        structer = Structer.from_names_typ(features, typ)
        return structer

    @structer_from_instance.register(MaxAbsScaler)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.from_names_typ(features, float)

    @structer_from_instance.register(Normalizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.from_names_typ(features, float)

    @structer_from_instance.register(PowerTransformer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.from_names_typ(features, float)

    @structer_from_instance.register(QuantileTransformer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.from_names_typ(features, float)

    @structer_from_instance.register(Binarizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.from_names_typ(features, float)

    @structer_from_instance.register(KNNImputer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.from_names_typ(features, float)

    # Feature selection
    @structer_from_instance.register(SelectKBest)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        # Get types and handle KV-encoded format - extract value type from Array[Struct{key, value}]
        types = {_get_kv_value_type(t) for t in expr.select(features).schema().values()}
        (typ, *rest) = types
        if rest:
            raise ValueError
        structer = Structer.from_n_typ_prefix(n=instance.k, typ=typ)
        return structer

    # Dimensionality reduction - use sklearn's naming convention
    @structer_from_instance.register(PCA)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if not isinstance(n_components, int):
            # For 'mle', float, or None, we need the fitted model to know n_components_
            # Use KV-encoded format
            return Structer.kv_encoded(input_columns=features)
        # sklearn uses 'pca0', 'pca1', etc.
        structer = Structer.from_n_typ_prefix(n=n_components, typ=float, prefix="pca")
        return structer

    @structer_from_instance.register(TruncatedSVD)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if not isinstance(n_components, int):
            # Use KV-encoded format
            return Structer.kv_encoded(input_columns=features)
        # sklearn uses 'truncatedsvd0', 'truncatedsvd1', etc.
        structer = Structer.from_n_typ_prefix(
            n=n_components, typ=float, prefix="truncatedsvd"
        )
        return structer

    @structer_from_instance.register(KernelPCA)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if not isinstance(n_components, int):
            return Structer.kv_encoded(input_columns=features)
        return Structer.from_n_typ_prefix(n=n_components, typ=float, prefix="kernelpca")

    @structer_from_instance.register(SparsePCA)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if not isinstance(n_components, int):
            return Structer.kv_encoded(input_columns=features)
        return Structer.from_n_typ_prefix(n=n_components, typ=float, prefix="sparsepca")

    @structer_from_instance.register(MiniBatchSparsePCA)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if not isinstance(n_components, int):
            return Structer.kv_encoded(input_columns=features)
        return Structer.from_n_typ_prefix(
            n=n_components, typ=float, prefix="minibatchsparsepca"
        )

    @structer_from_instance.register(FactorAnalysis)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if not isinstance(n_components, int):
            return Structer.kv_encoded(input_columns=features)
        return Structer.from_n_typ_prefix(n=n_components, typ=float, prefix="factor")

    @structer_from_instance.register(FastICA)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if not isinstance(n_components, int):
            return Structer.kv_encoded(input_columns=features)
        return Structer.from_n_typ_prefix(n=n_components, typ=float, prefix="ica")

    @structer_from_instance.register(NMF)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if not isinstance(n_components, int):
            return Structer.kv_encoded(input_columns=features)
        return Structer.from_n_typ_prefix(n=n_components, typ=float, prefix="nmf")

    @structer_from_instance.register(MiniBatchNMF)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if not isinstance(n_components, int):
            return Structer.kv_encoded(input_columns=features)
        return Structer.from_n_typ_prefix(
            n=n_components, typ=float, prefix="minibatchnmf"
        )

    @structer_from_instance.register(LatentDirichletAllocation)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if not isinstance(n_components, int):
            return Structer.kv_encoded(input_columns=features)
        return Structer.from_n_typ_prefix(n=n_components, typ=float, prefix="lda")

    # Random projections
    @structer_from_instance.register(GaussianRandomProjection)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if n_components == "auto" or not isinstance(n_components, int):
            return Structer.kv_encoded(input_columns=features)
        return Structer.from_n_typ_prefix(
            n=n_components, typ=float, prefix="gaussianrp"
        )

    @structer_from_instance.register(SparseRandomProjection)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        if n_components == "auto" or not isinstance(n_components, int):
            return Structer.kv_encoded(input_columns=features)
        return Structer.from_n_typ_prefix(n=n_components, typ=float, prefix="sparserp")

    # Kernel approximation
    @structer_from_instance.register(Nystroem)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        return Structer.from_n_typ_prefix(n=n_components, typ=float, prefix="nystroem")

    @structer_from_instance.register(RBFSampler)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        return Structer.from_n_typ_prefix(n=n_components, typ=float, prefix="rbf")

    @structer_from_instance.register(SkewedChi2Sampler)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_components = instance.n_components
        return Structer.from_n_typ_prefix(
            n=n_components, typ=float, prefix="skewedchi2"
        )

    @structer_from_instance.register(AdditiveChi2Sampler)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        # Output size is sample_steps * 2 * n_features + n_features
        # But we use KV-encoded since it depends on input feature count at runtime
        return Structer.kv_encoded(input_columns=features)

    # PolynomialFeatures - compute output size from parameters
    @structer_from_instance.register(PolynomialFeatures)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_input = len(features)
        # Compute the number of output features using sklearn's formula
        # We can compute this without fitting by using combinations
        from math import comb

        degree = instance.degree
        interaction_only = instance.interaction_only
        include_bias = instance.include_bias

        if interaction_only:
            # Only interaction terms (no x^2, etc.)
            n_output = sum(comb(n_input, i) for i in range(1, degree + 1))
        else:
            # All polynomial terms including powers
            n_output = comb(n_input + degree, degree) - 1  # -1 for constant term

        if include_bias:
            n_output += 1

        return Structer.from_n_typ_prefix(n=n_output, typ=float, prefix="poly")

    # Feature selection with fixed k
    @structer_from_instance.register(RFE)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        n_features_to_select = instance.n_features_to_select
        if isinstance(n_features_to_select, int):
            # Get types from input
            types = {
                _get_kv_value_type(t) for t in expr.select(features).schema().values()
            }
            (typ, *rest) = types
            if rest:
                raise ValueError("RFE requires homogeneous feature types")
            return Structer.from_n_typ_prefix(n=n_features_to_select, typ=typ)
        # Dynamic selection - use KV-encoded
        return Structer.kv_encoded(input_columns=features)

    # Encoders - use KV-encoded format (output depends on categories discovered during fit)
    @structer_from_instance.register(OneHotEncoder)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    @structer_from_instance.register(OrdinalEncoder)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    @structer_from_instance.register(LabelBinarizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    @structer_from_instance.register(MultiLabelBinarizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    @structer_from_instance.register(SplineTransformer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        # Output depends on n_knots and degree, but complex to compute
        return Structer.kv_encoded(input_columns=features)

    # Text vectorizers - KV-encoded (vocabulary discovered at fit time)
    @structer_from_instance.register(CountVectorizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    @structer_from_instance.register(TfidfVectorizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    @structer_from_instance.register(HashingVectorizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        # HashingVectorizer has fixed n_features but column names are hashes
        return Structer.kv_encoded(input_columns=features)

    @structer_from_instance.register(DictVectorizer)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        return Structer.kv_encoded(input_columns=features)

    # Missing indicator - outputs binary columns for each feature
    @structer_from_instance.register(MissingIndicator)
    def _(instance, expr, features=None):
        features = features or tuple(expr.columns)
        # Output depends on which features have missing values
        return Structer.kv_encoded(input_columns=features)

    # Container types - compute schema from children
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline as SklearnPipeline

    def _normalize_columns(columns):
        """Normalize columns to tuple format."""
        if isinstance(columns, list):
            return tuple(columns)
        elif isinstance(columns, str):
            return (columns,)
        elif columns is not None:
            return tuple(columns)
        return ()

    @structer_from_instance.register(ColumnTransformer)
    def _(instance, expr, features=None):
        """
        Compute ColumnTransformer schema from children.

        - If all children have known schemas, combine them.
        - If any child is KV-encoded, that child's output is KV-encoded.
        - Passthrough columns are tracked separately.
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
                if expr is not None:
                    for col in cols:
                        combined_fields[col] = expr.schema()[col]
                continue

            all_input_cols.extend(cols)

            # Try to get child's Structer
            try:
                # Create a subset expr with just the columns for this transformer
                if expr is not None and cols:
                    child_expr = expr.select(*cols)
                else:
                    child_expr = expr

                child_structer = structer_from_instance(
                    transformer, child_expr, features=cols if cols else None
                )

                if child_structer.is_kv_encoded:
                    any_kv_encoded = True
                else:
                    # Add child's output columns to combined schema
                    combined_fields.update(child_structer.struct.fields)
            except (ValueError, AttributeError):
                # Child has no registration or requires expr we don't have
                any_kv_encoded = True

        # Handle remainder if set to passthrough
        if instance.remainder == "passthrough" and expr is not None:
            # Get columns not explicitly handled
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

        FeatureUnion concatenates outputs from all transformers.
        If all children have known schemas, combine them.
        If any child is KV-encoded, use KV-encoded for the whole output.
        """
        features = features or (tuple(expr.columns) if expr is not None else None)

        combined_fields = {}
        any_kv_encoded = False

        for name, transformer in instance.transformer_list:
            try:
                child_structer = structer_from_instance(
                    transformer, expr, features=features
                )

                if child_structer.is_kv_encoded:
                    any_kv_encoded = True
                else:
                    # Add child's output columns to combined schema
                    combined_fields.update(child_structer.struct.fields)
            except (ValueError, AttributeError):
                any_kv_encoded = True

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

        Pipeline schema is determined by its final step's output.
        We chain through each step to compute the final schema.
        """
        features = features or (tuple(expr.columns) if expr is not None else None)

        current_expr = expr
        current_features = features
        last_structer = None

        for name, transformer in instance.steps:
            try:
                child_structer = structer_from_instance(
                    transformer, current_expr, features=current_features
                )
                last_structer = child_structer

                if child_structer.is_kv_encoded:
                    # Once we hit KV-encoded, we stay KV-encoded
                    return Structer(
                        struct=None,
                        input_columns=features,
                    )

                # Update features for next step based on this step's output
                current_features = child_structer.output_columns
                # We can't easily update current_expr without actually transforming
                # So we set it to None and rely on features
                current_expr = None
            except (ValueError, AttributeError):
                return Structer(
                    struct=None,
                    input_columns=features,
                )

        # Final step's structer determines the pipeline's output schema
        if last_structer is None:
            return Structer(
                struct=None,
                input_columns=features,
            )

        return Structer(
            struct=last_structer.struct,
            passthrough_columns=last_structer.passthrough_columns,
        )
