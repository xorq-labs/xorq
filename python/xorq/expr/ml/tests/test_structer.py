import importlib

import numpy as np
import pandas as pd
import pytest


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

import xorq.api as xo
import xorq.expr.datatypes as dt


sklearn = pytest.importorskip("sklearn")

from sklearn import feature_selection, preprocessing  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.feature_selection import (  # noqa: E402
    SelectKBest,
    VarianceThreshold,
    f_classif,
)
from sklearn.impute import MissingIndicator, SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import (  # noqa: E402
    FeatureUnion,
    Pipeline,
)
from sklearn.pipeline import Pipeline as SklearnPipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from xorq.expr.ml.structer import (  # noqa: E402
    KV_ENCODED_TYPE,
    KVEncoder,
    KVField,
    Structer,
    get_schema_out,
    get_structer_out,
)


def test_kvfield_is_strenum():
    """Test KVField inherits from StrEnum."""
    assert issubclass(KVField, StrEnum)


def test_kvfield_members_are_strings():
    """Test KVField members are string instances."""
    assert isinstance(KVField.KEY, str)
    assert isinstance(KVField.VALUE, str)


def test_kvfield_string_equality():
    """Test KVField members equal their string values."""
    assert KVField.KEY == "key"
    assert KVField.VALUE == "value"


def test_kvfield_membership_by_member():
    """Test membership check works with enum members."""
    assert KVField.KEY in KVField
    assert KVField.VALUE in KVField


def test_kvfield_callable_lookup():
    """Test KVField can be called with string value to get member."""
    assert KVField("key") is KVField.KEY
    assert KVField("value") is KVField.VALUE
    with pytest.raises(ValueError):
        KVField("nonexistent")


def test_kvfield_value_attribute():
    """Test .value attribute returns the string value."""
    assert KVField.KEY.value == "key"
    assert KVField.VALUE.value == "value"


def test_kvfield_name_attribute():
    """Test .name attribute returns the member name."""
    assert KVField.KEY.name == "KEY"
    assert KVField.VALUE.name == "VALUE"


def test_kvfield_iteration():
    """Test iterating over KVField yields all members."""
    members = list(KVField)
    assert len(members) == 2
    assert KVField.KEY in members
    assert KVField.VALUE in members


def test_kvfield_as_dict_key():
    """Test KVField members work as dictionary keys."""
    d = {KVField.KEY: "key_value", KVField.VALUE: "value_value"}
    assert d["key"] == "key_value"
    assert d["value"] == "value_value"
    assert d[KVField.KEY] == "key_value"
    assert d[KVField.VALUE] == "value_value"


def test_kvfield_string_operations():
    """Test KVField members support string operations."""
    assert KVField.KEY.upper() == "KEY"
    assert KVField.VALUE.capitalize() == "Value"
    assert KVField.KEY + "_suffix" == "key_suffix"


def test_kvfield_hash_matches_string():
    """Test KVField members hash the same as their string values."""
    assert hash(KVField.KEY) == hash("key")
    assert hash(KVField.VALUE) == hash("value")


def test_kv_encoded_type_uses_kvfield_values():
    """Test KV_ENCODED_TYPE struct uses KVField.value strings as keys."""
    struct_fields = KV_ENCODED_TYPE.value_type.fields
    # Keys are plain strings (for YAML serialization compatibility)
    assert "key" in struct_fields
    assert "value" in struct_fields
    # KVField members also work due to string equality
    assert KVField.KEY in struct_fields
    assert KVField.VALUE in struct_fields


def test_kv_encoder_encode_basic():
    """Test KVEncoder.encode with a simple transformer."""

    df = pd.DataFrame({"cat": ["a", "b", "a"]})
    model = OneHotEncoder(sparse_output=False)
    model.fit(df)

    result = KVEncoder.encode(model, df)

    assert isinstance(result, pd.Series)
    assert len(result) == 3
    # Each element should be a tuple of dicts with 'key' and 'value'
    first_row = result.iloc[0]
    assert all("key" in d and "value" in d for d in first_row)


def test_kv_encoder_encode_feature_names():
    """Test that encode uses get_feature_names_out() for keys."""

    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    model = OneHotEncoder(sparse_output=False)
    model.fit(df)

    result = KVEncoder.encode(model, df)
    keys = [d["key"] for d in result.iloc[0]]

    expected_names = list(model.get_feature_names_out())
    assert keys == expected_names


def test_kv_encoder_decode_basic():
    """Test KVEncoder.decode expands series to individual columns."""
    series = pd.Series(
        [
            ({"key": "a", "value": 1.0}, {"key": "b", "value": 0.0}),
            ({"key": "a", "value": 0.0}, {"key": "b", "value": 1.0}),
        ]
    )

    result = KVEncoder.decode(series)

    assert "a" in result.columns
    assert "b" in result.columns
    assert result["a"].tolist() == [1.0, 0.0]
    assert result["b"].tolist() == [0.0, 1.0]


def test_kv_encoder_decode_empty_series_raises():
    """Test decode raises ValueError for empty series."""
    series = pd.Series([], dtype=object)
    with pytest.raises(ValueError):
        KVEncoder.decode(series)


def test_kv_encoder_is_kv_encoded_type_true():
    """Test is_kv_encoded_type returns True for KV format."""
    assert KVEncoder.is_kv_encoded_type(KV_ENCODED_TYPE)


def test_kv_encoder_is_kv_encoded_type_false():
    """Test is_kv_encoded_type returns False for non-KV types."""
    assert not KVEncoder.is_kv_encoded_type(dt.float64)
    assert not KVEncoder.is_kv_encoded_type(dt.Array(dt.float64))
    assert not KVEncoder.is_kv_encoded_type(dt.Struct({"a": dt.float64}))


def test_structer_from_names_typ():
    """Test creating Structer with known schema."""
    structer = Structer.from_names_typ(("a", "b"), dt.float64)

    assert not structer.is_kv_encoded
    assert structer.struct is not None
    assert "a" in structer.struct.fields
    assert "b" in structer.struct.fields


def test_structer_kv_encoded_factory():
    """Test creating KV-encoded Structer."""
    structer = Structer.kv_encoded(input_columns=("x", "y"))

    assert structer.is_kv_encoded
    assert structer.struct is None
    assert structer.input_columns == ("x", "y")
    assert structer.return_type == KV_ENCODED_TYPE


def test_structer_needs_target_default_false():
    """Test needs_target defaults to False."""
    structer = Structer.from_names_typ(("a",), dt.float64)
    assert structer.needs_target is False


def test_structer_needs_target_explicit():
    """Test needs_target can be set explicitly."""
    base = Structer.from_names_typ(("a",), dt.float64)
    structer = Structer(struct=base.struct, needs_target=True)
    assert structer.needs_target is True


def test_structer_is_series_default_false():
    """Test is_series defaults to False."""
    structer = Structer.from_names_typ(("a",), dt.float64)
    assert structer.is_series is False


def test_structer_is_series_explicit():
    """Test is_series can be set explicitly."""
    structer = Structer(struct=None, input_columns=("text",), is_series=True)
    assert structer.is_series is True


def test_structer_return_type_struct():
    """Test return_type for known schema."""
    structer = Structer.from_names_typ(("a", "b"), dt.float64)
    assert isinstance(structer.return_type, dt.Struct)


def test_structer_return_type_kv_encoded():
    """Test return_type for KV-encoded."""
    structer = Structer.kv_encoded(input_columns=("x",))
    assert structer.return_type == KV_ENCODED_TYPE


def test_structer_output_columns_struct():
    """Test output_columns for known schema."""
    structer = Structer.from_names_typ(("a", "b"), dt.float64)
    assert structer.output_columns == ("a", "b")


def test_structer_output_columns_kv_encoded_raises():
    """Test output_columns raises for KV-encoded."""
    structer = Structer.kv_encoded(input_columns=("x",))
    with pytest.raises(ValueError, match="KV-encoded"):
        _ = structer.output_columns


def test_structer_get_convert_array_kv_encoded_raises():
    """Test get_convert_array raises for KV-encoded."""
    structer = Structer.kv_encoded(input_columns=("x",))
    with pytest.raises(ValueError, match="KV-encoded"):
        structer.get_convert_array()


def test_structer_get_output_columns_known_schema():
    """Test get_output_columns returns dtype keys for known schema."""
    structer = Structer.from_names_typ(("a", "b"), dt.float64)
    assert structer.get_output_columns() == ("a", "b")
    # dest_col is ignored for known schema
    assert structer.get_output_columns(dest_col="foo") == ("a", "b")


def test_structer_get_output_columns_kv_encoded():
    """Test get_output_columns returns dest_col tuple for KV-encoded."""
    structer = Structer.kv_encoded(input_columns=("x",))
    assert structer.get_output_columns() == ("transformed",)
    assert structer.get_output_columns(dest_col="encoded") == ("encoded",)


def test_structer_maybe_unpack_known_schema():
    """Test maybe_unpack unpacks struct column for known schema."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
    fitted = step.fit(t, features=("a", "b"))

    # Transform returns struct column that needs unpacking
    result = fitted.transform(t)
    df = result.execute()

    # Should have unpacked columns, not a struct column
    assert "a" in df.columns
    assert "b" in df.columns
    assert "transformed" not in df.columns


def test_structer_maybe_unpack_kv_encoded():
    """Test maybe_unpack returns expr unchanged for KV-encoded."""

    t = xo.memtable({"cat": ["a", "b", "a"]})
    step = xo.Step.from_instance_name(OneHotEncoder(), name="ohe")
    fitted = step.fit(t, features=("cat",))

    # Transform returns KV-encoded column (not unpacked)
    result = fitted.transform(t)
    df = result.execute()

    # Should have KV-encoded column, not unpacked
    assert "transformed" in df.columns
    # The column contains tuples of dicts
    assert all("key" in d and "value" in d for d in df["transformed"].iloc[0])


def test_structer_from_instance_standard_scaler():
    """Test StandardScaler produces known schema."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(StandardScaler(), t, features=("a", "b"))

    assert not structer.is_kv_encoded
    assert not structer.needs_target
    assert not structer.is_series


def test_structer_from_instance_simple_imputer():
    """Test SimpleImputer produces known schema."""

    t = xo.memtable({"a": [1.0, None], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(SimpleImputer(), t, features=("a", "b"))

    assert not structer.is_kv_encoded
    assert not structer.needs_target


def test_structer_from_instance_missing_indicator_features_all():
    """Test MissingIndicator with features='all' produces known boolean schema."""

    t = xo.memtable({"a": [1.0, None], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(
        MissingIndicator(features="all"), t, features=("a", "b")
    )

    assert not structer.is_kv_encoded
    assert structer.struct == dt.Struct({"a": dt.boolean, "b": dt.boolean})
    assert not structer.needs_target


def test_structer_from_instance_missing_indicator_features_missing_only():
    """Test MissingIndicator with features='missing-only' produces KV-encoded schema."""

    t = xo.memtable({"a": [1.0, None], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(
        MissingIndicator(features="missing-only"), t, features=("a", "b")
    )

    assert structer.is_kv_encoded
    assert structer.input_columns == ("a", "b")
    assert not structer.needs_target


def test_structer_from_instance_one_hot_encoder():
    """Test OneHotEncoder produces KV-encoded schema."""

    t = xo.memtable({"cat": ["a", "b", "c"]})
    structer = Structer.from_instance_expr(OneHotEncoder(), t, features=("cat",))

    assert structer.is_kv_encoded
    assert not structer.needs_target
    assert not structer.is_series


def test_structer_from_instance_select_k_best():
    """Test SelectKBest produces stub columns when k is known."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(SelectKBest(k=1), t, features=("a", "b"))

    # Known k -> stub column names (not KV-encoded)
    assert not structer.is_kv_encoded
    assert structer.needs_target is True
    assert structer.input_columns == ("a", "b")
    assert tuple(structer.struct.names) == ("transformed_0",)


def test_structer_from_instance_tfidf_vectorizer():
    """Test TfidfVectorizer is KV-encoded and is_series."""

    t = xo.memtable({"text": ["hello world", "foo bar"]})
    structer = Structer.from_instance_expr(TfidfVectorizer(), t, features=("text",))

    assert structer.is_kv_encoded
    assert structer.is_series
    assert not structer.needs_target


def test_structer_from_instance_unregistered_raises():
    """Test unregistered transformer raises ValueError."""

    class CustomTransformer:
        pass

    t = xo.memtable({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="can't handle type"):
        Structer.from_instance_expr(CustomTransformer(), t)


def test_kv_encoder_integration_one_hot_encoder_step():
    """Test OneHotEncoder through Step interface."""

    t = xo.memtable({"cat": ["a", "b", "a", "c"]})
    step = xo.Step.from_instance_name(OneHotEncoder(), name="ohe")
    fitted = step.fit(t, features=("cat",))

    result = fitted.transform(t)
    df = result.execute()

    # Should have KV-encoded column
    assert "transformed" in df.columns

    # Decode and verify
    decoded = KVEncoder.decode(df["transformed"])
    assert "cat_a" in decoded.columns
    assert "cat_b" in decoded.columns
    assert "cat_c" in decoded.columns


def test_kv_encoder_integration_tfidf_vectorizer_step():
    """Test TfidfVectorizer through Step interface."""

    t = xo.memtable({"text": ["hello world", "foo bar", "hello foo"]})
    step = xo.Step.from_instance_name(TfidfVectorizer(), name="tfidf")
    fitted = step.fit(t, features=("text",))

    result = fitted.transform(t)
    df = result.execute()

    # Should have KV-encoded column
    assert "transformed" in df.columns

    # Decode and verify we get vocabulary columns
    decoded = KVEncoder.decode(df["transformed"])
    assert "hello" in decoded.columns
    assert "world" in decoded.columns
    assert "foo" in decoded.columns
    assert "bar" in decoded.columns


def test_kv_encoder_integration_tfidf_matches_sklearn():
    """Test TfidfVectorizer output matches sklearn."""

    texts = ["hello world", "foo bar baz", "hello foo"]
    t = xo.memtable({"text": texts})

    # xorq result
    step = xo.Step.from_instance_name(TfidfVectorizer(), name="tfidf")
    fitted = step.fit(t, features=("text",))
    result = fitted.transform(t)
    df = result.execute()
    xorq_decoded = KVEncoder.decode(df["transformed"])

    # sklearn result
    model = TfidfVectorizer()
    sklearn_result = model.fit_transform(texts).toarray()
    sklearn_df = pd.DataFrame(sklearn_result, columns=model.get_feature_names_out())

    # Compare (reorder columns to match)
    xorq_sorted = xorq_decoded[sorted(xorq_decoded.columns)]
    sklearn_sorted = sklearn_df[sorted(sklearn_df.columns)]

    pd.testing.assert_frame_equal(
        xorq_sorted.reset_index(drop=True),
        sklearn_sorted.reset_index(drop=True),
        check_exact=False,
        atol=1e-10,
    )


def test_kv_encoder_integration_select_k_best_with_target():
    """Test SelectKBest works with target column."""

    t = xo.memtable(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "target": [0, 0, 1, 1],
        }
    )

    step = xo.Step.from_instance_name(
        SelectKBest(score_func=f_classif, k=1), name="skb"
    )
    fitted = step.fit(t, features=("a", "b"), target="target")

    result = fitted.transform(t)
    df = result.execute()

    # Should have selected 1 feature
    assert len(df.columns) == 2  # target + 1 selected feature


def test_kv_encoder_integration_get_kv_value_type_non_kv():
    """Test get_kv_value_type returns original type for non-KV types."""
    typ = dt.float64
    assert KVEncoder.get_kv_value_type(typ) == dt.float64


def test_kv_encoder_integration_structer_dtype_raises_for_kv_encoded():
    """Test dtype property raises for KV-encoded Structer."""
    structer = Structer.kv_encoded(input_columns=("x",))
    with pytest.raises(ValueError, match="KV-encoded"):
        _ = structer.dtype


def test_kv_encoder_integration_select_k_best_kv_encoded_input():
    """Test SelectKBest with KV-encoded input uses KV-encoding output."""

    # Create table with KV-encoded column from OneHotEncoder
    t = xo.memtable({"cat": ["a", "b", "a"], "target": [0, 1, 0]})
    ohe_step = xo.Step.from_instance_name(OneHotEncoder(), name="ohe")
    ohe_fitted = ohe_step.fit(t, features=("cat",))
    t_encoded = ohe_fitted.transform(t)

    # SelectKBest on KV-encoded input -> KV-encoded output
    structer = Structer.from_instance_expr(
        SelectKBest(k=1), t_encoded, features=("transformed",)
    )

    assert structer.is_kv_encoded
    assert structer.needs_target


def test_scaler_transformer_parity_minmax_scaler_matches_sklearn():
    """Test MinMaxScaler output matches sklearn."""

    np.random.seed(42)
    data = pd.DataFrame(
        {
            "a": np.random.randn(10) * 100,
            "b": np.random.randn(10) * 50 + 25,
        }
    )
    t = xo.memtable(data)

    # xorq result
    step = xo.Step.from_instance_name(MinMaxScaler(), name="scaler")
    fitted = step.fit(t, features=("a", "b"))
    result = fitted.transform(t)
    xorq_df = result.execute()

    # sklearn result
    scaler = MinMaxScaler()
    sklearn_result = scaler.fit_transform(data[["a", "b"]])
    sklearn_df = pd.DataFrame(sklearn_result, columns=["a", "b"])

    pd.testing.assert_frame_equal(
        xorq_df[["a", "b"]].reset_index(drop=True),
        sklearn_df.reset_index(drop=True),
        check_exact=False,
        atol=1e-10,
    )


def test_scaler_transformer_parity_maxabs_scaler_matches_sklearn():
    """Test MaxAbsScaler output matches sklearn."""

    np.random.seed(42)
    data = pd.DataFrame(
        {
            "a": np.random.randn(10) * 100,
            "b": np.random.randn(10) * 50 + 25,
        }
    )
    t = xo.memtable(data)

    # xorq result
    step = xo.Step.from_instance_name(MaxAbsScaler(), name="scaler")
    fitted = step.fit(t, features=("a", "b"))
    result = fitted.transform(t)
    xorq_df = result.execute()

    # sklearn result
    scaler = MaxAbsScaler()
    sklearn_result = scaler.fit_transform(data[["a", "b"]])
    sklearn_df = pd.DataFrame(sklearn_result, columns=["a", "b"])

    pd.testing.assert_frame_equal(
        xorq_df[["a", "b"]].reset_index(drop=True),
        sklearn_df.reset_index(drop=True),
        check_exact=False,
        atol=1e-10,
    )


def test_scaler_transformer_parity_robust_scaler_matches_sklearn():
    """Test RobustScaler output matches sklearn."""

    np.random.seed(42)
    # Include some outliers to test robustness
    data = pd.DataFrame(
        {
            "a": np.concatenate([np.random.randn(8), [100, -100]]),
            "b": np.concatenate([np.random.randn(8) * 10, [500, -500]]),
        }
    )
    t = xo.memtable(data)

    # xorq result
    step = xo.Step.from_instance_name(RobustScaler(), name="scaler")
    fitted = step.fit(t, features=("a", "b"))
    result = fitted.transform(t)
    xorq_df = result.execute()

    # sklearn result
    scaler = RobustScaler()
    sklearn_result = scaler.fit_transform(data[["a", "b"]])
    sklearn_df = pd.DataFrame(sklearn_result, columns=["a", "b"])

    pd.testing.assert_frame_equal(
        xorq_df[["a", "b"]].reset_index(drop=True),
        sklearn_df.reset_index(drop=True),
        check_exact=False,
        atol=1e-10,
    )


def test_scaler_transformer_parity_normalizer_matches_sklearn():
    """Test Normalizer output matches sklearn."""

    np.random.seed(42)
    data = pd.DataFrame(
        {
            "a": np.random.randn(10) * 100,
            "b": np.random.randn(10) * 50,
            "c": np.random.randn(10) * 25,
        }
    )
    t = xo.memtable(data)

    # xorq result
    step = xo.Step.from_instance_name(Normalizer(norm="l2"), name="normalizer")
    fitted = step.fit(t, features=("a", "b", "c"))
    result = fitted.transform(t)
    xorq_df = result.execute()

    # sklearn result
    normalizer = Normalizer(norm="l2")
    sklearn_result = normalizer.fit_transform(data[["a", "b", "c"]])
    sklearn_df = pd.DataFrame(sklearn_result, columns=["a", "b", "c"])

    pd.testing.assert_frame_equal(
        xorq_df[["a", "b", "c"]].reset_index(drop=True),
        sklearn_df.reset_index(drop=True),
        check_exact=False,
        atol=1e-10,
    )


def test_scaler_transformer_parity_power_transformer_matches_sklearn():
    """Test PowerTransformer output matches sklearn."""

    np.random.seed(42)
    # Use positive values for yeo-johnson (works with any values, but positive is simpler)
    data = pd.DataFrame(
        {
            "a": np.abs(np.random.randn(20)) * 100 + 1,
            "b": np.abs(np.random.randn(20)) * 50 + 1,
        }
    )
    t = xo.memtable(data)

    # xorq result
    step = xo.Step.from_instance_name(
        PowerTransformer(method="yeo-johnson"), name="power"
    )
    fitted = step.fit(t, features=("a", "b"))
    result = fitted.transform(t)
    xorq_df = result.execute()

    # sklearn result
    transformer = PowerTransformer(method="yeo-johnson")
    sklearn_result = transformer.fit_transform(data[["a", "b"]])
    sklearn_df = pd.DataFrame(sklearn_result, columns=["a", "b"])

    pd.testing.assert_frame_equal(
        xorq_df[["a", "b"]].reset_index(drop=True),
        sklearn_df.reset_index(drop=True),
        check_exact=False,
        atol=1e-10,
    )


def test_scaler_transformer_parity_quantile_transformer_matches_sklearn():
    """Test QuantileTransformer output matches sklearn."""

    np.random.seed(42)
    # Need enough samples for quantile estimation
    data = pd.DataFrame(
        {
            "a": np.random.randn(100) * 100,
            "b": np.random.randn(100) * 50,
        }
    )
    t = xo.memtable(data)

    # xorq result
    step = xo.Step.from_instance_name(
        QuantileTransformer(n_quantiles=50, output_distribution="uniform"),
        name="quantile",
    )
    fitted = step.fit(t, features=("a", "b"))
    result = fitted.transform(t)
    xorq_df = result.execute()

    # sklearn result
    transformer = QuantileTransformer(n_quantiles=50, output_distribution="uniform")
    sklearn_result = transformer.fit_transform(data[["a", "b"]])
    sklearn_df = pd.DataFrame(sklearn_result, columns=["a", "b"])

    pd.testing.assert_frame_equal(
        xorq_df[["a", "b"]].reset_index(drop=True),
        sklearn_df.reset_index(drop=True),
        check_exact=False,
        atol=1e-10,
    )


def test_scaler_minmax_scaler_produces_known_schema():
    """Test MinMaxScaler produces known schema."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(MinMaxScaler(), t, features=("a", "b"))

    assert not structer.is_kv_encoded
    assert not structer.needs_target
    assert not structer.is_series
    assert "a" in structer.struct.fields
    assert "b" in structer.struct.fields


def test_scaler_maxabs_scaler_produces_known_schema():
    """Test MaxAbsScaler produces known schema."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(MaxAbsScaler(), t, features=("a", "b"))

    assert not structer.is_kv_encoded
    assert not structer.needs_target
    assert not structer.is_series


def test_scaler_robust_scaler_produces_known_schema():
    """Test RobustScaler produces known schema."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(RobustScaler(), t, features=("a", "b"))

    assert not structer.is_kv_encoded
    assert not structer.needs_target
    assert not structer.is_series


def test_scaler_normalizer_produces_known_schema():
    """Test Normalizer produces known schema."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(Normalizer(), t, features=("a", "b"))

    assert not structer.is_kv_encoded
    assert not structer.needs_target
    assert not structer.is_series


def test_scaler_power_transformer_produces_known_schema():
    """Test PowerTransformer produces known schema."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(PowerTransformer(), t, features=("a", "b"))

    assert not structer.is_kv_encoded
    assert not structer.needs_target
    assert not structer.is_series


def test_scaler_quantile_transformer_produces_known_schema():
    """Test QuantileTransformer produces known schema."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    structer = Structer.from_instance_expr(
        QuantileTransformer(), t, features=("a", "b")
    )

    assert not structer.is_kv_encoded
    assert not structer.needs_target
    assert not structer.is_series


@pytest.fixture
def one_to_one_numeric_data():
    """Generate numeric data for scalers."""

    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "a": np.random.randn(n_samples) * 100,
            "b": np.random.randn(n_samples) * 50 + 25,
        }
    )


@pytest.fixture
def one_to_one_positive_data():
    """Generate positive numeric data for PowerTransformer box-cox."""

    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "a": np.abs(np.random.randn(n_samples)) * 100 + 1,
            "b": np.abs(np.random.randn(n_samples)) * 50 + 1,
        }
    )


@pytest.mark.parametrize(
    "transformer_cls,transformer_kwargs,requires_positive",
    [
        pytest.param("StandardScaler", {}, False, id="StandardScaler"),
        pytest.param("MinMaxScaler", {}, False, id="MinMaxScaler"),
        pytest.param("MaxAbsScaler", {}, False, id="MaxAbsScaler"),
        pytest.param("RobustScaler", {}, False, id="RobustScaler"),
        pytest.param("Normalizer", {"norm": "l2"}, False, id="Normalizer"),
        pytest.param(
            "PowerTransformer",
            {"method": "yeo-johnson"},
            False,
            id="PowerTransformer-yeojohnson",
        ),
        pytest.param(
            "PowerTransformer",
            {"method": "box-cox"},
            True,
            id="PowerTransformer-boxcox",
        ),
        pytest.param(
            "QuantileTransformer",
            {"n_quantiles": 50, "output_distribution": "uniform"},
            False,
            id="QuantileTransformer",
        ),
        pytest.param("Binarizer", {"threshold": 0.0}, False, id="Binarizer"),
    ],
)
def test_one_to_one_feature_mixin_structer_and_parity(
    one_to_one_numeric_data,
    one_to_one_positive_data,
    transformer_cls,
    transformer_kwargs,
    requires_positive,
):
    """Test OneToOneFeatureMixin estimators: structer schema and sklearn parity."""

    data = one_to_one_positive_data if requires_positive else one_to_one_numeric_data
    t = xo.memtable(data)
    features = ("a", "b")

    TransformerClass = getattr(preprocessing, transformer_cls)
    transformer = TransformerClass(**transformer_kwargs)

    # Test 1: Structer produces known schema (not KV-encoded)
    structer = Structer.from_instance_expr(transformer, t, features=features)

    assert not structer.is_kv_encoded, f"{transformer_cls} should not be KV-encoded"
    assert not structer.needs_target, f"{transformer_cls} should not need target"
    assert not structer.is_series, f"{transformer_cls} should not be series"
    assert "a" in structer.struct.fields, f"{transformer_cls} should preserve 'a'"
    assert "b" in structer.struct.fields, f"{transformer_cls} should preserve 'b'"

    # Test 2: Parity with sklearn output
    step = xo.Step.from_instance_name(
        TransformerClass(**transformer_kwargs), name="transformer"
    )
    fitted = step.fit(t, features=features)
    result = fitted.transform(t)
    xorq_df = result.execute()

    sklearn_transformer = TransformerClass(**transformer_kwargs)
    sklearn_result = sklearn_transformer.fit_transform(data[list(features)])
    sklearn_df = pd.DataFrame(sklearn_result, columns=list(features))

    pd.testing.assert_frame_equal(
        xorq_df[list(features)].reset_index(drop=True),
        sklearn_df.reset_index(drop=True),
        check_exact=False,
        atol=1e-10,
    )


def test_column_transformer_known_schema_all_transformers():
    """Test ColumnTransformer with all known-schema transformers."""

    t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["num1", "num2"]),
        ]
    )
    structer = Structer.from_instance_expr(ct, t)

    # is_kv_encoded is for leaf transformers only
    assert not structer.struct_has_kv_fields
    assert structer.struct is not None
    # sklearn-style prefixes
    assert "scaler__num1" in structer.struct.fields
    assert "scaler__num2" in structer.struct.fields


def test_column_transformer_kv_encoded_transformer():
    """Test ColumnTransformer with KV-encoded transformer."""

    t = xo.memtable({"cat": ["a", "b", "c"]})
    ct = ColumnTransformer(
        [
            ("encoder", OneHotEncoder(), ["cat"]),
        ]
    )
    structer = Structer.from_instance_expr(ct, t)

    # Containers use struct_has_kv_fields (has KV column in hybrid struct)
    assert structer.struct_has_kv_fields
    assert structer.input_columns == ("cat",)
    # KV-encoded child produces named KV column
    assert "encoder" in structer.struct.fields


def test_column_transformer_mixed_known_and_kv_encoded():
    """Test ColumnTransformer with mixed known-schema and KV-encoded."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["num"]),
            ("encoder", OneHotEncoder(), ["cat"]),
        ]
    )
    structer = Structer.from_instance_expr(ct, t)

    # Hybrid output: struct set with some KV columns
    assert structer.struct_has_kv_fields
    # Known-schema column with prefix
    assert "scaler__num" in structer.struct.fields
    # KV-encoded column named after transformer
    assert "encoder" in structer.struct.fields
    # Only cat feeds into KV transformer
    assert structer.input_columns == ("cat",)


def test_column_transformer_passthrough_explicit():
    """Test ColumnTransformer with explicit passthrough."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["num"]),
            ("pass", "passthrough", ["cat"]),
        ]
    )
    structer = Structer.from_instance_expr(ct, t)

    assert not structer.struct_has_kv_fields
    # sklearn-style prefixes
    assert "scaler__num" in structer.struct.fields
    assert "pass__cat" in structer.struct.fields


def test_column_transformer_remainder_passthrough():
    """Test ColumnTransformer with remainder='passthrough'."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"], "other": [3.0, 4.0]})
    ct = ColumnTransformer(
        [("scaler", StandardScaler(), ["num"])],
        remainder="passthrough",
    )
    structer = Structer.from_instance_expr(ct, t)

    assert not structer.struct_has_kv_fields
    # sklearn-style prefixes
    assert "scaler__num" in structer.struct.fields
    assert "remainder__cat" in structer.struct.fields
    assert "remainder__other" in structer.struct.fields


def test_column_transformer_remainder_passthrough_with_kv_encoded():
    """Test ColumnTransformer with remainder='passthrough' and KV-encoded transformer."""

    t = xo.memtable({"cat": ["a", "b"], "num1": [1.0, 2.0], "num2": [3.0, 4.0]})
    ct = ColumnTransformer(
        [("encoder", OneHotEncoder(), ["cat"])],
        remainder="passthrough",
    )
    structer = Structer.from_instance_expr(ct, t)

    assert structer.struct_has_kv_fields
    assert structer.input_columns == ("cat",)


def test_column_transformer_drop_transformer():
    """Test ColumnTransformer with drop transformer."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["num"]),
            ("drop", "drop", ["cat"]),
        ]
    )
    structer = Structer.from_instance_expr(ct, t)

    assert not structer.struct_has_kv_fields
    # sklearn-style prefixes
    assert "scaler__num" in structer.struct.fields
    assert "cat" not in structer.struct.fields
    assert "drop__cat" not in structer.struct.fields


def test_column_transformer_nested_column_transformer():
    """Test nested ColumnTransformer (ColumnTransformer inside ColumnTransformer)."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    inner_ct = ColumnTransformer(
        [
            ("scaler_inner", StandardScaler(), ["a", "b"]),
        ]
    )
    outer_ct = ColumnTransformer(
        [
            ("inner", inner_ct, ["a", "b"]),
            ("scaler_c", StandardScaler(), ["c"]),
        ]
    )
    structer = Structer.from_instance_expr(outer_ct, t)

    # Both inner and outer have known schemas
    assert not structer.struct_has_kv_fields
    # Nested sklearn-style prefixes: outer__inner__col
    assert "inner__scaler_inner__a" in structer.struct.fields
    assert "inner__scaler_inner__b" in structer.struct.fields
    assert "scaler_c__c" in structer.struct.fields


def test_column_transformer_nested_column_transformer_with_kv_encoded():
    """Test nested ColumnTransformer where inner has KV-encoded."""

    t = xo.memtable({"cat": ["a", "b"], "num": [1.0, 2.0]})
    inner_ct = ColumnTransformer(
        [
            ("encoder", OneHotEncoder(), ["cat"]),
        ]
    )
    outer_ct = ColumnTransformer(
        [
            ("inner", inner_ct, ["cat"]),
            ("scaler", StandardScaler(), ["num"]),
        ]
    )
    structer = Structer.from_instance_expr(outer_ct, t)

    # Inner is KV-encoded, so outer has KV column
    assert structer.struct_has_kv_fields
    # Inner CT with KV becomes a single KV column named "inner"
    assert "inner" in structer.struct.fields
    assert "scaler__num" in structer.struct.fields


def test_column_transformer_unregistered_child_raises():
    """Test ColumnTransformer with unregistered child transformer raises."""

    class CustomTransformer(BaseEstimator, TransformerMixin):
        """Custom transformer not registered with structer_from_instance."""

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    t = xo.memtable({"num": [1.0, 2.0]})
    ct = ColumnTransformer(
        [
            ("custom", CustomTransformer(), ["num"]),
        ]
    )

    with pytest.raises(ValueError, match="can't handle type"):
        Structer.from_instance_expr(ct, t)


# Tests for _normalize_columns (column spec normalization)


def test_column_transformer_normalize_columns_list():
    """Test _normalize_columns handles list -> tuple."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["a", "b"]),  # list
        ]
    )
    structer = Structer.from_instance_expr(ct, t)

    assert not structer.struct_has_kv_fields
    # sklearn-style prefixes
    assert "scaler__a" in structer.struct.fields
    assert "scaler__b" in structer.struct.fields


def test_column_transformer_normalize_columns_string():
    """Test _normalize_columns handles string -> single-element tuple."""

    t = xo.memtable({"num": [1.0, 2.0], "other": [3.0, 4.0]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), "num"),  # string, not list
        ]
    )
    structer = Structer.from_instance_expr(ct, t)

    assert not structer.struct_has_kv_fields
    # sklearn-style prefix
    assert "scaler__num" in structer.struct.fields


def test_column_transformer_normalize_columns_tuple():
    """Test _normalize_columns handles tuple -> tuple (passthrough)."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ("a", "b")),  # tuple, not list
        ]
    )
    structer = Structer.from_instance_expr(ct, t)

    assert not structer.struct_has_kv_fields
    # sklearn-style prefixes
    assert "scaler__a" in structer.struct.fields
    assert "scaler__b" in structer.struct.fields


def test_column_transformer_normalize_columns_none():
    """Test _normalize_columns handles None -> empty tuple."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["a"]),
            ("pass", "passthrough", None),  # None columns
        ]
    )
    structer = Structer.from_instance_expr(ct, t)

    assert not structer.struct_has_kv_fields
    # sklearn-style prefix
    assert "scaler__a" in structer.struct.fields


def test_column_transformer_normalize_columns_invalid_type_raises():
    """Test _normalize_columns raises TypeError for unsupported types."""

    t = xo.memtable({"a": [1.0, 2.0]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), 123),  # invalid type
        ]
    )

    with pytest.raises(TypeError, match="Unsupported columns type"):
        Structer.from_instance_expr(ct, t)


def test_feature_union_known_schema_all_transformers():
    """Test FeatureUnion with all known-schema transformers."""

    t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
    fu = FeatureUnion(
        [
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer()),
        ]
    )
    structer = Structer.from_instance_expr(fu, t, features=("num1", "num2"))

    assert not structer.struct_has_kv_fields
    assert structer.struct is not None
    # sklearn-style prefixes
    assert "scaler__num1" in structer.struct.fields
    assert "scaler__num2" in structer.struct.fields
    assert "imputer__num1" in structer.struct.fields
    assert "imputer__num2" in structer.struct.fields


def test_feature_union_kv_encoded_transformer():
    """Test FeatureUnion with KV-encoded transformer."""

    t = xo.memtable({"cat": ["a", "b", "c"]})
    fu = FeatureUnion(
        [
            ("encoder", OneHotEncoder()),
        ]
    )
    structer = Structer.from_instance_expr(fu, t, features=("cat",))

    # Containers use struct_has_kv_fields
    assert structer.struct_has_kv_fields
    assert structer.input_columns == ("cat",)
    # KV-encoded child produces named KV column
    assert "encoder" in structer.struct.fields


def test_feature_union_mixed_known_and_kv_encoded():
    """Test FeatureUnion with mixed known-schema and KV-encoded."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
    fu = FeatureUnion(
        [
            ("scaler", StandardScaler()),
            ("encoder", OneHotEncoder()),
        ]
    )
    structer = Structer.from_instance_expr(fu, t, features=("num", "cat"))

    # Hybrid output: struct set with some KV columns
    assert structer.struct_has_kv_fields
    assert set(structer.input_columns) == {"num", "cat"}
    # Known-schema columns with prefix
    assert "scaler__num" in structer.struct.fields
    assert "scaler__cat" in structer.struct.fields
    # KV-encoded column named after transformer
    assert "encoder" in structer.struct.fields


def test_feature_union_unregistered_child_raises():
    """Test FeatureUnion with unregistered child transformer raises."""

    class CustomTransformer(BaseEstimator, TransformerMixin):
        """Custom transformer not registered with structer_from_instance."""

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    t = xo.memtable({"num": [1.0, 2.0]})
    fu = FeatureUnion(
        [
            ("custom", CustomTransformer()),
        ]
    )

    with pytest.raises(ValueError, match="can't handle type"):
        Structer.from_instance_expr(fu, t, features=("num",))


def test_feature_union_nested_feature_union():
    """Test nested FeatureUnion."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    inner_fu = FeatureUnion(
        [
            ("scaler1", StandardScaler()),
        ]
    )
    outer_fu = FeatureUnion(
        [
            ("inner", inner_fu),
            ("scaler2", StandardScaler()),
        ]
    )
    structer = Structer.from_instance_expr(outer_fu, t, features=("a", "b"))

    assert not structer.struct_has_kv_fields
    # Nested sklearn-style prefixes
    assert "inner__scaler1__a" in structer.struct.fields
    assert "inner__scaler1__b" in structer.struct.fields
    assert "scaler2__a" in structer.struct.fields
    assert "scaler2__b" in structer.struct.fields


def test_sklearn_pipeline_known_schema_pipeline():
    """Test Pipeline with all known-schema transformers."""

    t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
        ]
    )
    structer = Structer.from_instance_expr(pipe, t, features=("num1", "num2"))

    assert not structer.struct_has_kv_fields
    assert structer.struct is not None
    # Pipeline doesn't add prefixes - just chains through
    assert "num1" in structer.struct.fields
    assert "num2" in structer.struct.fields


def test_sklearn_pipeline_kv_encoded_pipeline():
    """Test Pipeline with KV-encoded transformer."""

    t = xo.memtable({"cat": ["a", "b", "c"]})
    pipe = Pipeline(
        [
            ("encoder", OneHotEncoder()),
        ]
    )
    structer = Structer.from_instance_expr(pipe, t, features=("cat",))

    # Containers use struct_has_kv_fields
    assert structer.struct_has_kv_fields
    assert structer.input_columns == ("cat",)
    # Pipeline wraps KV output in "encoded" column
    assert "encoded" in structer.struct.fields


def test_sklearn_pipeline_kv_encoded_early_exit():
    """Test Pipeline returns KV-encoded once any step is KV-encoded."""

    t = xo.memtable({"cat": ["a", "b"]})
    # Even if there are steps after OneHotEncoder, we return KV-encoded
    pipe = Pipeline(
        [
            ("encoder", OneHotEncoder()),
            # StandardScaler can't actually follow OneHotEncoder in practice,
            # but for schema computation we exit early at KV-encoded
        ]
    )
    structer = Structer.from_instance_expr(pipe, t, features=("cat",))

    assert structer.struct_has_kv_fields


def test_sklearn_pipeline_nested_pipeline():
    """Test nested Pipeline (Pipeline inside Pipeline)."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    inner_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )
    outer_pipe = Pipeline(
        [
            ("inner", inner_pipe),
            ("imputer", SimpleImputer()),
        ]
    )
    structer = Structer.from_instance_expr(outer_pipe, t, features=("a", "b"))

    assert not structer.struct_has_kv_fields
    assert "a" in structer.struct.fields
    assert "b" in structer.struct.fields


def test_sklearn_pipeline_with_column_transformer():
    """Test Pipeline containing ColumnTransformer."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
    pipe = Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [
                        ("scaler", StandardScaler(), ["num"]),
                    ],
                    remainder="passthrough",
                ),
            ),
        ]
    )
    structer = Structer.from_instance_expr(pipe, t)

    assert not structer.struct_has_kv_fields
    # sklearn-style prefixes from CT
    assert "scaler__num" in structer.struct.fields
    assert "remainder__cat" in structer.struct.fields


def test_sklearn_pipeline_with_kv_column_transformer():
    """Test Pipeline containing ColumnTransformer with KV-encoded child."""

    t = xo.memtable({"cat": ["a", "b"], "num": [1.0, 2.0]})
    pipe = Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [
                        ("encoder", OneHotEncoder(), ["cat"]),
                    ],
                    remainder="passthrough",
                ),
            ),
        ]
    )
    structer = Structer.from_instance_expr(pipe, t)

    # Hybrid output
    assert structer.struct_has_kv_fields
    # KV column named "encoder", passthrough column with prefix
    assert "encoder" in structer.struct.fields
    assert "remainder__num" in structer.struct.fields


def test_sklearn_pipeline_unregistered_step_raises():
    """Test Pipeline with unregistered transformer raises."""

    class CustomTransformer(BaseEstimator, TransformerMixin):
        """Custom transformer not registered with structer_from_instance."""

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    t = xo.memtable({"num": [1.0, 2.0]})
    pipe = Pipeline(
        [
            ("custom", CustomTransformer()),
        ]
    )

    with pytest.raises(ValueError, match="can't handle type"):
        Structer.from_instance_expr(pipe, t, features=("num",))


def test_sklearn_pipeline_empty_pipeline_returns_input_schema():
    """Test empty Pipeline returns input schema (passthrough)."""

    t = xo.memtable({"num": [1.0, 2.0]})
    pipe = Pipeline([])

    # Empty pipeline is effectively passthrough
    structer = Structer.from_instance_expr(pipe, t, features=("num",))
    assert not structer.struct_has_kv_fields
    assert "num" in structer.struct.fields


def test_sklearn_pipeline_passthrough_step():
    """Test Pipeline with 'passthrough' step."""

    t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
    pipe = Pipeline(
        [
            ("pass", "passthrough"),
        ]
    )
    structer = Structer.from_instance_expr(pipe, t, features=("num1", "num2"))

    assert not structer.struct_has_kv_fields
    assert "num1" in structer.struct.fields
    assert "num2" in structer.struct.fields


def test_sklearn_pipeline_nested_pipeline_with_passthrough():
    """Test Pipeline(Pipeline()) where inner pipeline has passthrough columns."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
    inner_pipe = Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [
                        ("scaler", StandardScaler(), ["num"]),
                    ],
                    remainder="passthrough",
                ),
            ),
        ]
    )
    outer_pipe = Pipeline(
        [
            ("inner", inner_pipe),
        ]
    )
    structer = Structer.from_instance_expr(outer_pipe, t)

    assert not structer.struct_has_kv_fields
    # sklearn-style prefixes from CT
    assert "scaler__num" in structer.struct.fields
    assert "remainder__cat" in structer.struct.fields


def test_sklearn_pipeline_chained_feature_transforms():
    """Test Pipeline where features change between steps."""

    t = xo.memtable(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "target": [0, 0, 1, 1],
        }
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(k=1)),
        ]
    )
    structer = Structer.from_instance_expr(pipe, t, features=("a", "b"))

    # SelectKBest reduces to k=1 features
    assert not structer.is_kv_encoded
    assert len(structer.struct.fields) == 1


def test_get_structer_out_unregistered_base_estimator_raises_value_error():
    """Test unregistered BaseEstimator subclass raises ValueError."""

    class UnregisteredEstimator(BaseEstimator):
        pass

    t = xo.memtable({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="can't handle type"):
        get_structer_out(UnregisteredEstimator(), t)


def test_get_structer_out_non_base_estimator_raises_type_error():
    """Test non-BaseEstimator types raise TypeError."""

    class NotAnEstimator:
        pass

    t = xo.memtable({"a": [1.0, 2.0]})
    with pytest.raises(TypeError, match="Unexpected type in get_structer_out"):
        get_structer_out(NotAnEstimator(), t)


def test_get_structer_out_primitive_type_raises_type_error():
    """Test primitive types raise TypeError."""

    t = xo.memtable({"a": [1.0, 2.0]})
    with pytest.raises(TypeError, match="Unexpected type in get_structer_out"):
        get_structer_out("not an estimator", t)


def test_get_structer_out_none_raises_type_error():
    """Test None raises TypeError."""

    t = xo.memtable({"a": [1.0, 2.0]})
    with pytest.raises(TypeError, match="Unexpected type in get_structer_out"):
        get_structer_out(None, t)


def test_decode_encoded_columns_no_encoded_cols():
    """Test passthrough when no encoded columns."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    features = ("a", "b")
    encoded_cols = ()

    result_df, result_features = KVEncoder.decode_encoded_columns(
        df, features, encoded_cols
    )

    pd.testing.assert_frame_equal(result_df, df)
    assert result_features == features


def test_decode_encoded_columns_decode_encoded_column():
    """Test decoding KV-encoded column."""
    df = pd.DataFrame(
        {
            "encoded": [
                ({"key": "x", "value": 1.0}, {"key": "y", "value": 2.0}),
                ({"key": "x", "value": 3.0}, {"key": "y", "value": 4.0}),
            ],
            "other": [10, 20],
        }
    )
    features = ("encoded", "other")
    encoded_cols = ("encoded",)

    result_df, result_features = KVEncoder.decode_encoded_columns(
        df, features, encoded_cols
    )

    assert "encoded" not in result_df.columns
    assert "x" in result_df.columns
    assert "y" in result_df.columns
    assert "other" in result_df.columns
    assert set(result_features) == {"x", "y", "other"}


def test_decode_encoded_columns_missing_encoded_col_raises():
    """Test that missing encoded columns raise ValueError."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    features = ("a",)

    with pytest.raises(ValueError, match="nonexistent not in DataFrame"):
        KVEncoder.decode_encoded_column(df, features, "nonexistent")


def test_decode_encoded_columns_empty_encoded_col_raises():
    """Test that empty encoded column raises ValueError."""
    df = pd.DataFrame({"encoded": []})
    features = ("encoded",)

    with pytest.raises(ValueError, match="cannot decode empty column"):
        KVEncoder.decode_encoded_column(df, features, "encoded")


def test_get_schema_out_column_transformer_basic():
    """Test get_schema_out with ColumnTransformer uses sklearn-style prefixes."""

    t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["num1", "num2"]),
        ]
    )
    schema = get_schema_out(ct, t)

    # sklearn-style prefixing: name__col
    assert "scaler__num1" in schema.names
    assert "scaler__num2" in schema.names


def test_get_schema_out_column_transformer_passthrough():
    """Test get_schema_out with ColumnTransformer passthrough uses prefixes."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["num"]),
            ("pass", "passthrough", ["cat"]),
        ]
    )
    schema = get_schema_out(ct, t)

    assert "scaler__num" in schema.names
    assert "pass__cat" in schema.names


def test_get_schema_out_column_transformer_remainder_passthrough():
    """Test get_schema_out with ColumnTransformer remainder='passthrough'."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"], "other": [3.0, 4.0]})
    ct = ColumnTransformer(
        [("scaler", StandardScaler(), ["num"])],
        remainder="passthrough",
    )
    schema = get_schema_out(ct, t, features=("num", "cat", "other"))

    assert "scaler__num" in schema.names
    # remainder columns get 'remainder__' prefix
    assert "remainder__cat" in schema.names
    assert "remainder__other" in schema.names


def test_get_schema_out_column_transformer_drop():
    """Test get_schema_out with ColumnTransformer drop."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["num"]),
            ("drop", "drop", ["cat"]),
        ]
    )
    schema = get_schema_out(ct, t)

    assert "scaler__num" in schema.names
    # Dropped columns should not appear
    assert "cat" not in schema.names
    assert "drop__cat" not in schema.names


def test_get_schema_out_feature_union():
    """Test get_schema_out with FeatureUnion uses prefixes."""

    t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
    fu = FeatureUnion(
        [
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer()),
        ]
    )
    schema = get_schema_out(fu, t, features=("num1", "num2"))

    # FeatureUnion also uses prefixes
    assert "scaler__num1" in schema.names
    assert "scaler__num2" in schema.names
    assert "imputer__num1" in schema.names
    assert "imputer__num2" in schema.names


def test_get_schema_out_sklearn_pipeline():
    """Test get_schema_out with sklearn Pipeline (no prefixes for pipeline)."""

    t = xo.memtable({"num1": [1.0, 2.0], "num2": [3.0, 4.0]})
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
        ]
    )
    schema = get_schema_out(pipe, t, features=("num1", "num2"))

    # Pipeline doesn't add prefixes - just chains through
    assert "num1" in schema.names
    assert "num2" in schema.names


def test_get_schema_out_base_estimator():
    """Test get_schema_out with simple BaseEstimator (no prefix)."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    scaler = StandardScaler()
    schema = get_schema_out(scaler, t, features=("a", "b"))

    # Leaf estimators don't add prefixes
    assert "a" in schema.names
    assert "b" in schema.names


def test_get_schema_out_nested_column_transformer():
    """Test get_schema_out with nested ColumnTransformer uses nested prefixes."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    inner_ct = ColumnTransformer(
        [
            ("scaler_inner", StandardScaler(), ["a", "b"]),
        ]
    )
    outer_ct = ColumnTransformer(
        [
            ("inner", inner_ct, ["a", "b"]),
            ("scaler_c", StandardScaler(), ["c"]),
        ]
    )
    schema = get_schema_out(outer_ct, t)

    # Nested prefixes: outer__inner__col
    assert "inner__scaler_inner__a" in schema.names
    assert "inner__scaler_inner__b" in schema.names
    assert "scaler_c__c" in schema.names


def test_get_schema_out_pipeline_with_column_transformer():
    """Test get_schema_out with Pipeline containing ColumnTransformer."""

    t = xo.memtable({"num": [1.0, 2.0], "cat": ["a", "b"]})
    pipe = Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [
                        ("scaler", StandardScaler(), ["num"]),
                    ],
                    remainder="passthrough",
                ),
            ),
        ]
    )
    schema = get_schema_out(pipe, t, features=("num", "cat"))

    # Pipeline chains through, CT adds prefixes
    assert "scaler__num" in schema.names
    assert "remainder__cat" in schema.names


def test_get_schema_out_unsupported_type_raises():
    """Test get_schema_out raises TypeError for non-BaseEstimator types."""

    t = xo.memtable({"a": [1.0, 2.0]})

    with pytest.raises(TypeError, match="Unexpected type in get_structer_out"):
        get_schema_out("invalid", t)


def test_get_schema_out_kv_encoded_deeply_nested_pipeline():
    """Test get_schema_out with hybrid output - known-schema + KV-encoded columns.

    Pipeline structure:
    - ColumnTransformer (hybrid output)
      - FeatureUnion (known-schema: numeric features)
        - Pipeline (SimpleImputer -> StandardScaler)
        - Pipeline (SimpleImputer -> StandardScaler)
      - Pipeline (KV-encoded: categorical via OneHotEncoder)
    """

    np.random.seed(42)
    n_samples = 20

    data = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n_samples).astype(float),
            "income": np.random.randint(20000, 150000, n_samples).astype(float),
            "education": np.random.choice(
                ["high_school", "bachelor", "master"], n_samples
            ),
            "region": np.random.choice(["north", "south", "east"], n_samples),
        }
    )

    numeric_features = ["age", "income"]
    categorical_features = ["education", "region"]
    all_features = tuple(numeric_features + categorical_features)

    # Build nested sklearn pipeline
    scaled_pipeline = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    imputed_pipeline = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_union = FeatureUnion(
        [
            ("scaled", scaled_pipeline),
            ("imputed", imputed_pipeline),
        ]
    )

    categorical_pipeline = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_union, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    expr = xo.memtable(data)
    schema = get_schema_out(preprocessor, expr, features=all_features)

    # Hybrid output: known-schema numeric columns with sklearn-style prefixes
    # FeatureUnion: numeric__scaled__age, numeric__scaled__income, numeric__imputed__age, etc.
    assert "numeric__scaled__age" in schema.names
    assert "numeric__scaled__income" in schema.names
    assert "numeric__imputed__age" in schema.names
    assert "numeric__imputed__income" in schema.names

    # KV-encoded categorical gets single column named after transformer
    assert "categorical" in schema.names
    assert schema["categorical"] == KV_ENCODED_TYPE


def test_get_schema_out_non_kv_deeply_nested_pipeline():
    """Test get_schema_out with depth-4 nested pipeline with all known-schema transformers.

    Pipeline structure:
    - ColumnTransformer (known schema - no KV-encoded children)
      - Pipeline (SimpleImputer -> StandardScaler -> Pipeline)
        - Pipeline (SimpleImputer -> StandardScaler)
      - Pipeline (SimpleImputer -> StandardScaler)
    """

    np.random.seed(42)
    n_samples = 20

    data = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n_samples).astype(float),
            "income": np.random.randint(20000, 150000, n_samples).astype(float),
            "credit_score": np.random.randint(300, 850, n_samples).astype(float),
            "years_employed": np.random.randint(0, 40, n_samples).astype(float),
            "debt_ratio": np.random.uniform(0, 1, n_samples),
            "savings": np.random.randint(0, 100000, n_samples).astype(float),
        }
    )

    numeric_features_a = ["age", "income", "credit_score"]
    numeric_features_b = ["years_employed", "debt_ratio", "savings"]
    all_features = tuple(numeric_features_a + numeric_features_b)

    # Build nested sklearn pipeline (depth 4)
    inner_pipeline = SklearnPipeline(
        [
            ("imputer2", SimpleImputer(strategy="mean")),
            ("scaler2", StandardScaler()),
        ]
    )

    numeric_a_pipeline = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("inner", inner_pipeline),
        ]
    )

    numeric_b_pipeline = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric_a", numeric_a_pipeline, numeric_features_a),
            ("numeric_b", numeric_b_pipeline, numeric_features_b),
        ]
    )

    expr = xo.memtable(data)
    schema = get_schema_out(preprocessor, expr, features=all_features)

    # All features should be in the output schema with sklearn-style prefixes
    # Pipeline doesn't add prefixes, so CT adds: numeric_a__col, numeric_b__col
    for feature in numeric_features_a:
        assert f"numeric_a__{feature}" in schema.names, (
            f"numeric_a__{feature} not in schema"
        )
    for feature in numeric_features_b:
        assert f"numeric_b__{feature}" in schema.names, (
            f"numeric_b__{feature} not in schema"
        )

    # Should have exactly the same number of columns
    assert len(schema.names) == len(all_features)


def test_get_schema_out_multiple_kv_encoded_columns():
    """Test that multiple KV-encoded transformers produce separate KV columns.

    Two OneHotEncoders should produce two separate KV-encoded columns,
    not pollute the entire output into one KV column.
    """

    np.random.seed(42)
    n_samples = 20

    data = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n_samples).astype(float),
            "income": np.random.randint(20000, 150000, n_samples).astype(float),
            "education": np.random.choice(
                ["high_school", "bachelor", "master"], n_samples
            ),
            "region": np.random.choice(["north", "south", "east"], n_samples),
        }
    )

    # Two separate OneHotEncoders for different categorical columns
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age", "income"]),
            ("cat1", OneHotEncoder(sparse_output=False), ["education"]),
            ("cat2", OneHotEncoder(sparse_output=False), ["region"]),
        ]
    )

    expr = xo.memtable(data)
    schema = get_schema_out(ct, expr)

    # Known-schema numeric columns with prefixes
    assert "num__age" in schema.names
    assert "num__income" in schema.names

    # Each KV-encoded transformer gets its own column (no pollution!)
    assert "cat1" in schema.names
    assert "cat2" in schema.names
    assert schema["cat1"] == KV_ENCODED_TYPE
    assert schema["cat2"] == KV_ENCODED_TYPE

    # Should have exactly 4 columns: 2 numeric + 2 KV
    assert len(schema.names) == 4


def test_get_schema_out_hybrid_output_preserves_types():
    """Test that hybrid output preserves correct types for each column."""

    data = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "c"],
            "other": [4, 5, 6],  # int column
        }
    )

    ct = ColumnTransformer(
        [
            ("scaler", StandardScaler(), ["num"]),
            ("encoder", OneHotEncoder(sparse_output=False), ["cat"]),
            ("pass", "passthrough", ["other"]),
        ]
    )

    expr = xo.memtable(data)
    schema = get_schema_out(ct, expr)

    # Check types are preserved
    assert schema["scaler__num"] == dt.float64
    assert schema["encoder"] == KV_ENCODED_TYPE
    assert schema["pass__other"] == dt.int64


def test_convert_struct_with_kv_kv_then_non_kv_slicing():
    """Test slicing when KV field precedes non-KV field.

    This catches a potential bug where end = start + (start + 1) instead of
    end = start + 1 for non-KV fields. With incorrect logic:
    - cat field: start=0, end=0+2=2 (correct)
    - num field: start=2, end=2+(2+1)=5 (WRONG, should be 3)

    The non-KV field would get wrong values or cause index errors.
    """

    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a"],
            "num": [1.0, 2.0, 3.0],
        }
    )

    ct = ColumnTransformer(
        [
            ("cat", OneHotEncoder(sparse_output=False), ["cat"]),
            ("num", StandardScaler(), ["num"]),
        ]
    )
    ct.fit(df)

    # Get structer and verify it has KV fields
    expr = xo.memtable(df)
    structer = get_structer_out(ct, expr)
    assert structer.struct_has_kv_fields

    # Struct fields: "cat" for KV (named after transformer), "num__num" for scalar
    assert "cat" in structer.struct.fields
    assert "num__num" in structer.struct.fields

    # Use convert_struct_with_kv to convert the output
    convert_fn = structer.get_convert_struct_with_kv()
    result = convert_fn(ct, df)

    # Verify we got 3 rows
    assert len(result) == 3

    # Verify KV field has correct structure (2 categories: a, b)
    for row in result:
        assert "cat" in row
        assert "num__num" in row
        # KV field should be tuple of dicts
        assert isinstance(row["cat"], tuple)
        assert len(row["cat"]) == 2  # a and b categories
        # num field should be a float (scalar)
        assert isinstance(row["num__num"], float)

    # Verify actual values for num field match StandardScaler output
    sklearn_result = ct.transform(df)
    # num is the last column (index 2) in sklearn output
    expected_num_values = sklearn_result[:, 2].tolist()
    actual_num_values = [row["num__num"] for row in result]
    assert actual_num_values == pytest.approx(expected_num_values)


def test_convert_struct_with_kv_non_kv_then_kv_slicing():
    """Test slicing when non-KV field precedes KV field."""

    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["x", "y", "x"],
        }
    )

    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), ["num"]),
            ("cat", OneHotEncoder(sparse_output=False), ["cat"]),
        ]
    )
    ct.fit(df)

    expr = xo.memtable(df)
    structer = get_structer_out(ct, expr)
    convert_fn = structer.get_convert_struct_with_kv()
    result = convert_fn(ct, df)

    assert len(result) == 3
    for row in result:
        assert isinstance(row["num__num"], float)
        assert isinstance(row["cat"], tuple)
        assert len(row["cat"]) == 2  # x and y categories

    # Verify num values (first column in sklearn output)
    sklearn_result = ct.transform(df)
    expected_num_values = sklearn_result[:, 0].tolist()
    actual_num_values = [row["num__num"] for row in result]
    assert actual_num_values == pytest.approx(expected_num_values)


def test_convert_struct_with_kv_multiple_kv_fields_slicing():
    """Test slicing with multiple KV fields."""

    df = pd.DataFrame(
        {
            "cat1": ["a", "b", "c"],
            "num": [1.0, 2.0, 3.0],
            "cat2": ["x", "y", "x"],
        }
    )

    ct = ColumnTransformer(
        [
            ("cat1", OneHotEncoder(sparse_output=False), ["cat1"]),
            ("num", StandardScaler(), ["num"]),
            ("cat2", OneHotEncoder(sparse_output=False), ["cat2"]),
        ]
    )
    ct.fit(df)

    expr = xo.memtable(df)
    structer = get_structer_out(ct, expr)
    convert_fn = structer.get_convert_struct_with_kv()
    result = convert_fn(ct, df)

    assert len(result) == 3
    for row in result:
        assert isinstance(row["cat1"], tuple)
        assert len(row["cat1"]) == 3  # a, b, c
        assert isinstance(row["num__num"], float)
        assert isinstance(row["cat2"], tuple)
        assert len(row["cat2"]) == 2  # x, y

    # sklearn output: cat1 (3 cols) + num (1 col) + cat2 (2 cols) = 6 cols
    sklearn_result = ct.transform(df)
    assert sklearn_result.shape[1] == 6

    # Verify num values (column index 3)
    expected_num_values = sklearn_result[:, 3].tolist()
    actual_num_values = [row["num__num"] for row in result]
    assert actual_num_values == pytest.approx(expected_num_values)


@pytest.fixture
def classification_data():
    """Generate classification data with features and target."""

    np.random.seed(42)
    n_samples = 100
    # Create features where some have clear relationship with target
    X = pd.DataFrame(
        {
            "f1": np.random.randn(n_samples),
            "f2": np.random.randn(n_samples),
            "f3": np.random.randn(n_samples),
            "f4": np.random.randn(n_samples),
        }
    )
    # Target correlated with f1 and f2
    y = ((X["f1"] + X["f2"]) > 0).astype(int)
    X["target"] = y
    return X


@pytest.mark.parametrize(
    "selector_cls,selector_kwargs",
    [
        pytest.param(
            "SelectKBest",
            {"k": 2},
            id="SelectKBest",
        ),
        pytest.param(
            "SelectPercentile",
            {"percentile": 50},
            id="SelectPercentile",
        ),
        pytest.param(
            "SelectFpr",
            {"alpha": 0.05},
            id="SelectFpr",
        ),
        pytest.param(
            "SelectFdr",
            {"alpha": 0.05},
            id="SelectFdr",
        ),
        pytest.param(
            "SelectFwe",
            {"alpha": 0.05},
            id="SelectFwe",
        ),
        pytest.param(
            "GenericUnivariateSelect",
            {"mode": "k_best", "param": 2},
            id="GenericUnivariateSelect",
        ),
    ],
)
def test_feature_selector_parity_univariate_selector_matches_sklearn(
    classification_data, selector_cls, selector_kwargs
):
    """Test univariate feature selectors match sklearn output."""

    data = classification_data
    features = ("f1", "f2", "f3", "f4")
    t = xo.memtable(data)

    # Get the selector class
    SelectorClass = getattr(feature_selection, selector_cls)
    selector = SelectorClass(**selector_kwargs)

    # xorq result
    step = xo.Step.from_instance_name(selector, name="selector")
    fitted = step.fit(t, features=features, target="target")
    result = fitted.transform(t, retain_others=False)
    xorq_raw = result.execute()

    # Handle both KV-encoded and struct output
    if "transformed" in xorq_raw.columns:
        # KV-encoded output - decode it
        xorq_df = KVEncoder.decode(xorq_raw["transformed"])
    else:
        # Struct output - columns are already unpacked (transformed_0, etc.)
        xorq_df = xorq_raw

    # sklearn result
    sklearn_selector = SelectorClass(**selector_kwargs)
    sklearn_result = sklearn_selector.fit_transform(
        data[list(features)], data["target"]
    )
    sklearn_df = pd.DataFrame(
        sklearn_result, columns=sklearn_selector.get_feature_names_out()
    )

    # Compare values (struct output has stub names, so compare values only)

    np.testing.assert_allclose(
        xorq_df.reset_index(drop=True).values,
        sklearn_df.reset_index(drop=True).values,
        atol=1e-10,
    )


def test_feature_selector_parity_variance_threshold_matches_sklearn(
    classification_data,
):
    """Test VarianceThreshold (unsupervised selector) matches sklearn output.

    VarianceThreshold is unique among feature selectors - it doesn't require
    a target variable, making it an unsupervised feature selector.
    """

    data = classification_data
    features = ("f1", "f2", "f3", "f4")
    t = xo.memtable(data)

    # Use a low threshold so we keep most features
    selector = VarianceThreshold(threshold=0.1)

    # xorq result
    step = xo.Step.from_instance_name(selector, name="selector")
    fitted = step.fit(t, features=features)  # No target needed
    result = fitted.transform(t, retain_others=False)
    xorq_raw = result.execute()

    # Handle both KV-encoded and struct output
    if "transformed" in xorq_raw.columns:
        xorq_df = KVEncoder.decode(xorq_raw["transformed"])
    else:
        xorq_df = xorq_raw

    # sklearn result
    sklearn_selector = VarianceThreshold(threshold=0.1)
    sklearn_result = sklearn_selector.fit_transform(data[list(features)])
    sklearn_df = pd.DataFrame(
        sklearn_result, columns=sklearn_selector.get_feature_names_out()
    )

    # Compare values
    np.testing.assert_allclose(
        xorq_df.reset_index(drop=True).values,
        sklearn_df.reset_index(drop=True).values,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "selector_cls,selector_kwargs",
    [
        pytest.param(
            "RFE",
            {"n_features_to_select": 2},
            id="RFE",
        ),
        pytest.param(
            "RFECV",
            {"min_features_to_select": 2, "cv": 3},
            id="RFECV",
        ),
        pytest.param(
            "SelectFromModel",
            {"threshold": "median"},
            id="SelectFromModel",
        ),
        pytest.param(
            "SequentialFeatureSelector",
            {"n_features_to_select": 2, "cv": 3},
            id="SequentialFeatureSelector",
        ),
    ],
)
def test_feature_selector_parity_model_based_selector_matches_sklearn(
    classification_data, selector_cls, selector_kwargs
):
    """Test model-based feature selectors match sklearn output."""

    data = classification_data
    features = ("f1", "f2", "f3", "f4")
    t = xo.memtable(data)

    # Get the selector class
    SelectorClass = getattr(feature_selection, selector_cls)
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    selector = SelectorClass(estimator=estimator, **selector_kwargs)

    # xorq result
    step = xo.Step.from_instance_name(selector, name="selector")
    fitted = step.fit(t, features=features, target="target")
    result = fitted.transform(t, retain_others=False)
    xorq_raw = result.execute()

    # Handle both KV-encoded and struct output
    if "transformed" in xorq_raw.columns:
        # KV-encoded output - decode it
        xorq_df = KVEncoder.decode(xorq_raw["transformed"])
    else:
        # Struct output - columns are already unpacked (transformed_0, etc.)
        xorq_df = xorq_raw

    # sklearn result
    sklearn_estimator = LogisticRegression(max_iter=1000, random_state=42)
    sklearn_selector = SelectorClass(estimator=sklearn_estimator, **selector_kwargs)
    sklearn_result = sklearn_selector.fit_transform(
        data[list(features)], data["target"]
    )
    sklearn_df = pd.DataFrame(
        sklearn_result, columns=sklearn_selector.get_feature_names_out()
    )

    # Compare values (struct output has stub names, so compare values only)

    np.testing.assert_allclose(
        xorq_df.reset_index(drop=True).values,
        sklearn_df.reset_index(drop=True).values,
        atol=1e-10,
    )


@pytest.fixture
def numeric_data():
    """Generate numeric data for transformers."""

    np.random.seed(42)
    n_samples = 100
    # Create positive features (required for NMF, LDA topic modeling)
    X = pd.DataFrame(
        {
            "f1": np.abs(np.random.randn(n_samples)) + 0.1,
            "f2": np.abs(np.random.randn(n_samples)) + 0.1,
            "f3": np.abs(np.random.randn(n_samples)) + 0.1,
            "f4": np.abs(np.random.randn(n_samples)) + 0.1,
        }
    )
    # Add target for supervised methods (LDA, NCA)
    X["target"] = np.random.randint(0, 3, n_samples)
    return X


@pytest.mark.parametrize(
    "transformer_cls,transformer_kwargs,module,needs_target,nondeterministic",
    [
        # Decomposition - Unsupervised
        pytest.param(
            "PCA",
            {"n_components": 2, "random_state": 42},
            "sklearn.decomposition",
            False,
            False,
            id="PCA",
        ),
        pytest.param(
            "IncrementalPCA",
            {"n_components": 2},
            "sklearn.decomposition",
            False,
            False,
            id="IncrementalPCA",
        ),
        pytest.param(
            "KernelPCA",
            {"n_components": 2, "kernel": "rbf", "random_state": 42},
            "sklearn.decomposition",
            False,
            False,
            id="KernelPCA",
        ),
        pytest.param(
            "FastICA",
            {"n_components": 2, "random_state": 42},
            "sklearn.decomposition",
            False,
            False,
            id="FastICA",
        ),
        pytest.param(
            "TruncatedSVD",
            {"n_components": 2, "random_state": 42},
            "sklearn.decomposition",
            False,
            False,
            id="TruncatedSVD",
        ),
        pytest.param(
            "NMF",
            {"n_components": 2, "random_state": 42, "init": "random"},
            "sklearn.decomposition",
            False,
            False,
            id="NMF",
        ),
        pytest.param(
            "MiniBatchNMF",
            {"n_components": 2, "random_state": 42},
            "sklearn.decomposition",
            False,
            True,
            id="MiniBatchNMF",
        ),
        pytest.param(
            "FactorAnalysis",
            {"n_components": 2, "random_state": 42},
            "sklearn.decomposition",
            False,
            False,
            id="FactorAnalysis",
        ),
        pytest.param(
            "LatentDirichletAllocation",
            {"n_components": 2, "random_state": 42},
            "sklearn.decomposition",
            False,
            False,
            id="LatentDirichletAllocation",
        ),
        pytest.param(
            "DictionaryLearning",
            {"n_components": 2, "random_state": 42, "max_iter": 5},
            "sklearn.decomposition",
            False,
            True,
            id="DictionaryLearning",
        ),
        pytest.param(
            "MiniBatchDictionaryLearning",
            {"n_components": 2, "random_state": 42, "max_iter": 5},
            "sklearn.decomposition",
            False,
            False,
            id="MiniBatchDictionaryLearning",
        ),
        pytest.param(
            "MiniBatchSparsePCA",
            {"n_components": 2, "random_state": 42, "max_iter": 5},
            "sklearn.decomposition",
            False,
            False,
            id="MiniBatchSparsePCA",
        ),
        pytest.param(
            "SparsePCA",
            {"n_components": 2, "random_state": 42, "max_iter": 5},
            "sklearn.decomposition",
            False,
            False,
            id="SparsePCA",
        ),
        # Manifold Learning - Unsupervised
        pytest.param(
            "Isomap",
            {"n_components": 2},
            "sklearn.manifold",
            False,
            False,
            id="Isomap",
        ),
        pytest.param(
            "LocallyLinearEmbedding",
            {"n_components": 2, "random_state": 42, "n_neighbors": 5},
            "sklearn.manifold",
            False,
            False,
            id="LocallyLinearEmbedding",
        ),
        # Discriminant Analysis - Supervised
        pytest.param(
            "LinearDiscriminantAnalysis",
            {"n_components": 2},
            "sklearn.discriminant_analysis",
            True,
            False,
            id="LinearDiscriminantAnalysis",
        ),
        # Neighbors - Supervised
        pytest.param(
            "NeighborhoodComponentsAnalysis",
            {"n_components": 2, "random_state": 42},
            "sklearn.neighbors",
            True,
            False,
            id="NeighborhoodComponentsAnalysis",
        ),
    ],
)
def test_classname_prefix_features_out_mixin_parity_classname_prefix_transformer_matches_sklearn(
    numeric_data,
    transformer_cls,
    transformer_kwargs,
    module,
    needs_target,
    nondeterministic,
):
    """Test ClassNamePrefixFeaturesOutMixin transformers match sklearn output."""

    data = numeric_data
    features = ("f1", "f2", "f3", "f4")
    t = xo.memtable(data)

    # Get the transformer class
    mod = importlib.import_module(module)
    TransformerClass = getattr(mod, transformer_cls)
    transformer = TransformerClass(**transformer_kwargs)

    # xorq result
    step = xo.Step.from_instance_name(transformer, name="reducer")
    if needs_target:
        fitted = step.fit(t, features=features, target="target")
    else:
        fitted = step.fit(t, features=features)
    result = fitted.transform(t, retain_others=False)
    xorq_raw = result.execute()

    # Handle both KV-encoded and struct output
    if "transformed" in xorq_raw.columns:
        # KV-encoded output - decode it
        xorq_df = KVEncoder.decode(xorq_raw["transformed"])
    else:
        # Struct output - columns are already unpacked
        xorq_df = xorq_raw

    # sklearn result
    sklearn_transformer = TransformerClass(**transformer_kwargs)
    X = data[list(features)]
    if needs_target:
        sklearn_transformer.fit(X, data["target"])
        sklearn_result = sklearn_transformer.transform(X)
    else:
        sklearn_result = sklearn_transformer.fit_transform(X)

    sklearn_df = pd.DataFrame(
        sklearn_result, columns=sklearn_transformer.get_feature_names_out()
    )

    # Also verify column names match sklearn's get_feature_names_out()
    assert list(xorq_df.columns) == list(sklearn_df.columns)

    if nondeterministic:
        # For non-deterministic estimators (MiniBatchNMF, DictionaryLearning),
        # independent fits can diverge. Verify shape and finiteness instead.
        assert xorq_df.shape == sklearn_df.shape
        assert np.all(np.isfinite(xorq_df.reset_index(drop=True).values))
    else:
        np.testing.assert_allclose(
            xorq_df.reset_index(drop=True).values,
            sklearn_df.reset_index(drop=True).values,
            atol=1e-2,
        )


@pytest.fixture
def test_data():
    """Create test data with all required column types."""

    np.random.seed(42)
    n_samples = 20

    # Categorical data for encoders (simple values to avoid double prefixes)
    categories = ["a", "b", "c", "d"]
    cat_col = np.random.choice(categories, n_samples)

    # Numeric data for feature expansion and kernel methods
    num1 = np.random.randn(n_samples) * 2 + 5
    num2 = np.random.randn(n_samples) * 3 + 10

    # Positive numeric data for chi2 samplers
    pos1 = np.abs(np.random.randn(n_samples)) + 0.1
    pos2 = np.abs(np.random.randn(n_samples)) + 0.1

    # Text data for vectorizers
    words = ["hello", "world", "foo", "bar", "baz", "qux"]
    text_col = [
        " ".join(np.random.choice(words, size=np.random.randint(2, 5)))
        for _ in range(n_samples)
    ]

    # Binary target for supervised transformers
    target = np.random.randint(0, 2, n_samples)

    return pd.DataFrame(
        {
            "cat": cat_col,
            "num1": num1,
            "num2": num2,
            "pos1": pos1,
            "pos2": pos2,
            "text": text_col,
            "target": target,
        }
    )


@pytest.mark.parametrize(
    "transformer_cls,transformer_kwargs,module,features,target_col,input_type",
    [
        # Encoders
        pytest.param(
            "OneHotEncoder",
            {"sparse_output": False},
            "sklearn.preprocessing",
            ("cat",),
            None,
            "categorical",
            id="OneHotEncoder",
        ),
        pytest.param(
            "OrdinalEncoder",
            {},
            "sklearn.preprocessing",
            ("cat",),
            None,
            "categorical",
            id="OrdinalEncoder",
        ),
        pytest.param(
            "TargetEncoder",
            {"random_state": 42},
            "sklearn.preprocessing",
            ("cat",),
            "target",
            "categorical",
            id="TargetEncoder",
        ),
        # Text Vectorizers
        pytest.param(
            "TfidfVectorizer",
            {},
            "sklearn.feature_extraction.text",
            ("text",),
            None,
            "text",
            id="TfidfVectorizer",
        ),
        pytest.param(
            "CountVectorizer",
            {},
            "sklearn.feature_extraction.text",
            ("text",),
            None,
            "text",
            id="CountVectorizer",
        ),
        # Feature Expansion
        pytest.param(
            "PolynomialFeatures",
            {"degree": 2, "include_bias": False},
            "sklearn.preprocessing",
            ("num1", "num2"),
            None,
            "numeric",
            id="PolynomialFeatures",
        ),
        pytest.param(
            "SplineTransformer",
            {"n_knots": 4},
            "sklearn.preprocessing",
            ("num1",),
            None,
            "numeric",
            id="SplineTransformer",
        ),
        pytest.param(
            "KBinsDiscretizer",
            {"n_bins": 3, "encode": "onehot", "strategy": "uniform"},
            "sklearn.preprocessing",
            ("num1",),
            None,
            "numeric",
            id="KBinsDiscretizer",
        ),
        # Kernel Approximation
        pytest.param(
            "AdditiveChi2Sampler",
            {"sample_steps": 2},
            "sklearn.kernel_approximation",
            ("pos1", "pos2"),
            None,
            "positive",
            id="AdditiveChi2Sampler",
        ),
        pytest.param(
            "RBFSampler",
            {"n_components": 5, "random_state": 42},
            "sklearn.kernel_approximation",
            ("num1", "num2"),
            None,
            "numeric",
            id="RBFSampler",
        ),
        pytest.param(
            "Nystroem",
            {"n_components": 5, "random_state": 42},
            "sklearn.kernel_approximation",
            ("num1", "num2"),
            None,
            "numeric",
            id="Nystroem",
        ),
        pytest.param(
            "SkewedChi2Sampler",
            {"n_components": 5, "random_state": 42},
            "sklearn.kernel_approximation",
            ("pos1", "pos2"),
            None,
            "positive",
            id="SkewedChi2Sampler",
        ),
        pytest.param(
            "PolynomialCountSketch",
            {"n_components": 5, "random_state": 42},
            "sklearn.kernel_approximation",
            ("num1", "num2"),
            None,
            "numeric",
            id="PolynomialCountSketch",
        ),
        # Neighbor Transformers (output depends on n_samples)
        pytest.param(
            "KNeighborsTransformer",
            {"n_neighbors": 3, "mode": "distance"},
            "sklearn.neighbors",
            ("num1", "num2"),
            None,
            "numeric",
            id="KNeighborsTransformer",
        ),
        pytest.param(
            "RadiusNeighborsTransformer",
            {"radius": 5.0, "mode": "distance"},
            "sklearn.neighbors",
            ("num1", "num2"),
            None,
            "numeric",
            id="RadiusNeighborsTransformer",
        ),
        # Tree-based transformer (output depends on leaf nodes)
        pytest.param(
            "RandomTreesEmbedding",
            {"n_estimators": 5, "max_depth": 2, "random_state": 42},
            "sklearn.ensemble",
            ("num1", "num2"),
            None,
            "numeric",
            id="RandomTreesEmbedding",
        ),
        # Clustering (output depends on subcluster count, not n_clusters)
        pytest.param(
            "Birch",
            {"n_clusters": 3},
            "sklearn.cluster",
            ("num1", "num2"),
            None,
            "numeric",
            id="Birch",
        ),
    ],
)
def test_kv_encoded_transformers_parity_kv_encoded_transformer_matches_sklearn(
    test_data,
    transformer_cls,
    transformer_kwargs,
    module,
    features,
    target_col,
    input_type,
):
    """Test KV-encoded transformers match sklearn output."""

    data = test_data
    t = xo.memtable(data)

    # Get the transformer class
    mod = importlib.import_module(module)
    TransformerClass = getattr(mod, transformer_cls)
    transformer = TransformerClass(**transformer_kwargs)

    # xorq result
    step = xo.Step.from_instance_name(transformer, name="transformer")

    if target_col:
        fitted = step.fit(t, features=features, target=target_col)
    else:
        fitted = step.fit(t, features=features)

    result = fitted.transform(t, retain_others=False)
    xorq_raw = result.execute()

    # Handle both KV-encoded and struct output
    if "transformed" in xorq_raw.columns:
        # KV-encoded output - decode it
        xorq_df = KVEncoder.decode(xorq_raw["transformed"])
    else:
        # Struct output - columns are already unpacked
        xorq_df = xorq_raw

    # sklearn result - prepare input based on type
    # Note: Use fit then transform (not fit_transform) to match xorq behavior
    # Some transformers like TargetEncoder have special fit_transform behavior
    if input_type == "text":
        # Text vectorizers expect a 1D array of strings
        sklearn_input = data[features[0]].tolist()
        sklearn_y = data[target_col].values if target_col else None
        sklearn_transformer = TransformerClass(**transformer_kwargs)
        if sklearn_y is not None:
            sklearn_transformer.fit(sklearn_input, sklearn_y)
        else:
            sklearn_transformer.fit(sklearn_input)
        sklearn_result = sklearn_transformer.transform(sklearn_input)
    else:
        # Pass DataFrame to sklearn so it uses same feature names as xorq
        sklearn_input = data[list(features)]
        sklearn_y = data[target_col].values if target_col else None
        sklearn_transformer = TransformerClass(**transformer_kwargs)
        if sklearn_y is not None:
            sklearn_transformer.fit(sklearn_input, sklearn_y)
        else:
            sklearn_transformer.fit(sklearn_input)
        sklearn_result = sklearn_transformer.transform(sklearn_input)

    # Convert sparse to dense if needed
    if hasattr(sklearn_result, "toarray"):
        sklearn_result = sklearn_result.toarray()

    sklearn_df = pd.DataFrame(
        sklearn_result, columns=sklearn_transformer.get_feature_names_out()
    )

    # Compare values
    xorq_sorted = xorq_df[sorted(xorq_df.columns)].reset_index(drop=True)
    sklearn_sorted = sklearn_df[sorted(sklearn_df.columns)].reset_index(drop=True)

    np.testing.assert_allclose(
        xorq_sorted.values,
        sklearn_sorted.values,
        atol=1e-6,
        rtol=1e-6,
    )

    # Verify column names match sklearn's get_feature_names_out()
    assert sorted(xorq_df.columns) == sorted(sklearn_df.columns)
