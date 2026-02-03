import pandas as pd
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.expr.ml.structer import (
    KV_ENCODED_TYPE,
    KVEncoder,
    KVField,
    Structer,
    _accumulate_pipeline_step,
    _get_remainder_fields,
    _merge_ct_results,
    _normalize_columns,
    _prefix_fields,
    _process_child_structer,
    _process_ct_item,
    _process_fu_item,
)


sklearn = pytest.importorskip("sklearn")


class TestKVField:
    """Tests for KVField StrEnum behavior and string compatibility."""

    def test_kvfield_is_strenum(self):
        """Test KVField inherits from StrEnum."""
        try:
            from enum import StrEnum
        except ImportError:
            from strenum import StrEnum
        assert issubclass(KVField, StrEnum)

    def test_kvfield_members_are_strings(self):
        """Test KVField members are string instances."""
        assert isinstance(KVField.KEY, str)
        assert isinstance(KVField.VALUE, str)

    def test_kvfield_string_equality(self):
        """Test KVField members equal their string values."""
        assert KVField.KEY == "key"
        assert KVField.VALUE == "value"

    def test_kvfield_membership_by_member(self):
        """Test membership check works with enum members."""
        assert KVField.KEY in KVField
        assert KVField.VALUE in KVField

    def test_kvfield_callable_lookup(self):
        """Test KVField can be called with string value to get member."""
        assert KVField("key") is KVField.KEY
        assert KVField("value") is KVField.VALUE
        with pytest.raises(ValueError):
            KVField("nonexistent")

    def test_kvfield_value_attribute(self):
        """Test .value attribute returns the string value."""
        assert KVField.KEY.value == "key"
        assert KVField.VALUE.value == "value"

    def test_kvfield_name_attribute(self):
        """Test .name attribute returns the member name."""
        assert KVField.KEY.name == "KEY"
        assert KVField.VALUE.name == "VALUE"

    def test_kvfield_iteration(self):
        """Test iterating over KVField yields all members."""
        members = list(KVField)
        assert len(members) == 2
        assert KVField.KEY in members
        assert KVField.VALUE in members

    def test_kvfield_as_dict_key(self):
        """Test KVField members work as dictionary keys."""
        d = {KVField.KEY: "key_value", KVField.VALUE: "value_value"}
        assert d["key"] == "key_value"
        assert d["value"] == "value_value"
        assert d[KVField.KEY] == "key_value"
        assert d[KVField.VALUE] == "value_value"

    def test_kvfield_string_operations(self):
        """Test KVField members support string operations."""
        assert KVField.KEY.upper() == "KEY"
        assert KVField.VALUE.capitalize() == "Value"
        assert KVField.KEY + "_suffix" == "key_suffix"

    def test_kvfield_hash_matches_string(self):
        """Test KVField members hash the same as their string values."""
        assert hash(KVField.KEY) == hash("key")
        assert hash(KVField.VALUE) == hash("value")

    def test_kv_encoded_type_uses_kvfield_values(self):
        """Test KV_ENCODED_TYPE struct uses KVField.value strings as keys."""
        struct_fields = KV_ENCODED_TYPE.value_type.fields
        # Keys are plain strings (for YAML serialization compatibility)
        assert "key" in struct_fields
        assert "value" in struct_fields
        # KVField members also work due to string equality
        assert KVField.KEY in struct_fields
        assert KVField.VALUE in struct_fields


class TestKVEncoder:
    """Tests for KVEncoder encode/decode functionality."""

    def test_encode_basic(self):
        """Test KVEncoder.encode with a simple transformer."""
        from sklearn.preprocessing import OneHotEncoder

        df = pd.DataFrame({"cat": ["a", "b", "a"]})
        model = OneHotEncoder(sparse_output=False)
        model.fit(df)

        result = KVEncoder.encode(model, df)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        # Each element should be a tuple of dicts with 'key' and 'value'
        first_row = result.iloc[0]
        assert all("key" in d and "value" in d for d in first_row)

    def test_encode_feature_names(self):
        """Test that encode uses get_feature_names_out() for keys."""
        from sklearn.preprocessing import OneHotEncoder

        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        model = OneHotEncoder(sparse_output=False)
        model.fit(df)

        result = KVEncoder.encode(model, df)
        keys = [d["key"] for d in result.iloc[0]]

        expected_names = list(model.get_feature_names_out())
        assert keys == expected_names

    def test_decode_basic(self):
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

    def test_decode_empty_series_raises(self):
        """Test decode raises ValueError for empty series."""
        series = pd.Series([], dtype=object)
        with pytest.raises(ValueError):
            KVEncoder.decode(series)

    def test_is_kv_encoded_type_true(self):
        """Test is_kv_encoded_type returns True for KV format."""
        assert KVEncoder.is_kv_encoded_type(KV_ENCODED_TYPE)

    def test_is_kv_encoded_type_false(self):
        """Test is_kv_encoded_type returns False for non-KV types."""
        assert not KVEncoder.is_kv_encoded_type(dt.float64)
        assert not KVEncoder.is_kv_encoded_type(dt.Array(dt.float64))
        assert not KVEncoder.is_kv_encoded_type(dt.Struct({"a": dt.float64}))


class TestStructer:
    """Tests for Structer class."""

    def test_from_names_typ(self):
        """Test creating Structer with known schema."""
        structer = Structer.from_names_typ(("a", "b"), dt.float64)

        assert not structer.is_kv_encoded
        assert structer.struct is not None
        assert "a" in structer.struct.fields
        assert "b" in structer.struct.fields

    def test_kv_encoded_factory(self):
        """Test creating KV-encoded Structer."""
        structer = Structer.kv_encoded(input_columns=("x", "y"))

        assert structer.is_kv_encoded
        assert structer.struct is None
        assert structer.input_columns == ("x", "y")
        assert structer.return_type == KV_ENCODED_TYPE

    def test_needs_target_default_false(self):
        """Test needs_target defaults to False."""
        structer = Structer.from_names_typ(("a",), dt.float64)
        assert structer.needs_target is False

    def test_needs_target_explicit(self):
        """Test needs_target can be set explicitly."""
        base = Structer.from_names_typ(("a",), dt.float64)
        structer = Structer(struct=base.struct, needs_target=True)
        assert structer.needs_target is True

    def test_is_series_default_false(self):
        """Test is_series defaults to False."""
        structer = Structer.from_names_typ(("a",), dt.float64)
        assert structer.is_series is False

    def test_is_series_explicit(self):
        """Test is_series can be set explicitly."""
        structer = Structer(struct=None, input_columns=("text",), is_series=True)
        assert structer.is_series is True

    def test_return_type_struct(self):
        """Test return_type for known schema."""
        structer = Structer.from_names_typ(("a", "b"), dt.float64)
        assert isinstance(structer.return_type, dt.Struct)

    def test_return_type_kv_encoded(self):
        """Test return_type for KV-encoded."""
        structer = Structer.kv_encoded(input_columns=("x",))
        assert structer.return_type == KV_ENCODED_TYPE

    def test_output_columns_struct(self):
        """Test output_columns for known schema."""
        structer = Structer.from_names_typ(("a", "b"), dt.float64)
        assert structer.output_columns == ("a", "b")

    def test_output_columns_kv_encoded_raises(self):
        """Test output_columns raises for KV-encoded."""
        structer = Structer.kv_encoded(input_columns=("x",))
        with pytest.raises(ValueError, match="KV-encoded"):
            _ = structer.output_columns

    def test_get_convert_array_kv_encoded_raises(self):
        """Test get_convert_array raises for KV-encoded."""
        structer = Structer.kv_encoded(input_columns=("x",))
        with pytest.raises(ValueError, match="KV-encoded"):
            structer.get_convert_array()

    def test_get_output_columns_known_schema(self):
        """Test get_output_columns returns dtype keys for known schema."""
        structer = Structer.from_names_typ(("a", "b"), dt.float64)
        assert structer.get_output_columns() == ("a", "b")
        # dest_col is ignored for known schema
        assert structer.get_output_columns(dest_col="foo") == ("a", "b")

    def test_get_output_columns_kv_encoded(self):
        """Test get_output_columns returns dest_col tuple for KV-encoded."""
        structer = Structer.kv_encoded(input_columns=("x",))
        assert structer.get_output_columns() == ("transformed",)
        assert structer.get_output_columns(dest_col="encoded") == ("encoded",)

    def test_maybe_unpack_known_schema(self):
        """Test maybe_unpack unpacks struct column for known schema."""
        from sklearn.preprocessing import StandardScaler

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

    def test_maybe_unpack_kv_encoded(self):
        """Test maybe_unpack returns expr unchanged for KV-encoded."""
        from sklearn.preprocessing import OneHotEncoder

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


class TestStructerFromInstance:
    """Tests for structer_from_instance dispatch."""

    def test_standard_scaler(self):
        """Test StandardScaler produces known schema."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(StandardScaler(), t, features=("a", "b"))

        assert not structer.is_kv_encoded
        assert not structer.needs_target
        assert not structer.is_series

    def test_simple_imputer(self):
        """Test SimpleImputer produces known schema."""
        from sklearn.impute import SimpleImputer

        t = xo.memtable({"a": [1.0, None], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(SimpleImputer(), t, features=("a", "b"))

        assert not structer.is_kv_encoded
        assert not structer.needs_target

    def test_missing_indicator_features_all(self):
        """Test MissingIndicator with features='all' produces known boolean schema."""
        from sklearn.impute import MissingIndicator

        t = xo.memtable({"a": [1.0, None], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(
            MissingIndicator(features="all"), t, features=("a", "b")
        )

        assert not structer.is_kv_encoded
        assert structer.struct == dt.Struct({"a": dt.boolean, "b": dt.boolean})
        assert not structer.needs_target

    def test_missing_indicator_features_missing_only(self):
        """Test MissingIndicator with features='missing-only' produces KV-encoded schema."""
        from sklearn.impute import MissingIndicator

        t = xo.memtable({"a": [1.0, None], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(
            MissingIndicator(features="missing-only"), t, features=("a", "b")
        )

        assert structer.is_kv_encoded
        assert structer.input_columns == ("a", "b")
        assert not structer.needs_target

    def test_one_hot_encoder(self):
        """Test OneHotEncoder produces KV-encoded schema."""
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b", "c"]})
        structer = Structer.from_instance_expr(OneHotEncoder(), t, features=("cat",))

        assert structer.is_kv_encoded
        assert not structer.needs_target
        assert not structer.is_series

    def test_select_k_best(self):
        """Test SelectKBest produces stub columns when k is known."""
        from sklearn.feature_selection import SelectKBest

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(SelectKBest(k=1), t, features=("a", "b"))

        # Known k -> stub column names (not KV-encoded)
        assert not structer.is_kv_encoded
        assert structer.needs_target is True
        assert structer.input_columns == ("a", "b")
        assert tuple(structer.struct.names) == ("transformed_0",)

    def test_tfidf_vectorizer(self):
        """Test TfidfVectorizer is KV-encoded and is_series."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        t = xo.memtable({"text": ["hello world", "foo bar"]})
        structer = Structer.from_instance_expr(TfidfVectorizer(), t, features=("text",))

        assert structer.is_kv_encoded
        assert structer.is_series
        assert not structer.needs_target

    def test_unregistered_raises(self):
        """Test unregistered transformer raises ValueError."""

        class CustomTransformer:
            pass

        t = xo.memtable({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="can't handle type"):
            Structer.from_instance_expr(CustomTransformer(), t)


class TestKVEncoderIntegration:
    """Integration tests for KV encoding with xorq."""

    def test_one_hot_encoder_step(self):
        """Test OneHotEncoder through Step interface."""
        from sklearn.preprocessing import OneHotEncoder

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

    def test_tfidf_vectorizer_step(self):
        """Test TfidfVectorizer through Step interface."""
        from sklearn.feature_extraction.text import TfidfVectorizer

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

    def test_tfidf_matches_sklearn(self):
        """Test TfidfVectorizer output matches sklearn."""
        from sklearn.feature_extraction.text import TfidfVectorizer

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

    def test_select_k_best_with_target(self):
        """Test SelectKBest works with target column."""
        from sklearn.feature_selection import SelectKBest, f_classif

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

    def test_get_kv_value_type_non_kv(self):
        """Test get_kv_value_type returns original type for non-KV types."""
        typ = dt.float64
        assert KVEncoder.get_kv_value_type(typ) == dt.float64

    def test_structer_dtype_raises_for_kv_encoded(self):
        """Test dtype property raises for KV-encoded Structer."""
        structer = Structer.kv_encoded(input_columns=("x",))
        with pytest.raises(ValueError, match="KV-encoded"):
            _ = structer.dtype

    def test_select_k_best_kv_encoded_input(self):
        """Test SelectKBest with KV-encoded input uses KV-encoding output."""
        from sklearn.feature_selection import SelectKBest
        from sklearn.preprocessing import OneHotEncoder

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


class TestScalerTransformerParity:
    """Parity tests for scalers/transformers comparing xorq output with sklearn."""

    def test_minmax_scaler_matches_sklearn(self):
        """Test MinMaxScaler output matches sklearn."""
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler

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

    def test_maxabs_scaler_matches_sklearn(self):
        """Test MaxAbsScaler output matches sklearn."""
        import numpy as np
        from sklearn.preprocessing import MaxAbsScaler

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

    def test_robust_scaler_matches_sklearn(self):
        """Test RobustScaler output matches sklearn."""
        import numpy as np
        from sklearn.preprocessing import RobustScaler

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

    def test_normalizer_matches_sklearn(self):
        """Test Normalizer output matches sklearn."""
        import numpy as np
        from sklearn.preprocessing import Normalizer

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

    def test_power_transformer_matches_sklearn(self):
        """Test PowerTransformer output matches sklearn."""
        import numpy as np
        from sklearn.preprocessing import PowerTransformer

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

    def test_quantile_transformer_matches_sklearn(self):
        """Test QuantileTransformer output matches sklearn."""
        import numpy as np
        from sklearn.preprocessing import QuantileTransformer

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


class TestScalerTransformerStructer:
    """Tests for scaler/transformer structer_from_instance registrations."""

    def test_minmax_scaler_produces_known_schema(self):
        """Test MinMaxScaler produces known schema."""
        from sklearn.preprocessing import MinMaxScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(MinMaxScaler(), t, features=("a", "b"))

        assert not structer.is_kv_encoded
        assert not structer.needs_target
        assert not structer.is_series
        assert "a" in structer.struct.fields
        assert "b" in structer.struct.fields

    def test_maxabs_scaler_produces_known_schema(self):
        """Test MaxAbsScaler produces known schema."""
        from sklearn.preprocessing import MaxAbsScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(MaxAbsScaler(), t, features=("a", "b"))

        assert not structer.is_kv_encoded
        assert not structer.needs_target
        assert not structer.is_series

    def test_robust_scaler_produces_known_schema(self):
        """Test RobustScaler produces known schema."""
        from sklearn.preprocessing import RobustScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(RobustScaler(), t, features=("a", "b"))

        assert not structer.is_kv_encoded
        assert not structer.needs_target
        assert not structer.is_series

    def test_normalizer_produces_known_schema(self):
        """Test Normalizer produces known schema."""
        from sklearn.preprocessing import Normalizer

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(Normalizer(), t, features=("a", "b"))

        assert not structer.is_kv_encoded
        assert not structer.needs_target
        assert not structer.is_series

    def test_power_transformer_produces_known_schema(self):
        """Test PowerTransformer produces known schema."""
        from sklearn.preprocessing import PowerTransformer

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(
            PowerTransformer(), t, features=("a", "b")
        )

        assert not structer.is_kv_encoded
        assert not structer.needs_target
        assert not structer.is_series

    def test_quantile_transformer_produces_known_schema(self):
        """Test QuantileTransformer produces known schema."""
        from sklearn.preprocessing import QuantileTransformer

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        structer = Structer.from_instance_expr(
            QuantileTransformer(), t, features=("a", "b")
        )

        assert not structer.is_kv_encoded
        assert not structer.needs_target
        assert not structer.is_series


class TestOneToOneFeatureMixinParameterized:
    """Parameterized tests for OneToOneFeatureMixin estimators (scalers/transformers).

    These estimators maintain a 1:1 correspondence between input and output features.
    This test class consolidates coverage for all registered OneToOneFeatureMixin
    transformers with both structer and parity tests.
    """

    @pytest.fixture
    def numeric_data(self):
        """Generate numeric data for scalers."""
        import numpy as np

        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "a": np.random.randn(n_samples) * 100,
                "b": np.random.randn(n_samples) * 50 + 25,
            }
        )

    @pytest.fixture
    def positive_data(self):
        """Generate positive numeric data for PowerTransformer box-cox."""
        import numpy as np

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
    def test_structer_and_parity(
        self,
        numeric_data,
        positive_data,
        transformer_cls,
        transformer_kwargs,
        requires_positive,
    ):
        """Test OneToOneFeatureMixin estimators: structer schema and sklearn parity."""
        from sklearn import preprocessing

        data = positive_data if requires_positive else numeric_data
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


class TestColumnTransformerStructer:
    """Tests for ColumnTransformer structer_from_instance registration."""

    def test_known_schema_all_transformers(self):
        """Test ColumnTransformer with all known-schema transformers."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

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

    def test_kv_encoded_transformer(self):
        """Test ColumnTransformer with KV-encoded transformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

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

    def test_mixed_known_and_kv_encoded(self):
        """Test ColumnTransformer with mixed known-schema and KV-encoded."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    def test_passthrough_explicit(self):
        """Test ColumnTransformer with explicit passthrough."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

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

    def test_remainder_passthrough(self):
        """Test ColumnTransformer with remainder='passthrough'."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

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

    def test_remainder_passthrough_with_kv_encoded(self):
        """Test ColumnTransformer with remainder='passthrough' and KV-encoded transformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b"], "num1": [1.0, 2.0], "num2": [3.0, 4.0]})
        ct = ColumnTransformer(
            [("encoder", OneHotEncoder(), ["cat"])],
            remainder="passthrough",
        )
        structer = Structer.from_instance_expr(ct, t)

        assert structer.struct_has_kv_fields
        assert structer.input_columns == ("cat",)

    def test_drop_transformer(self):
        """Test ColumnTransformer with drop transformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

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

    def test_nested_column_transformer(self):
        """Test nested ColumnTransformer (ColumnTransformer inside ColumnTransformer)."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

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

    def test_nested_column_transformer_with_kv_encoded(self):
        """Test nested ColumnTransformer where inner has KV-encoded."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    def test_unregistered_child_raises(self):
        """Test ColumnTransformer with unregistered child transformer raises."""
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.compose import ColumnTransformer

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

    def test_normalize_columns_list(self):
        """Test _normalize_columns handles list -> tuple."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

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

    def test_normalize_columns_string(self):
        """Test _normalize_columns handles string -> single-element tuple."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

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

    def test_normalize_columns_tuple(self):
        """Test _normalize_columns handles tuple -> tuple (passthrough)."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

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

    def test_normalize_columns_none(self):
        """Test _normalize_columns handles None -> empty tuple."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

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

    def test_normalize_columns_invalid_type_raises(self):
        """Test _normalize_columns raises TypeError for unsupported types."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), 123),  # invalid type
            ]
        )

        with pytest.raises(TypeError, match="Unsupported columns type"):
            Structer.from_instance_expr(ct, t)


class TestFeatureUnionStructer:
    """Tests for FeatureUnion structer_from_instance registration."""

    def test_known_schema_all_transformers(self):
        """Test FeatureUnion with all known-schema transformers."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

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

    def test_kv_encoded_transformer(self):
        """Test FeatureUnion with KV-encoded transformer."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import OneHotEncoder

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

    def test_mixed_known_and_kv_encoded(self):
        """Test FeatureUnion with mixed known-schema and KV-encoded."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    def test_unregistered_child_raises(self):
        """Test FeatureUnion with unregistered child transformer raises."""
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.pipeline import FeatureUnion

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

    def test_nested_feature_union(self):
        """Test nested FeatureUnion."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

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


class TestSklearnPipelineStructer:
    """Tests for sklearn Pipeline structer_from_instance registration."""

    def test_known_schema_pipeline(self):
        """Test Pipeline with all known-schema transformers."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

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

    def test_kv_encoded_pipeline(self):
        """Test Pipeline with KV-encoded transformer."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

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

    def test_kv_encoded_early_exit(self):
        """Test Pipeline returns KV-encoded once any step is KV-encoded."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

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

    def test_nested_pipeline(self):
        """Test nested Pipeline (Pipeline inside Pipeline)."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

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

    def test_pipeline_with_column_transformer(self):
        """Test Pipeline containing ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

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

    def test_pipeline_with_kv_column_transformer(self):
        """Test Pipeline containing ColumnTransformer with KV-encoded child."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

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

    def test_unregistered_step_raises(self):
        """Test Pipeline with unregistered transformer raises."""
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.pipeline import Pipeline

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

    def test_empty_pipeline_returns_input_schema(self):
        """Test empty Pipeline returns input schema (passthrough)."""
        from sklearn.pipeline import Pipeline

        t = xo.memtable({"num": [1.0, 2.0]})
        pipe = Pipeline([])

        # Empty pipeline is effectively passthrough
        structer = Structer.from_instance_expr(pipe, t, features=("num",))
        assert not structer.struct_has_kv_fields
        assert "num" in structer.struct.fields

    def test_passthrough_step(self):
        """Test Pipeline with 'passthrough' step."""
        from sklearn.pipeline import Pipeline

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

    def test_nested_pipeline_with_passthrough(self):
        """Test Pipeline(Pipeline()) where inner pipeline has passthrough columns."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

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

    def test_chained_feature_transforms(self):
        """Test Pipeline where features change between steps."""
        from sklearn.feature_selection import SelectKBest
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

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


class TestGetStructerOutErrorHandling:
    """Tests for get_structer_out error handling."""

    def test_unregistered_base_estimator_raises_value_error(self):
        """Test unregistered BaseEstimator subclass raises ValueError."""
        from sklearn.base import BaseEstimator

        from xorq.expr.ml.structer import get_structer_out

        class UnregisteredEstimator(BaseEstimator):
            pass

        t = xo.memtable({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="can't handle type"):
            get_structer_out(UnregisteredEstimator(), t)

    def test_non_base_estimator_raises_type_error(self):
        """Test non-BaseEstimator types raise TypeError."""
        from xorq.expr.ml.structer import get_structer_out

        class NotAnEstimator:
            pass

        t = xo.memtable({"a": [1.0, 2.0]})
        with pytest.raises(TypeError, match="Unexpected type in get_structer_out"):
            get_structer_out(NotAnEstimator(), t)

    def test_primitive_type_raises_type_error(self):
        """Test primitive types raise TypeError."""
        from xorq.expr.ml.structer import get_structer_out

        t = xo.memtable({"a": [1.0, 2.0]})
        with pytest.raises(TypeError, match="Unexpected type in get_structer_out"):
            get_structer_out("not an estimator", t)

    def test_none_raises_type_error(self):
        """Test None raises TypeError."""
        from xorq.expr.ml.structer import get_structer_out

        t = xo.memtable({"a": [1.0, 2.0]})
        with pytest.raises(TypeError, match="Unexpected type in get_structer_out"):
            get_structer_out(None, t)


class TestDecodeEncodedColumns:
    """Tests for KVEncoder.decode_encoded_column and decode_encoded_columns."""

    def test_no_encoded_cols(self):
        """Test passthrough when no encoded columns."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        features = ("a", "b")
        encoded_cols = ()

        result_df, result_features = KVEncoder.decode_encoded_columns(
            df, features, encoded_cols
        )

        pd.testing.assert_frame_equal(result_df, df)
        assert result_features == features

    def test_decode_encoded_column(self):
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

    def test_missing_encoded_col_raises(self):
        """Test that missing encoded columns raise ValueError."""
        df = pd.DataFrame({"a": [1.0, 2.0]})
        features = ("a",)

        with pytest.raises(ValueError, match="nonexistent not in DataFrame"):
            KVEncoder.decode_encoded_column(df, features, "nonexistent")

    def test_empty_encoded_col_raises(self):
        """Test that empty encoded column raises ValueError."""
        df = pd.DataFrame({"encoded": []})
        features = ("encoded",)

        with pytest.raises(ValueError, match="cannot decode empty column"):
            KVEncoder.decode_encoded_column(df, features, "encoded")


class TestGetSchemaOut:
    """Tests for get_schema_out function with sklearn-style prefixing and hybrid output."""

    def test_column_transformer_basic(self):
        """Test get_schema_out with ColumnTransformer uses sklearn-style prefixes."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

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

    def test_column_transformer_passthrough(self):
        """Test get_schema_out with ColumnTransformer passthrough uses prefixes."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

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

    def test_column_transformer_remainder_passthrough(self):
        """Test get_schema_out with ColumnTransformer remainder='passthrough'."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

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

    def test_column_transformer_drop(self):
        """Test get_schema_out with ColumnTransformer drop."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

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

    def test_feature_union(self):
        """Test get_schema_out with FeatureUnion uses prefixes."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

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

    def test_sklearn_pipeline(self):
        """Test get_schema_out with sklearn Pipeline (no prefixes for pipeline)."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

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

    def test_base_estimator(self):
        """Test get_schema_out with simple BaseEstimator (no prefix)."""
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        scaler = StandardScaler()
        schema = get_schema_out(scaler, t, features=("a", "b"))

        # Leaf estimators don't add prefixes
        assert "a" in schema.names
        assert "b" in schema.names

    def test_nested_column_transformer(self):
        """Test get_schema_out with nested ColumnTransformer uses nested prefixes."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

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

    def test_pipeline_with_column_transformer(self):
        """Test get_schema_out with Pipeline containing ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

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

    def test_unsupported_type_raises(self):
        """Test get_schema_out raises TypeError for non-BaseEstimator types."""
        from xorq.expr.ml.structer import get_schema_out

        t = xo.memtable({"a": [1.0, 2.0]})

        with pytest.raises(TypeError, match="Unexpected type in get_structer_out"):
            get_schema_out("invalid", t)

    def test_kv_encoded_deeply_nested_pipeline(self):
        """Test get_schema_out with hybrid output - known-schema + KV-encoded columns.

        Pipeline structure:
        - ColumnTransformer (hybrid output)
          - FeatureUnion (known-schema: numeric features)
            - Pipeline (SimpleImputer -> StandardScaler)
            - Pipeline (SimpleImputer -> StandardScaler)
          - Pipeline (KV-encoded: categorical via OneHotEncoder)
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import FeatureUnion
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.structer import KV_ENCODED_TYPE, get_schema_out

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

    def test_non_kv_deeply_nested_pipeline(self):
        """Test get_schema_out with depth-4 nested pipeline with all known-schema transformers.

        Pipeline structure:
        - ColumnTransformer (known schema - no KV-encoded children)
          - Pipeline (SimpleImputer -> StandardScaler -> Pipeline)
            - Pipeline (SimpleImputer -> StandardScaler)
          - Pipeline (SimpleImputer -> StandardScaler)
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.structer import get_schema_out

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

    def test_multiple_kv_encoded_columns(self):
        """Test that multiple KV-encoded transformers produce separate KV columns.

        Two OneHotEncoders should produce two separate KV-encoded columns,
        not pollute the entire output into one KV column.
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.structer import KV_ENCODED_TYPE, get_schema_out

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

    def test_hybrid_output_preserves_types(self):
        """Test that hybrid output preserves correct types for each column."""
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        import xorq.expr.datatypes as dt
        from xorq.expr.ml.structer import KV_ENCODED_TYPE, get_schema_out

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


class TestStructerHasKvOutput:
    """Tests for has_kv_output property combining is_kv_encoded and struct_has_kv_fields."""

    def test_has_kv_output_leaf_kv_encoded(self):
        """Test has_kv_output is True for leaf KV-encoded (struct is None)."""
        structer = Structer.kv_encoded(input_columns=("x",))
        assert structer.is_kv_encoded is True
        assert structer.struct_has_kv_fields is False
        assert structer.has_kv_output is True

    def test_has_kv_output_known_schema(self):
        """Test has_kv_output is False for known-schema (no KV)."""
        structer = Structer.from_names_typ(("a", "b"), dt.float64)
        assert structer.is_kv_encoded is False
        assert structer.struct_has_kv_fields is False
        assert structer.has_kv_output is False

    def test_has_kv_output_struct_with_kv_fields(self):
        """Test has_kv_output is True for struct containing KV fields."""
        struct = dt.Struct(
            {
                "numeric": dt.float64,
                "encoded": KV_ENCODED_TYPE,
            }
        )
        structer = Structer(struct=struct, input_columns=("cat",))
        assert structer.is_kv_encoded is False
        assert structer.struct_has_kv_fields is True
        assert structer.has_kv_output is True

    def test_has_kv_output_struct_without_kv_fields(self):
        """Test has_kv_output is False for struct without KV fields."""
        struct = dt.Struct({"a": dt.float64, "b": dt.int64})
        structer = Structer(struct=struct)
        assert structer.is_kv_encoded is False
        assert structer.struct_has_kv_fields is False
        assert structer.has_kv_output is False

    def test_is_kv_encoded_and_struct_has_kv_fields_mutually_exclusive(self):
        """Test is_kv_encoded and struct_has_kv_fields cannot both be True."""
        # When struct is None, is_kv_encoded=True, struct_has_kv_fields=False
        kv_structer = Structer.kv_encoded(input_columns=("x",))
        assert kv_structer.is_kv_encoded is True
        assert kv_structer.struct_has_kv_fields is False

        # When struct exists with KV fields, is_kv_encoded=False, struct_has_kv_fields=True
        hybrid_struct = dt.Struct({"enc": KV_ENCODED_TYPE})
        hybrid_structer = Structer(struct=hybrid_struct, input_columns=("x",))
        assert hybrid_structer.is_kv_encoded is False
        assert hybrid_structer.struct_has_kv_fields is True


class TestMakeKvTuple:
    """Tests for KVEncoder._make_kv_tuple converting names/values to KV format."""

    def test_make_kv_tuple_basic(self):
        """Test _make_kv_tuple creates correct structure."""
        names = ("a", "b", "c")
        values = (1.0, 2.0, 3.0)
        result = KVEncoder._make_kv_tuple(names, values)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == {KVField.KEY: "a", KVField.VALUE: 1.0}
        assert result[1] == {KVField.KEY: "b", KVField.VALUE: 2.0}
        assert result[2] == {KVField.KEY: "c", KVField.VALUE: 3.0}

    def test_make_kv_tuple_converts_to_float(self):
        """Test _make_kv_tuple converts values to float."""
        names = ("x",)
        values = (42,)  # int, not float
        result = KVEncoder._make_kv_tuple(names, values)

        assert result[0][KVField.VALUE] == 42.0
        assert isinstance(result[0][KVField.VALUE], float)

    def test_make_kv_tuple_empty(self):
        """Test _make_kv_tuple with empty inputs."""
        result = KVEncoder._make_kv_tuple((), ())
        assert result == ()

    def test_make_kv_tuple_single_element(self):
        """Test _make_kv_tuple with single element."""
        result = KVEncoder._make_kv_tuple(("only",), (99.9,))
        assert result == ({"key": "only", "value": 99.9},)

    def test_make_kv_tuple_preserves_order(self):
        """Test _make_kv_tuple preserves element order."""
        names = ("z", "a", "m")
        values = (3.0, 1.0, 2.0)
        result = KVEncoder._make_kv_tuple(names, values)

        assert tuple(d["key"] for d in result) == ("z", "a", "m")
        assert tuple(d["value"] for d in result) == (3.0, 1.0, 2.0)


class TestPrefixFields:
    """Tests for _prefix_fields adding sklearn-style 'name__' prefixes to field names."""

    def test_prefix_fields_basic(self):
        """Test _prefix_fields adds prefix to all field names."""
        fields = {"a": dt.float64, "b": dt.int64}
        result = _prefix_fields(fields, "scaler")

        assert "scaler__a" in result
        assert "scaler__b" in result
        assert result["scaler__a"] == dt.float64
        assert result["scaler__b"] == dt.int64

    def test_prefix_fields_empty(self):
        """Test _prefix_fields with empty fields dict."""
        result = _prefix_fields({}, "prefix")
        assert result == {}

    def test_prefix_fields_single(self):
        """Test _prefix_fields with single field."""
        result = _prefix_fields({"col": dt.string}, "pass")
        assert result == {"pass__col": dt.string}

    def test_prefix_fields_preserves_types(self):
        """Test _prefix_fields preserves all field types correctly."""
        fields = {
            "float_col": dt.float64,
            "int_col": dt.int64,
            "str_col": dt.string,
            "kv_col": KV_ENCODED_TYPE,
        }
        result = _prefix_fields(fields, "test")

        assert result["test__float_col"] == dt.float64
        assert result["test__int_col"] == dt.int64
        assert result["test__str_col"] == dt.string
        assert result["test__kv_col"] == KV_ENCODED_TYPE

    def test_prefix_fields_does_not_mutate_input(self):
        """Test _prefix_fields returns new dict, doesn't mutate input."""
        original = {"a": dt.float64}
        original_copy = dict(original)
        _prefix_fields(original, "prefix")
        assert original == original_copy


class TestProcessChildStructer:
    """Tests for _process_child_structer handling KV vs known-schema children."""

    def test_process_child_structer_kv_encoded(self):
        """Test _process_child_structer with KV-encoded child."""
        child = Structer.kv_encoded(input_columns=("cat",))
        fields, kv_cols = _process_child_structer(child, "encoder", ("cat",))

        assert fields == {"encoder": KV_ENCODED_TYPE}
        assert kv_cols == ("cat",)

    def test_process_child_structer_known_schema(self):
        """Test _process_child_structer with known-schema child."""
        child = Structer.from_names_typ(("a", "b"), dt.float64)
        fields, kv_cols = _process_child_structer(child, "scaler", ("a", "b"))

        assert "scaler__a" in fields
        assert "scaler__b" in fields
        assert fields["scaler__a"] == dt.float64
        assert kv_cols is None

    def test_process_child_structer_hybrid(self):
        """Test _process_child_structer with hybrid child (struct with KV fields)."""
        struct = dt.Struct(
            {
                "numeric": dt.float64,
                "encoded": KV_ENCODED_TYPE,
            }
        )
        child = Structer(struct=struct, input_columns=("cat",))
        fields, kv_cols = _process_child_structer(child, "ct", ("num", "cat"))

        # Hybrid child also becomes KV column
        assert fields == {"ct": KV_ENCODED_TYPE}
        assert kv_cols == ("num", "cat")


class TestMergeCtResults:
    """Tests for _merge_ct_results accumulating schema fields and kv_input_cols."""

    def test_merge_ct_results_empty_acc(self):
        """Test _merge_ct_results with empty accumulator."""
        acc = ({}, [])
        result = ({"a": dt.float64}, None)
        new_acc = _merge_ct_results(acc, result)

        assert new_acc == ({"a": dt.float64}, [])

    def test_merge_ct_results_with_kv_cols(self):
        """Test _merge_ct_results accumulates kv_cols."""
        acc = ({"a": dt.float64}, ["x"])
        result = ({"b": KV_ENCODED_TYPE}, ("y", "z"))
        new_acc = _merge_ct_results(acc, result)

        assert new_acc[0] == {"a": dt.float64, "b": KV_ENCODED_TYPE}
        assert new_acc[1] == ["x", "y", "z"]

    def test_merge_ct_results_none_kv_cols(self):
        """Test _merge_ct_results handles None kv_cols."""
        acc = ({"a": dt.float64}, ["x"])
        result = ({"b": dt.int64}, None)
        new_acc = _merge_ct_results(acc, result)

        assert new_acc[0] == {"a": dt.float64, "b": dt.int64}
        assert new_acc[1] == ["x"]

    def test_merge_ct_results_multiple_merges(self):
        """Test _merge_ct_results with chain of merges."""
        acc = ({}, [])
        results = [
            ({"a": dt.float64}, None),
            ({"b": KV_ENCODED_TYPE}, ("cat1",)),
            ({"c": dt.int64}, None),
            ({"d": KV_ENCODED_TYPE}, ("cat2",)),
        ]

        for result in results:
            acc = _merge_ct_results(acc, result)

        assert acc[0] == {
            "a": dt.float64,
            "b": KV_ENCODED_TYPE,
            "c": dt.int64,
            "d": KV_ENCODED_TYPE,
        }
        assert acc[1] == ["cat1", "cat2"]

    def test_merge_ct_results_empty_fields(self):
        """Test _merge_ct_results with empty fields (drop case)."""
        acc = ({"a": dt.float64}, [])
        result = ({}, None)  # drop transformer returns empty
        new_acc = _merge_ct_results(acc, result)

        assert new_acc == ({"a": dt.float64}, [])


class TestGetRemainderFields:
    """Tests for _get_remainder_fields finding unhandled columns for remainder='passthrough'."""

    def test_get_remainder_fields_basic(self):
        """Test _get_remainder_fields finds unhandled columns."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        transformer_items = [
            ("scaler", StandardScaler(), ["a"]),
        ]
        features = ("a", "b", "c")
        result = _get_remainder_fields(transformer_items, features, t)

        assert "b" in result
        assert "c" in result
        assert "a" not in result

    def test_get_remainder_fields_all_handled(self):
        """Test _get_remainder_fields when all columns are handled."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0], "b": [2.0]})
        transformer_items = [
            ("scaler", StandardScaler(), ["a", "b"]),
        ]
        features = ("a", "b")
        result = _get_remainder_fields(transformer_items, features, t)

        assert result == {}

    def test_get_remainder_fields_ignores_drop(self):
        """Test _get_remainder_fields ignores 'drop' transformer columns."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        transformer_items = [
            ("scaler", StandardScaler(), ["a"]),
            ("dropped", "drop", ["b"]),
        ]
        features = ("a", "b", "c")
        result = _get_remainder_fields(transformer_items, features, t)

        # b is handled by drop (excluded from remainder), c is remainder
        assert "c" in result
        assert "a" not in result
        assert "b" not in result

    def test_get_remainder_fields_preserves_types(self):
        """Test _get_remainder_fields preserves column types."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"num": [1.0], "cat": ["a"], "int_col": [42]})
        transformer_items = [
            ("scaler", StandardScaler(), ["num"]),
        ]
        features = ("num", "cat", "int_col")
        result = _get_remainder_fields(transformer_items, features, t)

        assert result["cat"] == dt.string
        assert result["int_col"] == dt.int64


class TestProcessCtItem:
    """Tests for _process_ct_item handling drop/passthrough/transformer items."""

    def test_process_ct_item_drop(self):
        """Test _process_ct_item with 'drop' transformer."""
        t = xo.memtable({"a": [1.0]})
        item = ("dropped", "drop", ["a"])
        fields, kv_cols = _process_ct_item(t, item)

        assert fields == {}
        assert kv_cols is None

    def test_process_ct_item_passthrough(self):
        """Test _process_ct_item with 'passthrough' transformer."""
        t = xo.memtable({"a": [1.0], "b": ["x"]})
        item = ("pass", "passthrough", ["a", "b"])
        fields, kv_cols = _process_ct_item(t, item)

        assert "pass__a" in fields
        assert "pass__b" in fields
        assert fields["pass__a"] == dt.float64
        assert fields["pass__b"] == dt.string
        assert kv_cols is None

    def test_process_ct_item_known_schema_transformer(self):
        """Test _process_ct_item with known-schema transformer."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0], "b": [2.0]})
        item = ("scaler", StandardScaler(), ["a", "b"])
        fields, kv_cols = _process_ct_item(t, item)

        assert "scaler__a" in fields
        assert "scaler__b" in fields
        assert kv_cols is None

    def test_process_ct_item_kv_encoded_transformer(self):
        """Test _process_ct_item with KV-encoded transformer."""
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b"]})
        item = ("encoder", OneHotEncoder(), ["cat"])
        fields, kv_cols = _process_ct_item(t, item)

        assert fields == {"encoder": KV_ENCODED_TYPE}
        assert kv_cols == ("cat",)

    def test_process_ct_item_normalizes_columns(self):
        """Test _process_ct_item normalizes column specs."""
        t = xo.memtable({"a": [1.0]})

        # String column
        item = ("pass1", "passthrough", "a")
        fields, _ = _process_ct_item(t, item)
        assert "pass1__a" in fields

        # List column
        item = ("pass2", "passthrough", ["a"])
        fields, _ = _process_ct_item(t, item)
        assert "pass2__a" in fields

        # Tuple column
        item = ("pass3", "passthrough", ("a",))
        fields, _ = _process_ct_item(t, item)
        assert "pass3__a" in fields


class TestProcessFuItem:
    """Tests for _process_fu_item processing FeatureUnion transformer items."""

    def test_process_fu_item_known_schema(self):
        """Test _process_fu_item with known-schema transformer."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0], "b": [2.0]})
        item = ("scaler", StandardScaler())
        fields, kv_cols = _process_fu_item(t, ("a", "b"), item)

        assert "scaler__a" in fields
        assert "scaler__b" in fields
        assert kv_cols is None

    def test_process_fu_item_kv_encoded(self):
        """Test _process_fu_item with KV-encoded transformer."""
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b"]})
        item = ("encoder", OneHotEncoder())
        fields, kv_cols = _process_fu_item(t, ("cat",), item)

        assert fields == {"encoder": KV_ENCODED_TYPE}
        assert kv_cols == ("cat",)

    def test_process_fu_item_uses_all_features(self):
        """Test _process_fu_item applies transformer to all features."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0], "b": [2.0], "c": [3.0]})
        item = ("scaler", StandardScaler())
        fields, _ = _process_fu_item(t, ("a", "b", "c"), item)

        assert "scaler__a" in fields
        assert "scaler__b" in fields
        assert "scaler__c" in fields


class TestAccumulatePipelineStep:
    """Tests for _accumulate_pipeline_step threading features through pipeline."""

    def test_accumulate_pipeline_step_passthrough(self):
        """Test _accumulate_pipeline_step with passthrough step."""
        t = xo.memtable({"a": [1.0]})
        acc = (("a",), None)
        step = ("pass", "passthrough")

        new_acc = _accumulate_pipeline_step(t, acc, step)

        # Passthrough returns acc unchanged
        assert new_acc == acc

    def test_accumulate_pipeline_step_known_schema(self):
        """Test _accumulate_pipeline_step with known-schema transformer."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0], "b": [2.0]})
        acc = (("a", "b"), None)
        step = ("scaler", StandardScaler())

        new_features, new_structer = _accumulate_pipeline_step(t, acc, step)

        assert new_features == ("a", "b")  # Same features for known-schema
        assert new_structer.struct is not None
        assert "a" in new_structer.struct.fields
        assert "b" in new_structer.struct.fields

    def test_accumulate_pipeline_step_kv_encoded(self):
        """Test _accumulate_pipeline_step with KV-encoded transformer."""
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b"]})
        acc = (("cat",), None)
        step = ("encoder", OneHotEncoder())

        new_features, new_structer = _accumulate_pipeline_step(t, acc, step)

        # Features unchanged when KV-encoded
        assert new_features == ("cat",)
        assert new_structer.is_kv_encoded is True

    def test_accumulate_pipeline_step_updates_features(self):
        """Test _accumulate_pipeline_step with known output count updates features."""
        from sklearn.feature_selection import SelectKBest

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "target": [0, 1]})
        acc = (("a", "b"), None)
        step = ("selector", SelectKBest(k=1))

        new_features, new_structer = _accumulate_pipeline_step(t, acc, step)

        # SelectKBest with known k produces stub column names
        assert new_features == ("transformed_0",)
        assert not new_structer.is_kv_encoded
        assert new_structer.needs_target

    def test_accumulate_pipeline_step_chain(self):
        """Test _accumulate_pipeline_step through multiple steps."""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0], "b": [2.0]})
        features = ("a", "b")

        # Chain: imputer -> scaler
        acc = (features, None)
        acc = _accumulate_pipeline_step(t, acc, ("imputer", SimpleImputer()))
        acc = _accumulate_pipeline_step(t, acc, ("scaler", StandardScaler()))

        final_features, final_structer = acc
        assert final_features == ("a", "b")
        assert final_structer.struct is not None


class TestNormalizeColumns:
    """Tests for _normalize_columns converting column specs to tuples."""

    def test_normalize_columns_list(self):
        """Test _normalize_columns converts list to tuple."""
        assert _normalize_columns(["a", "b"]) == ("a", "b")

    def test_normalize_columns_string(self):
        """Test _normalize_columns converts string to single-element tuple."""
        assert _normalize_columns("col") == ("col",)

    def test_normalize_columns_tuple(self):
        """Test _normalize_columns passes through tuple."""
        assert _normalize_columns(("a", "b")) == ("a", "b")

    def test_normalize_columns_none(self):
        """Test _normalize_columns converts None to empty tuple."""
        assert _normalize_columns(None) == ()

    def test_normalize_columns_empty_list(self):
        """Test _normalize_columns handles empty list."""
        assert _normalize_columns([]) == ()

    def test_normalize_columns_invalid_raises(self):
        """Test _normalize_columns raises for unsupported types."""
        with pytest.raises(TypeError, match="Unsupported columns type"):
            _normalize_columns(123)
        with pytest.raises(TypeError, match="Unsupported columns type"):
            _normalize_columns({"a": 1})


class TestConvertStructWithKv:
    """Tests for Structer.convert_struct_with_kv column slicing.

    These tests verify the accumulate_sklearn_col_slice logic correctly
    computes column indices for hybrid output (KV + non-KV fields).
    """

    def test_kv_then_non_kv_slicing(self):
        """Test slicing when KV field precedes non-KV field.

        This catches a potential bug where end = start + (start + 1) instead of
        end = start + 1 for non-KV fields. With incorrect logic:
        - cat field: start=0, end=0+2=2 (correct)
        - num field: start=2, end=2+(2+1)=5 (WRONG, should be 3)

        The non-KV field would get wrong values or cause index errors.
        """
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.structer import get_structer_out

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

    def test_non_kv_then_kv_slicing(self):
        """Test slicing when non-KV field precedes KV field."""
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.structer import get_structer_out

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

    def test_multiple_kv_fields_slicing(self):
        """Test slicing with multiple KV fields."""
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.structer import get_structer_out

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


class TestFeatureSelectorParity:
    """Parameterized tests for feature selector sklearn parity."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification data with features and target."""
        import numpy as np

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
                {"alpha": 0.5},
                id="SelectFpr",
            ),
            pytest.param(
                "SelectFdr",
                {"alpha": 0.5},
                id="SelectFdr",
            ),
            pytest.param(
                "SelectFwe",
                {"alpha": 0.5},
                id="SelectFwe",
            ),
            pytest.param(
                "GenericUnivariateSelect",
                {"mode": "k_best", "param": 2},
                id="GenericUnivariateSelect",
            ),
        ],
    )
    def test_univariate_selector_matches_sklearn(
        self, classification_data, selector_cls, selector_kwargs
    ):
        """Test univariate feature selectors match sklearn output."""
        from sklearn import feature_selection

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
        import numpy as np

        np.testing.assert_allclose(
            xorq_df.reset_index(drop=True).values,
            sklearn_df.reset_index(drop=True).values,
            atol=1e-10,
        )

    def test_variance_threshold_matches_sklearn(self, classification_data):
        """Test VarianceThreshold (unsupervised selector) matches sklearn output.

        VarianceThreshold is unique among feature selectors - it doesn't require
        a target variable, making it an unsupervised feature selector.
        """
        import numpy as np
        from sklearn.feature_selection import VarianceThreshold

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
    def test_model_based_selector_matches_sklearn(
        self, classification_data, selector_cls, selector_kwargs
    ):
        """Test model-based feature selectors match sklearn output."""
        from sklearn import feature_selection
        from sklearn.linear_model import LogisticRegression

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
        import numpy as np

        np.testing.assert_allclose(
            xorq_df.reset_index(drop=True).values,
            sklearn_df.reset_index(drop=True).values,
            atol=1e-10,
        )


class TestClassNamePrefixFeaturesOutMixinParity:
    """Parameterized tests for all ClassNamePrefixFeaturesOutMixin sklearn parity.

    This covers decomposition, kernel approximation, manifold learning,
    random projections, discriminant analysis, and neighborhood components.
    """

    @pytest.fixture
    def numeric_data(self):
        """Generate numeric data for transformers."""
        import numpy as np

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
        "transformer_cls,transformer_kwargs,module,needs_target",
        [
            # Decomposition - basic
            pytest.param(
                "PCA",
                {"n_components": 2},
                "sklearn.decomposition",
                False,
                id="PCA",
            ),
            pytest.param(
                "TruncatedSVD",
                {"n_components": 2},
                "sklearn.decomposition",
                False,
                id="TruncatedSVD",
            ),
            pytest.param(
                "NMF",
                {"n_components": 2, "max_iter": 500, "random_state": 42},
                "sklearn.decomposition",
                False,
                id="NMF",
            ),
            pytest.param(
                "FastICA",
                {"n_components": 2, "max_iter": 500, "random_state": 42},
                "sklearn.decomposition",
                False,
                id="FastICA",
            ),
            pytest.param(
                "FactorAnalysis",
                {"n_components": 2, "random_state": 42},
                "sklearn.decomposition",
                False,
                id="FactorAnalysis",
            ),
            # Decomposition - additional
            pytest.param(
                "KernelPCA",
                {"n_components": 2, "kernel": "rbf"},
                "sklearn.decomposition",
                False,
                id="KernelPCA",
            ),
            pytest.param(
                "IncrementalPCA",
                {"n_components": 2},
                "sklearn.decomposition",
                False,
                id="IncrementalPCA",
            ),
            pytest.param(
                "MiniBatchNMF",
                {"n_components": 2, "max_iter": 500, "random_state": 42},
                "sklearn.decomposition",
                False,
                id="MiniBatchNMF",
            ),
            # Random Projection
            pytest.param(
                "GaussianRandomProjection",
                {"n_components": 2, "random_state": 42},
                "sklearn.random_projection",
                False,
                id="GaussianRandomProjection",
            ),
            pytest.param(
                "SparseRandomProjection",
                {"n_components": 2, "random_state": 42},
                "sklearn.random_projection",
                False,
                id="SparseRandomProjection",
            ),
            # Manifold Learning
            pytest.param(
                "Isomap",
                {"n_components": 2, "n_neighbors": 5},
                "sklearn.manifold",
                False,
                id="Isomap",
            ),
            pytest.param(
                "LocallyLinearEmbedding",
                {"n_components": 2, "n_neighbors": 5, "random_state": 42},
                "sklearn.manifold",
                False,
                id="LocallyLinearEmbedding",
            ),
            # Discriminant Analysis (supervised)
            pytest.param(
                "LinearDiscriminantAnalysis",
                {"n_components": 2},
                "sklearn.discriminant_analysis",
                True,
                id="LinearDiscriminantAnalysis",
            ),
            # Neighborhood Components Analysis (supervised)
            pytest.param(
                "NeighborhoodComponentsAnalysis",
                {"n_components": 2, "max_iter": 100, "random_state": 42},
                "sklearn.neighbors",
                True,
                id="NeighborhoodComponentsAnalysis",
            ),
        ],
    )
    def test_classname_prefix_transformer_matches_sklearn(
        self, numeric_data, transformer_cls, transformer_kwargs, module, needs_target
    ):
        """Test ClassNamePrefixFeaturesOutMixin transformers match sklearn output."""
        import importlib

        import numpy as np

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

        # Compare values (use larger tolerance for iterative algorithms like
        # MiniBatchNMF, LocallyLinearEmbedding which can have numerical differences)
        np.testing.assert_allclose(
            xorq_df.reset_index(drop=True).values,
            sklearn_df.reset_index(drop=True).values,
            atol=1e-2,
        )

        # Also verify column names match sklearn's get_feature_names_out()
        assert list(xorq_df.columns) == list(sklearn_df.columns)


class TestKVEncodedTransformersParity:
    """Comprehensive tests for all KV-encoded transformers matching sklearn output."""

    @pytest.fixture
    def test_data(self):
        """Create test data with all required column types."""
        import numpy as np

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
        ],
    )
    def test_kv_encoded_transformer_matches_sklearn(
        self,
        test_data,
        transformer_cls,
        transformer_kwargs,
        module,
        features,
        target_col,
        input_type,
    ):
        """Test KV-encoded transformers match sklearn output."""
        import importlib

        import numpy as np

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
